from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch

from utils.logger_utils import get_logger
from utils.inference_utils import lidar_pose_to_projected_spatial_feature, spatial_feature_to_intermediate_feature, pointcloud_to_spatial_feature
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils

logger = get_logger(__name__)


class ModelRuntime:
    """线程安全的模型运行时.

    设计目标:
    1) 模型只加载一次.
    2) 降低加载期间锁竞争.
    3) 推理阶段按功能拆分锁粒度.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._pre_processor: Any | None = None
        self._hypes: dict[str, Any] | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._state_lock = threading.Lock()
        self._load_condition = threading.Condition(self._state_lock)
        self._is_loading = False
        self._model_lock = threading.Lock()
        self._preprocess_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def hypes(self) -> dict[str, Any]:
        if self._hypes is None:
            raise RuntimeError("Model config has not been loaded yet")
        return self._hypes

    def load_model(self, checkpoint_path: str) -> None:
        """加载模型权重与预处理器.

        行为说明:
        1) 幂等: 已加载则直接返回.
        2) 并发安全: 仅允许一个线程执行加载, 其余线程等待加载结束.
        3) 细粒度锁: 耗时加载在状态锁外执行, 仅在状态切换时持锁.
        """
        if self._model is not None:
            return

        with self._load_condition:
            if self._model is not None:
                return

            while self._is_loading:
                self._load_condition.wait()
                if self._model is not None:
                    return

            self._is_loading = True

        try:
            config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config not found: {config_path}")

            loaded_hypes = load_yaml(config_path)

            where2comm_cfg = loaded_hypes.get("model", {}).get("args", {}).get("where2comm_fusion")
            if isinstance(where2comm_cfg, dict):
                communication_cfg = where2comm_cfg.get("communication")
                if isinstance(communication_cfg, dict) and "threshold" in communication_cfg:
                    logger.info(
                        "Updating communication threshold from %s to %s",
                        communication_cfg["threshold"],
                        0.02,
                    )
                    communication_cfg["threshold"] = 0.02

            logger.info('Creating Model')
            model = train_utils.create_model(loaded_hypes)
            if self._device == "cuda":
                model = model.cuda()

            logger.info('Loading Model from checkpoint')
            _, model = train_utils.load_saved_model(checkpoint_path, model)
            model.eval()

            pre_processor = build_preprocessor(loaded_hypes["preprocess"], train=False)

            with self._load_condition:
                if self._model is None:
                    self._hypes = loaded_hypes
                    self._model = model
                    self._pre_processor = pre_processor
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise e
        finally:
            with self._load_condition:
                self._is_loading = False
                self._load_condition.notify_all()

    def pointcloud_inference(
        self,
        pointcloud: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """执行单车点云推理并输出中间通信特征.

        参数
        ----
        pointcloud : np.ndarray
            输入点云, 通常为 (N, 4).

        返回
        ----
        intermediate_feature : dict[str, np.ndarray]
            中间特征字典, 包含:
            - feature: 通信特征
            - mask: 通信掩码

        说明
        ----
        该接口不做跨车位姿投影, 直接基于当前点云生成空间特征并提取中间通信特征.
        """
        assert self._model is not None
        assert self._pre_processor is not None

        _, spatial_feature = pointcloud_to_spatial_feature(
            pointcloud=pointcloud,
            hypes=self.hypes,
            pre_processor=self._pre_processor,
            model=self._model,
            device=self._device,
            preprocess_lock=self._preprocess_lock,
            model_lock=self._model_lock,
        )
        intermediate_feature = spatial_feature_to_intermediate_feature(
            spatial_feature=spatial_feature,
            model=self._model,
            device=self._device,
            model_lock=self._model_lock,
        )
        return intermediate_feature

    def projected_pointcloud_inference(
        self,
        pointcloud: np.ndarray,
        lidar_pose: np.ndarray,
        target_pose: np.ndarray,
        target_request_map: np.ndarray,
        *,
        gps: bool = False,
    ) -> dict[str, np.ndarray]:
        """执行点云到中间通信特征的推理流程.

        参数
        ----
        pointcloud : np.ndarray
            源车点云.
        lidar_pose : np.ndarray
            源车位姿.
        target_pose : np.ndarray
            目标车位姿.
        target_request_map : np.ndarray
            目标请求图.
        gps : bool, optional
            是否按 GPS 模式做位姿变换.

        返回
        ----
        intermediate_feature : dict[str, np.ndarray]
            中间特征字典, 包含:
            - feature: 通信特征
            - mask: 通信掩码
        """
        assert self._model is not None
        assert self._pre_processor is not None

        projected_spatial_feature = lidar_pose_to_projected_spatial_feature(
            source_lidar_pose=lidar_pose,
            source_pointcloud=pointcloud,
            target_lidar_pose=target_pose,
            hypes=self.hypes,
            pre_processor=self._pre_processor,
            model=self._model,
            device=self._device,
            preprocess_lock=self._preprocess_lock,
            model_lock=self._model_lock,
            gps=gps,
        )
        intermediate_feature = spatial_feature_to_intermediate_feature(
            spatial_feature=projected_spatial_feature,
            model=self._model,
            device=self._device,
            request_map=target_request_map,
            model_lock=self._model_lock,
        )
        return intermediate_feature

model_runtime = ModelRuntime()