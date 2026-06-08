from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch

from utils.logger_utils import get_logger
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
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
        self._post_processor: Any | None = None
        self._hypes: dict[str, Any] | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._state_lock = threading.Lock()
        self._load_condition = threading.Condition(self._state_lock)
        self._is_loading = False
        self._model_lock = threading.Lock()
        self._preprocess_lock = threading.Lock()
        self._post_processor_lock = threading.Lock()

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
            post_prcessor = build_postprocessor(loaded_hypes["postprocess"], train=False)

            with self._load_condition:
                if self._model is None:
                    self._hypes = loaded_hypes
                    self._model = model
                    self._pre_processor = pre_processor
                    self._post_processor = post_prcessor
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise e
        finally:
            with self._load_condition:
                self._is_loading = False
                self._load_condition.notify_all()

    def process_intermediate_features(self, intermediate_features: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model has not been loaded yet")

        fused_feature = None

        with self._model_lock:
            with torch.no_grad():
                ego_feature = self._model.fusion_net.fuse_features([intermediate_features[0]], backbone=self._model.backbone)
                if len(intermediate_features) > 1:
                    fused_feature = self._model.fusion_net.fuse_features(intermediate_features, backbone=self._model.backbone)

                if self._model.shrink_flag:
                    ego_feature = self._model.shrink_conv(ego_feature)
                    if len(intermediate_features) > 1:
                        fused_feature = self._model.shrink_conv(fused_feature)

                ego_psm = self._model.cls_head(ego_feature)
                ego_rm = self._model.reg_head(ego_feature)

                if len(intermediate_features) > 1:
                    fused_psm = self._model.cls_head(fused_feature)
                    fused_rm = self._model.reg_head(fused_feature)

                ego_feature = ego_feature.cpu().data.numpy()
                if len(intermediate_features) > 1:
                    fused_feature = fused_feature.cpu().data.numpy()

                output_dict = {'ego': {'psm': ego_psm,
                                       'rm': ego_rm,
                                       'feature': ego_feature.copy(),
                                       }}
                
                if len(intermediate_features) > 1:
                    output_dict['fused'] = {'psm': fused_psm,
                                            'rm': fused_rm,
                                            'feature': fused_feature.copy(),}

        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix = torch.from_numpy(transformation_matrix)

        with self._post_processor_lock:
            anchor_box = self._post_processor.generate_anchor_box()
        anchor_box = torch.from_numpy(anchor_box)

        batch_data = {'ego': {
            'transformation_matrix': transformation_matrix,
            'anchor_box': anchor_box
        }}

        if len(intermediate_features) > 1:
            batch_data['fused'] = batch_data['ego']

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self._device)

        with self._post_processor_lock:
            ego_pred_box_tensor, _ = self._post_processor.post_process({'ego': batch_data['ego']}, {'ego': output_dict['ego']})
            if len(intermediate_features) > 1:
                fused_pred_box_tensor, _ = self._post_processor.post_process({'fused': batch_data['fused']},
                                                                            {'fused': output_dict['fused']})
        if ego_pred_box_tensor is not None:
            ego_pred_box = ego_pred_box_tensor.cpu().data.numpy()
        else:
            ego_pred_box = np.array([])

        if len(intermediate_features) > 1 and fused_pred_box_tensor is not None:
            fused_pred_box = fused_pred_box_tensor.cpu().data.numpy()
        else:
            fused_pred_box = np.array([])

        return ego_pred_box, fused_pred_box, ego_feature, fused_feature

model_runtime = ModelRuntime()