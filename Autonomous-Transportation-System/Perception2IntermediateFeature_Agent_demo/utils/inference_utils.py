from __future__ import annotations

from threading import Lock
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from opencood.tools import train_utils
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range
from opencood.utils.transformation_utils import gps_to_utm_transformation, x1_to_x2

def project_pointcloud(pointcloud: np.ndarray, lidar_pose: np.ndarray, target_pose: np.ndarray, hypes: dict, gps: bool = False) -> np.ndarray:
    """
    将源点云从 source 坐标系投影到 target 坐标系, 并按检测范围裁剪.

    参数
    ----
    pointcloud : np.ndarray
        输入点云, 形状通常为 (N, 4), 前三列为 xyz.
    lidar_pose : np.ndarray
        源激光雷达位姿. GPS 模式下可为经纬高, 非 GPS 模式下为位姿表示.
    target_pose : np.ndarray
        目标激光雷达位姿.
    hypes : dict
        模型配置字典, 需包含 preprocess.cav_lidar_range.
    gps : bool, optional
        是否使用 GPS 到 UTM 的转换流程.

    返回
    ----
    projected_pointcloud : np.ndarray
        投影并裁剪后的点云, 与输入 dtype 保持一致.
    """
    projected_pointcloud = pointcloud.copy()

    # projected_pointcloud = shuffle_points(projected_pointcloud)
    projected_pointcloud = mask_ego_points(projected_pointcloud)

    if gps:
        transformation_matrix = gps_to_utm_transformation(lidar_pose, target_pose)
    else:
        transformation_matrix = x1_to_x2(lidar_pose, target_pose)

    projected_pointcloud[:, :3] = box_utils.project_points_by_matrix_torch(projected_pointcloud[:, :3], transformation_matrix)
    projected_pointcloud = mask_points_by_range(projected_pointcloud, hypes['preprocess']['cav_lidar_range'])

    return projected_pointcloud

def voxel_to_spatial_feature(
    voxel: dict,
    model: Any,
    device: Union[str, torch.device],
    model_lock: Lock,
) -> torch.Tensor:
    """
    将体素化结果编码为 BEV 空间特征.

    参数
    ----
    voxel : dict
        体素化后的字典, 必须包含:
        - voxel_features
        - voxel_coords
        - voxel_num_points
    model : Any
        已加载的协同感知模型, 需包含 pillar_vfe 与 scatter 模块.
    device : Union[str, torch.device]
        推理设备.
    model_lock : Lock
        模型前向锁, 用于串行化模型内部非线程安全状态.

    返回
    ----
    spatial_feature : torch.Tensor
        空间特征张量, 常见形状为 (N, C, H, W).
        在 Where2comm 场景通常为 (1, 64, 192, 704).
    """
    voxel_features = torch.from_numpy(voxel['voxel_features'])

    voxel_coords = np.pad(voxel['voxel_coords'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxel_coords = torch.from_numpy(voxel_coords)

    voxel_num_points = torch.from_numpy(voxel['voxel_num_points'])

    record_len = torch.empty(1, dtype=torch.int32)

    batch_dict = {'voxel_features': voxel_features,
                  'voxel_coords': voxel_coords,
                  'voxel_num_points': voxel_num_points,
                  'record_len': record_len}

    with torch.no_grad():
        batch_dict = train_utils.to_device(batch_dict, device)

        with model_lock:
            # pillar_vfe: n x 4 -> n x c
            batch_dict = model.pillar_vfe(batch_dict)
            # scatter: n x c -> N x C x H x W
            batch_dict = model.scatter(batch_dict)

    spatial_feature = batch_dict['spatial_features']
    return spatial_feature

def process_pointcloud(pointcloud: np.ndarray, hypes: dict) -> np.ndarray:
    """
    对原始点云做基础预处理（按感知范围裁剪）.

    参数
    ----
    pointcloud : np.ndarray
        原始点云, 形状通常为 (N, 4), 包含 (x, y, z, intensity).
    hypes : dict
        配置字典, 必须包含 preprocess.cav_lidar_range.

    返回
    ----
    processed_pointcloud : np.ndarray
        过滤范围外点后的点云, 形状为 (M, 4), 其中 M <= N.
    """
    processed_pointcloud = mask_points_by_range(pointcloud, hypes["preprocess"]["cav_lidar_range"])
    return processed_pointcloud

def pointcloud_to_spatial_feature(
    pointcloud: np.ndarray,
    hypes: dict,
    pre_processor: Any,
    model: Any,
    device: Union[str, torch.device],
    preprocess_lock: Lock,
    model_lock: Lock,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将原始点云转换为空间特征, 返回中间处理结果和最终特征.

    参数
    ----
    pointcloud : np.ndarray
        原始点云.
    hypes : dict
        配置字典, 用于点云裁剪.
    pre_processor : Any
        预处理器实例.
    model : Any
        已加载模型.
    device : Union[str, torch.device]
        推理设备.
    preprocess_lock : Lock
        预处理阶段锁.
    model_lock : Lock
        模型前向阶段锁.

    返回
    ----
    processed_pointcloud : np.ndarray
        经过范围裁剪后的点云.
    spatial_feature : np.ndarray
        对应的 BEV 空间特征数组.
    """
    processed_pointcloud = process_pointcloud(pointcloud, hypes)
    spatial_feature = processed_pointcloud_to_spatial_feature(
        processed_pointcloud,
        pre_processor,
        model,
        device,
        preprocess_lock,
        model_lock,
    )
    return processed_pointcloud, spatial_feature

def processed_pointcloud_to_spatial_feature(
    processed_pointcloud: np.ndarray,
    pre_processor: Any,
    model: Any,
    device: Union[str, torch.device],
    preprocess_lock: Lock,
    model_lock: Lock,
) -> np.ndarray:
    """
    将处理后的点云数据转为 Numpy 格式的 BEV 空间特征.

    参数
    ----
    processed_pointcloud : np.ndarray
        输入点云数组.
    pre_processor : Any
        预处理器实例, 需提供 preprocess 方法.
    model : Any
        已加载模型.
    device : Union[str, torch.device]
        推理设备.
    preprocess_lock : Lock
        预处理阶段锁.
    model_lock : Lock
        模型前向阶段锁.

    返回
    ----
    spatial_feature : np.ndarray
        CPU 上的 Numpy 特征数组, 形状通常为 (N, C, H, W).
    """
    with preprocess_lock:
        voxel = pre_processor.preprocess(processed_pointcloud)

    spatial_feature_tensor = voxel_to_spatial_feature(voxel, model, device, model_lock)
    spatial_feature = spatial_feature_tensor.cpu().data.numpy()
    return spatial_feature


def lidar_pose_to_projected_spatial_feature(
    source_lidar_pose: np.ndarray,
    source_pointcloud: np.ndarray,
    target_lidar_pose: np.ndarray,
    hypes: dict,
    pre_processor: Any,
    model: Any,
    device: Union[str, torch.device],
    preprocess_lock: Lock,
    model_lock: Lock,
    gps: bool = False,
) -> np.ndarray:
    """
    从源车点云与位姿出发, 生成目标车坐标系下的空间特征.

    参数
    ----
    source_lidar_pose : np.ndarray
        源车激光雷达位姿.
    source_pointcloud : np.ndarray
        源车点云.
    target_lidar_pose : np.ndarray
        目标车激光雷达位姿.
    hypes : dict
        配置字典.
    pre_processor : Any
        预处理器实例.
    model : Any
        已加载模型.
    device : Union[str, torch.device]
        推理设备.
    preprocess_lock : Lock
        预处理阶段锁.
    model_lock : Lock
        模型前向阶段锁.
    gps : bool, optional
        是否使用 GPS 坐标变换.

    返回
    ----
    np.ndarray
        目标坐标系下的空间特征.
    """
    projected_pointcloud = project_pointcloud(
        source_pointcloud,
        source_lidar_pose,
        target_lidar_pose,
        hypes,
        gps,
    )
    projected_spatial_feature = processed_pointcloud_to_spatial_feature(
        projected_pointcloud,
        pre_processor,
        model,
        device,
        preprocess_lock,
        model_lock,
    )

    return projected_spatial_feature

def spatial_feature_to_intermediate_feature(
    spatial_feature: Union[np.ndarray, torch.Tensor],
    model: Any,
    device: Union[str, torch.device],
    model_lock: Lock,
    request_map: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    根据空间特征与请求图生成中间通信特征和通信掩码.

    参数
    ----
    spatial_feature : np.ndarray or torch.Tensor
        空间特征输入. 当为 Numpy 时会先转为 Tensor.
    model : Any
        已加载模型, 需包含 backbone / cls_head / fusion_net.
    device : Union[str, torch.device]
        推理设备.
    model_lock : Lock
        模型前向阶段锁.
    request_map : np.ndarray, optional
        请求图, 用于指导通信区域筛选.

    返回
    ----
    intermediate_feature : dict[str, np.ndarray]
        中间特征字典, 包含两个键:
        - feature: 掩码后的通信特征, 空结果时为 np.array([])
        - mask: 通信掩码, 空结果时为 np.array([])
    """
    if isinstance(spatial_feature, np.ndarray):
        spatial_feature = torch.from_numpy(spatial_feature).to(device)

    if request_map is not None:
        request_map = torch.from_numpy(request_map.copy()).to(device)

    # record_len = torch.tensor([spatial_feature.shape[0]], dtype=torch.int32).to(device)
    # pairwise_t_matrix = torch.zeros((1, 5, 5, 4, 4), dtype=torch.float64).to(device)

    with model_lock:
        with torch.no_grad():
            spatial_features_2d = model.backbone({'spatial_features': spatial_feature})['spatial_features_2d']

            if model.shrink_flag:
                spatial_features_2d = model.shrink_conv(spatial_features_2d)

            psm_single = model.cls_head(spatial_features_2d)

            if model.compression:
                # The ego feature is also compressed
                spatial_features_2d = model.naive_compressor(spatial_features_2d)

            if model.multi_scale:
                # Bypass communication cost, communicate at high resolution, neither shrink nor compress
                comm_masked_feature_tensor, comm_mask_tensor = model.fusion_net.spatial_feature_to_comm_masked_feature(
                    spatial_feature,
                    psm_single,
                    # record_len,
                    # pairwise_t_matrix,
                    model.backbone,
                    request_map=request_map)
            else:
                comm_masked_feature_tensor, comm_mask_tensor = model.fusion_net.spatial_feature_to_comm_masked_feature(
                    spatial_features_2d,
                    psm_single,
                    # record_len,
                    # pairwise_t_matrix,
                    request_map=request_map)
    if comm_masked_feature_tensor is not None:
        comm_masked_feature = comm_masked_feature_tensor.cpu().data.numpy()
    else:
        comm_masked_feature = np.array([])

    if comm_mask_tensor is not None:
        comm_mask = comm_mask_tensor.cpu().data.numpy()
    else:
        comm_mask = np.array([])

    intermediate_feature = {"feature": comm_masked_feature, "mask": comm_mask}

    return intermediate_feature