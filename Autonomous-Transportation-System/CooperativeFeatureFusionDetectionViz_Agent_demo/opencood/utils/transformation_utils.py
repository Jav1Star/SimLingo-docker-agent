# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import math
import numpy as np


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def get_utm_epsg(lon, lat):
    """
    根据给定的经纬度获取 UTM 投影的 EPSG 代码. 

    Parameters
    ----------
    lon : float
        经度. 
    lat : float
        纬度. 

    Returns
    -------
    epsg_code : int
        计算得到的 EPSG 代码, 用于标识 UTM 分区. 
    """
    zone = math.floor((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def gps_to_utm(pose, target_epsg=None):
    """
    将 GPS 坐标 (经度, 纬度, 海拔) 转换为 UTM 坐标, 并调整航向角. 

    Parameters
    ----------
    pose : list or tuple
        GPS 位姿列表 [lon, lat, alt, 0, 0, hea]. 
        - lon: 经度 (度)
        - lat: 纬度 (度)
        - alt: 海拔高度 (米)
        - hea: 航向角 (度, 正北为 0, 顺时针增加)
    target_epsg : int, optional
        目标 UTM 投影的 EPSG 代码. 如果未提供, 将根据经纬度自动计算. 

    Returns
    -------
    utm_pose : list
        转换后的 UTM 位姿 [x, y, z, roll, pitch, yaw]. 
        - x: 东向坐标 (Easting, 米)
        - y: 北向坐标 (Northing, 米)
        - z: 海拔高度 (米)
        - roll: 翻滚角 (设为 0)
        - pitch: 俯仰角 (设为 0)
        - yaw: 偏航角 (弧度, 数学坐标系定义: 正东为 0, 逆时针增加)
    """
    from pyproj import Transformer
    lon, lat, alt, _, _, hea = pose

    # 如果没有指定目标EPSG，则自动确定
    if target_epsg is None:
        target_epsg = get_utm_epsg(lon, lat)

    # 创建WGS84到UTM的转换器
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

    # 转换坐标
    x, y = transformer.transform(lon, lat)
    z = alt

    # 转换航向角：正北顺时针 → 正东逆时针
    # 航向角0°(北) → 90°, 90°(东) → 0°, 180°(南) → -90°
    yaw = math.radians(90 - hea)

    return [x, y, z, 0, 0, yaw]  # roll和pitch设为0


def gps_to_enu(ref_gps, target_gps):
    """
    将目标 GPS 坐标转换为相对于参考点的 ENU (East-North-Up) 局部坐标. 

    Parameters
    ----------
    ref_gps : list
        参考点的 GPS 位姿 [lon_ref, lat_ref, alt_ref, 0, 0, hea_ref]. 
    target_gps : list
        目标点的 GPS 位姿 [lon, lat, alt, 0, 0, hea]. 

    Returns
    -------
    enu_pose : list
        目标点在以参考点为原点的 ENU 坐标系下的位姿 [e, n, u, 0, 0, yaw]. 
        - e: 东向距离 (米)
        - n: 北向距离 (米)
        - u: 垂直高度差 (米)
        - yaw: 偏航角 (弧度, 数学坐标系定义)
    """
    from pyproj import Geod
    # 解析参考点和目标点GPS数据
    lon_ref, lat_ref, alt_ref, _, _, hea_ref = ref_gps
    lon, lat, alt, _, _, hea = target_gps

    # 创建大地测量对象
    geod = Geod(ellps='WGS84')

    # 计算相对方位角、距离和高差
    az, _, dist = geod.inv(lon_ref, lat_ref, lon, lat)
    du = alt - alt_ref

    # 将距离分解为东向和北向分量
    # 方位角az是从北向顺时针的角度（与航向角定义一致）
    azimuth_rad = math.radians(az)
    e = dist * math.sin(azimuth_rad)  # 东向分量
    n = dist * math.cos(azimuth_rad)  # 北向分量

    # 计算绝对航向角（在ENU坐标系中）
    # 航向角0°(北) → 90°(东), 90°(东) → 0°, 180°(南) → -90°(西)
    yaw = math.radians(90 - hea)

    return [e, n, du, 0, 0, yaw]


def pose_to_matrix(pose):
    """
    将位姿列表 [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵. 

    Parameters
    ----------
    pose : list
        位姿列表 [x, y, z, roll, pitch, yaw]. 
        其中 yaw 为弧度制. 

    Returns
    -------
    transformation_matrix : np.ndarray
        4x4 的齐次变换矩阵, 表示该位姿对应的刚体变换. 
    """
    x, y, z, roll, pitch, yaw = pose

    # 创建旋转矩阵 (仅考虑yaw，忽略roll和pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # 绕Z轴旋转 (yaw)
    rotation_matrix = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    # 创建4x4变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 3] = z

    return transformation_matrix


def gps_to_utm_transformation(x1_gps, x2_gps):
    """
    计算从 x1 坐标系到 x2 坐标系的变换矩阵 (基于 UTM 投影). 
    即: T_x2_x1, 使得 P_x2 = T_x2_x1 * P_x1. 

    Parameters
    ----------
    x1_gps : list
        x1 车辆的 GPS 位姿 [lon1, lat1, alt1, 0, 0, hea1]. 
    x2_gps : list
        x2 车辆的 GPS 位姿 [lon2, lat2, alt2, 0, 0, hea2]. 

    Returns
    -------
    transformation_matrix : np.ndarray
        从 x1 局部坐标系变换到 x2 局部坐标系的 4x4 变换矩阵. 
    """
    # 解析GPS数据
    lon1, lat1, alt1, _, _, hea1 = x1_gps
    lon2, lat2, alt2, _, _, hea2 = x2_gps

    # 确定参考UTM分区（以x1所在分区为参考）
    ref_epsg = get_utm_epsg(lon1, lat1)

    # 转换为UTM坐标系
    x1_pose = gps_to_utm(x1_gps, target_epsg=ref_epsg)
    x2_pose = gps_to_utm(x2_gps, target_epsg=ref_epsg)

    # 转换为变换矩阵
    x1_to_world = pose_to_matrix(x1_pose)
    x2_to_world = pose_to_matrix(x2_pose)

    # 计算世界坐标系到x2坐标系的逆变换
    world_to_x2 = np.linalg.inv(x2_to_world)

    # 计算从x1到x2的变换
    transformation_matrix = np.dot(world_to_x2, x1_to_world)

    return transformation_matrix


def gps_to_enu_transformation(x1_gps, x2_gps):
    """
    计算从 x1 坐标系到 x2 坐标系的变换矩阵 (基于 ENU 局部坐标系). 
    计算方法: 以 x1 为原点建立 ENU 坐标系, 计算 x2 在此坐标系下的位姿, 从而求出相对变换. 

    Parameters
    ----------
    x1_gps : list
        x1 车辆的 GPS 位姿 [lon1, lat1, alt1, 0, 0, hea1]. 
    x2_gps : list
        x2 车辆的 GPS 位姿 [lon2, lat2, alt2, 0, 0, hea2]. 

    Returns
    -------
    transformation_matrix : np.ndarray
        从 x1 局部坐标系变换到 x2 局部坐标系的 4x4 变换矩阵. 
    """
    # 以x1为参考点建立ENU坐标系
    # x1在ENU坐标系中的位姿是原点
    # 计算绝对航向角（在ENU坐标系中）
    # 航向角0°(北) → 90°(东), 90°(东) → 0°, 180°(南) → -90°(西)
    x1_pose = [0, 0, 0, 0, 0, math.radians(90 - x1_gps[5])]  # (e, n, u) = (0,0,0), yaw=90-hea1

    # 计算x2相对于x1的ENU位姿
    x2_pose = gps_to_enu(x1_gps, x2_gps)

    # 转换为变换矩阵
    x1_to_world = pose_to_matrix(x1_pose)
    x2_to_world = pose_to_matrix(x2_pose)

    # 计算世界坐标系(ENU)到x2坐标系的逆变换
    world_to_x2 = np.linalg.inv(x2_to_world)

    # 计算从x1到x2的变换
    transformation_matrix = np.dot(world_to_x2, x1_to_world)

    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()
    Args:
        pairwise_t_matrix: torch.tensor
            [B, L, L, 4, 4], B batchsize, L max_cav
        H: num.
            Feature map height
        W: num.
            Feature map width
        discrete_ratio * downsample_rate: num.
            One pixel on the feature map corresponds to the actual physical distance

    Returns:
        affine_matrix: torch.tensor
            [B, L, L, 2, 3]
    """

    affine_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    affine_matrix[...,0,1] = affine_matrix[...,0,1] * H / W
    affine_matrix[...,1,0] = affine_matrix[...,1,0] * W / H
    affine_matrix[...,0,2] = affine_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    affine_matrix[...,1,2] = affine_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    return affine_matrix
