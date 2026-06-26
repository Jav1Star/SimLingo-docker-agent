import os
import numpy as np
import yaml
import pickle
import base64

from PIL import Image
from typing import Dict, Any, Tuple

from .yaml_utils import load_yaml
from .json_utils import load_json
from .transformation_utils import x1_to_x2
from .logger_utils import get_logger
from .numpy_utils import encode_array_to_dict


logger = get_logger(__name__)


def get_ext_int(params: dict, camera_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从参数字典中获取指定相机的外参 (相机到激光雷达) 和内参矩阵. 

    Parameters
    ----------
    params : dict
        包含相机参数和激光雷达位姿的字典 (通常来自 yaml 文件). 
    camera_id : int
        相机的 ID (0-3). 

    Returns
    -------
    camera_to_lidar : np.ndarray
        相机坐标系到激光雷达坐标系的变换矩阵 (4x4). 
    camera_intrinsic : np.ndarray
        相机内参矩阵 (3x3 或 3x4). 
    """
    camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
        np.float32)
    camera_to_lidar = x1_to_x2(
        camera_coords, params["lidar_pose"]
    ).astype(np.float32)  # T_LiDAR_camera
    camera_to_lidar = camera_to_lidar @ np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32)  # UE4 coord to opencv coord
    camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
        np.float32
    )
    return camera_to_lidar, camera_intrinsic

def fetch_info_from_pcd_dataset(file_path: str, load_camera: bool = False) -> Dict[str, Any]:
    """
    从点云数据集文件中读取完整信息, 包括位姿、点云、速度、时间戳以及可选的相机数据. 

    Parameters
    ----------
    file_path : str
        数据文件路径. 
    load_camera : bool, optional
        是否加载对应的相机图像及内外参数据. 默认为 False. 

    Returns
    -------
    info_dict : dict
        包含以下键值的字典:
        - 'lidar_pose': np.ndarray, 激光雷达位姿. 
        - 'ts_lidar_pose': int, 位姿时间戳. 
        - 'pcd': np.ndarray, 点云数据. 
        - 'ts_pcd': int, 点云时间戳. 
        - 'speed': np.ndarray, 自车速度. 
        - 'ts_speed': int, 速度时间戳. 
        - 'camera_infos': list (可选), 包含相机数据的字典列表. 每个元素包含:
            - 'camera_to_lidar': np.ndarray, 相机到激光雷达的变换矩阵. 
            - 'camera_intrinsic': np.ndarray, 相机内参矩阵. 
            - 'img': np.ndarray, 图像数据. 
    """
    file_name, file_type = file_path.split('.')

    if os.name == 'nt':
        ts = file_name.split('\\')[-1]
    else:
        ts = file_name.split('/')[-1]
    ts = int(ts)

    lidar_pose = np.array([])
    speed = np.array([])
    pcd = np.array([])
    camera_infos = []

    if file_type == 'pcd' or file_type == 'yaml':
        import open3d as o3d

        path = file_path.split('.')[0]
        try:
            yaml_load = load_yaml(path + '.yaml')
            lidar_pose = np.asarray(yaml_load['lidar_pose'])
            speed = np.asarray(yaml_load['ego_speed'])

            if load_camera:
                camera_file_name = f'{path}_camera'

                for idx in range(4):
                    camera_to_lidar, camera_intrinsic = get_ext_int(yaml_load, idx)
                    img = np.array(Image.open(f"{camera_file_name}{idx}.png"))
                    camera_data = {
                        'camera_to_lidar': camera_to_lidar,
                        'camera_intrinsic': camera_intrinsic,
                        'img': img
                    }
                    camera_infos.append(camera_data)
        except Exception as e:
            logger.error(f"Error in fetch_info_from_pcd_dataset: Failed to load YAML file for {file_path}: {e}")
            return {"status": "error", "message": "Failed to fetch perception info due to internal error."}

        pcd_load = o3d.io.read_point_cloud(path + '.pcd')

        # 将Open3D的点云对象转换为NumPy数组
        xyz = np.asarray(pcd_load.points).astype(np.float32)
        intensity = np.expand_dims(np.asarray(pcd_load.colors)[:, 0], -1).astype(np.float32)
        pcd = np.hstack((xyz, intensity))

    elif file_type == 'json':
        json_load = load_json(file_path)

        lidar_pose = np.array(json_load['lidar_pose']) if 'lidar_pose' in json_load else None
        pcd = np.array(json_load['pcd']) if 'pcd' in json_load else None
        if isinstance(pcd, np.ndarray):
            pcd[:, 3] = pcd[:, 3] / 255.0
    elif file_type == 'txt':
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        data_dict = pickle.loads(binary_data)

        lidar_pose = np.array(data_dict['lidar_pose']) if 'lidar_pose' in data_dict else None
        pcd = np.array(data_dict['pcd']) if 'pcd' in data_dict else None
        if isinstance(pcd, np.ndarray):
            pcd[:, 3] = pcd[:, 3] / 255.0

    perception_info = {
        "status": "success",
        "lidar_pose": lidar_pose.tolist() if isinstance(lidar_pose, np.ndarray) else None,
        "ts_lidar_pose": ts,
        "pcd": encode_array_to_dict(pcd) if isinstance(pcd, np.ndarray) else {"data": None, "shape": None, "dtype": None},
        "ts_pcd": ts,
        "speed": speed.tolist() if isinstance(speed, np.ndarray) else None,
        "ts_speed": ts,
    }

    if load_camera:
        perception_info["camera_infos"] = camera_infos

    return perception_info
