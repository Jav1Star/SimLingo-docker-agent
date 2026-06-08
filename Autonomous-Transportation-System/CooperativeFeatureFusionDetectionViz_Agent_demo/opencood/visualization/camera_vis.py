import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import copy
import os
from opencood.utils.camera_utils import draw_coord_3d_to_2d


def visualize_camera(infer_result, camera_data, camera_params, save_path=None,
                     camera_idx=0, show_labels=True, show_scores=True):
    """
    在单个相机图像上可视化检测结果和GT框

    Parameters
    ----------
    infer_result : dict
        推理结果字典，包含:
        - pred_box: np.ndarray, (N, 8, 3) 预测框
        - fused_pred_box: np.ndarray, (M, 8, 3) 融合后检测框
        - score_tensor: torch.Tensor, (N,) 预测分数 (可选)
        - uncertainty_tensor: torch.Tensor, (N, ?) 不确定性 (可选)

    camera_data : PIL.Image or np.ndarray
        相机图像数据

    camera_params : dict
        相机参数字典，包含:
        - camera_to_lidar: np.ndarray, (4, 4) 相机到LiDAR的变换矩阵
        - camera_intrinsic: np.ndarray, (3, 3) 相机内参矩阵
        - post_rot: np.ndarray, (3, 3) 数据增强旋转矩阵 (可选)
        - post_tran: np.ndarray, (3,) 数据增强平移向量 (可选)

    save_path : str
        保存路径

    camera_idx : int
        相机索引

    show_labels : bool
        是否显示标签

    show_scores : bool
        是否显示分数

    Returns
    -------
    vis_image : PIL.Image
        可视化后的图像
    """

    # 转换图像格式
    if isinstance(camera_data, np.ndarray):
        if camera_data.dtype == np.uint8:
            image = Image.fromarray(camera_data)
        else:
            image = Image.fromarray((camera_data * 255).astype(np.uint8))
    else:
        image = camera_data.copy()

    # 获取图像尺寸
    image_width, image_height = image.size

    # 获取相机参数
    camera_to_lidar = camera_params['camera_to_lidar']
    camera_intrinsic = camera_params['camera_intrinsic']
    post_rot = camera_params.get('post_rot', None)
    post_tran = camera_params.get('post_tran', None)

    # 确保内参矩阵是3x3
    if camera_intrinsic.shape == (4, 4):
        camera_intrinsic = camera_intrinsic[:3, :3]

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # 处理预测框
    pred_box_np = infer_result.get("pred_box", None)
    if pred_box_np is not None and pred_box_np.shape[0] > 0:

        # 获取分数
        score = infer_result.get("score_tensor", None)
        score_np = None
        if score is not None:
            score_np = score.cpu().numpy()

        # 获取不确定性
        uncertainty = infer_result.get("uncertainty_tensor", None)
        uncertainty_np = None
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)

        # 投影3D框到2D，考虑数据增强
        pred_box2d, pred_mask, _ = draw_coord_3d_to_2d(
            pred_box_np, camera_intrinsic, camera_to_lidar,
            image_height, image_width,
            post_rot=post_rot, post_tran=post_tran
        )

        # 绘制预测框
        for i, (box2d, is_valid) in enumerate(zip(pred_box2d, pred_mask)):
            if not is_valid:
                continue

            # 计算2D边界框
            u_coords = box2d[:, 0]
            v_coords = box2d[:, 1]
            u_min, u_max = u_coords.min(), u_coords.max()
            v_min, v_max = v_coords.min(), v_coords.max()

            # 绘制边界框
            draw.rectangle([u_min, v_min, u_max, v_max],
                           outline='red', width=2)

            # 绘制标签
            if show_labels:
                label_text = f"Pred"
                if show_scores and score_np is not None:
                    label_text += f": {score_np[i]:.3f}"
                if uncertainty_np is not None:
                    if uncertainty_np.shape[1] >= 2:
                        label_text += f"\nU: {uncertainty_np[i, 0]:.3f}"

                # 绘制文本背景
                if font:
                    bbox = draw.textbbox((u_min, v_min - 25), label_text, font=font)
                    draw.rectangle(bbox, fill='red')
                    draw.text((u_min, v_min - 25), label_text, fill='white', font=font)
                else:
                    draw.text((u_min, v_min - 25), label_text, fill='red')

    # 处理融合后框
    fused_pred_box = infer_result.get("fused_pred_box", None)
    if fused_pred_box is not None and fused_pred_box.shape[0] > 0:
        # 投影3D框到2D，考虑数据增强
        gt_box2d, gt_mask, _ = draw_coord_3d_to_2d(
            fused_pred_box, camera_intrinsic, camera_to_lidar,
            image_height, image_width,
            post_rot=post_rot, post_tran=post_tran
        )

        # 绘制融合后框
        for i, (box2d, is_valid) in enumerate(zip(gt_box2d, gt_mask)):
            if not is_valid:
                continue

            # 计算2D边界框
            u_coords = box2d[:, 0]
            v_coords = box2d[:, 1]
            u_min, u_max = u_coords.min(), u_coords.max()
            v_min, v_max = v_coords.min(), v_coords.max()

            # 绘制边界框
            draw.rectangle([u_min, v_min, u_max, v_max],
                           outline='green', width=2)

            # 绘制标签
            if show_labels:
                label_text = "fused"

                # 绘制文本背景
                if font:
                    bbox = draw.textbbox((u_min, v_min - 25), label_text, font=font)
                    draw.rectangle(bbox, fill='green')
                    draw.text((u_min, v_min - 25), label_text, fill='white', font=font)
                else:
                    draw.text((u_min, v_min - 25), label_text, fill='green')

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    return image


def visualize_multiple_cameras(infer_result, batch_data, save_dir, frame_idx,
                               show_labels=True, show_scores=True):
    """
    可视化多个相机图像

    Parameters
    ----------
    infer_result : dict
        推理结果字典

    batch_data : dict
        批次数据，包含相机数据和参数

    save_dir : str
        保存目录

    frame_idx : int
        帧索引

    show_labels : bool
        是否显示标签

    show_scores : bool
        是否显示分数
    """

    if 'ego' not in batch_data or 'camera_data' not in batch_data['ego']:
        print("Warning: No camera data found in batch_data")
        return

    camera_data_list = batch_data['ego']['camera_data']
    params = batch_data['ego'].get('params', {})

    # 获取相机参数
    camera_to_lidar_list = params.get('camera_to_lidar', [])
    camera_intrinsic_list = params.get('camera_intrinsic', [])

    # 确保参数列表长度匹配
    num_cameras = len(camera_data_list)
    if len(camera_to_lidar_list) < num_cameras:
        camera_to_lidar_list.extend([np.eye(4)] * (num_cameras - len(camera_to_lidar_list)))
    if len(camera_intrinsic_list) < num_cameras:
        default_intrinsic = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]])
        camera_intrinsic_list.extend([default_intrinsic] * (num_cameras - len(camera_intrinsic_list)))

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 可视化每个相机
    for i, camera_data in enumerate(camera_data_list):
        camera_params = {
            'camera_to_lidar': camera_to_lidar_list[i],
            'camera_intrinsic': camera_intrinsic_list[i]
        }

        save_path = os.path.join(save_dir, f'camera_{i:02d}_frame_{frame_idx:05d}.png')

        try:
            visualize_camera(
                infer_result=infer_result,
                camera_data=camera_data,
                camera_params=camera_params,
                save_path=save_path,
                camera_idx=i,
                show_labels=show_labels,
                show_scores=show_scores
            )
        except Exception as e:
            print(f"Error visualizing camera {i}: {e}")


def create_camera_visualization_summary(infer_result, batch_data, save_dir, frame_idx):
    """
    创建包含所有相机的总结可视化图

    Parameters
    ----------
    infer_result : dict
        推理结果字典

    batch_data : dict
        批次数据

    save_dir : str
        保存目录

    frame_idx : int
        帧索引
    """

    if 'ego' not in batch_data or 'camera_data' not in batch_data['ego']:
        print("Warning: No camera data found in batch_data")
        return

    camera_data_list = batch_data['ego']['camera_data']
    params = batch_data['ego'].get('params', {})

    # 获取相机参数
    camera_to_lidar_list = params.get('camera_to_lidar', [])
    camera_intrinsic_list = params.get('camera_intrinsic', [])

    # 确保参数列表长度匹配
    num_cameras = len(camera_data_list)
    if len(camera_to_lidar_list) < num_cameras:
        camera_to_lidar_list.extend([np.eye(4)] * (num_cameras - len(camera_to_lidar_list)))
    if len(camera_intrinsic_list) < num_cameras:
        default_intrinsic = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]])
        camera_intrinsic_list.extend([default_intrinsic] * (num_cameras - len(camera_intrinsic_list)))

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 计算子图布局
    if num_cameras <= 2:
        rows, cols = 1, num_cameras
    elif num_cameras <= 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(np.sqrt(num_cameras)))
        cols = int(np.ceil(num_cameras / rows))

    # 创建总结图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    if num_cameras == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # 可视化每个相机
    for i, camera_data in enumerate(camera_data_list):
        if i >= len(axes):
            break

        camera_params = {
            'camera_to_lidar': camera_to_lidar_list[i],
            'camera_intrinsic': camera_intrinsic_list[i]
        }

        try:
            # 获取可视化图像
            vis_image = visualize_camera(
                infer_result=infer_result,
                camera_data=camera_data,
                camera_params=camera_params,
                save_path=None,  # 不保存单独图像
                camera_idx=i,
                show_labels=True,
                show_scores=True
            )

            # 显示在子图中
            axes[i].imshow(vis_image)
            axes[i].set_title(f'Camera {i}')
            axes[i].axis('off')

        except Exception as e:
            print(f"Error visualizing camera {i}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {e}',
                         transform=axes[i].transAxes, ha='center', va='center')
            axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(num_cameras, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    summary_path = os.path.join(save_dir, f'camera_summary_frame_{frame_idx:05d}.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_camera_params(params_dict, camera_idx=0):
    """
    从参数字典中提取相机参数

    Parameters
    ----------
    params_dict : dict
        参数字典

    camera_idx : int
        相机索引

    Returns
    -------
    camera_params : dict
        相机参数字典
    """

    camera_to_lidar_list = params_dict.get('camera_to_lidar', [])
    camera_intrinsic_list = params_dict.get('camera_intrinsic', [])

    # 获取指定相机的参数
    if camera_idx < len(camera_to_lidar_list):
        camera_to_lidar = camera_to_lidar_list[camera_idx]
    else:
        camera_to_lidar = np.eye(4)

    if camera_idx < len(camera_intrinsic_list):
        camera_intrinsic = camera_intrinsic_list[camera_idx]
    else:
        camera_intrinsic = np.array([[500, 0, 400], [0, 500, 300], [0, 0, 1]])

    return {
        'camera_to_lidar': camera_to_lidar,
        'camera_intrinsic': camera_intrinsic
    }


def visualize(infer_result, camera_data, camera_params, save_path,
              camera_idx=0, show_labels=True, show_scores=True):
    """
    主要的可视化函数，与simple_vis.visualize保持一致的接口

    Parameters
    ----------
    infer_result : dict
        推理结果字典

    camera_data : PIL.Image or np.ndarray
        相机图像数据

    camera_params : dict
        相机参数字典

    save_path : str
        保存路径

    camera_idx : int
        相机索引

    show_labels : bool
        是否显示标签

    show_scores : bool
        是否显示分数
    """

    return visualize_camera(
        infer_result=infer_result,
        camera_data=camera_data,
        camera_params=camera_params,
        save_path=save_path,
        camera_idx=camera_idx,
        show_labels=show_labels,
        show_scores=show_scores
    )