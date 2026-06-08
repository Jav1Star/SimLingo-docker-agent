import numpy as np
import open3d as o3d

from typing import Tuple

from opencood.visualization.vis_utils import bbx2oabb, color_encoding

def init_vis() -> o3d.visualization.Visualizer:
    vis = o3d.visualization.Visualizer()
    
    vis.create_window(visible=False)
    vis_opt = vis.get_render_option()
    vis_opt.background_color = np.asarray([0, 0, 0])
    vis_opt.point_size = 3.0
    return vis


def get_grid_map(x_range: Tuple[float, float] = (-140.8, 140.8), y_range: Tuple[float, float] = (-38.4, 38.4), z: float = -3.0, grid_size: float = 6.4, color: list = [0.5, 0.5, 0.5]) -> object:
    """
    在指定的高度平面上创建一个网格地图的可视化对象.

    Parameters
    -----------
    x_range : tuple
        X 轴的范围 (x_min, x_max).
    y_range : tuple
        Y 轴的范围 (y_min, y_max).
    z : float
        网格所在的 Z 轴高度.
    grid_size : float
        网格的单元格大小.
    color : list
        网格线的颜色 [r, g, b].

    Returns
    --------
    line_set : open3d.geometry.LineSet
        用于可视化的 Open3D 线集合对象.
    """
    lines = []
    points = []

    x_min, x_max = x_range
    y_min, y_max = y_range

    # 垂直线
    x_coords = np.arange(x_min, x_max + grid_size, grid_size)
    for x in x_coords:
        p1 = [x, y_min, z]
        p2 = [x, y_max, z]
        points.extend([p1, p2])
        lines.append([len(points) - 2, len(points) - 1])

    # 水平线
    y_coords = np.arange(y_min, y_max + grid_size, grid_size)
    for y in y_coords:
        p1 = [x_min, y, z]
        p2 = [x_max, y, z]
        points.extend([p1, p2])
        lines.append([len(points) - 2, len(points) - 1])

    colors = [color for _ in range(len(lines))]

    import open3d as o3d

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

grid_map = get_grid_map()
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
left_hand_coordinate = True
vis_zoom = 0.3

def render_vis(vis, processed_pcd, ego_pred_box=None, fused_pred_box=None, others_oabbs=None, projected_others_pcds=None):
        processed_pcd = processed_pcd.copy()
        ego_pred_box = ego_pred_box.copy() if ego_pred_box is not None else None
        fused_pred_box = fused_pred_box.copy() if fused_pred_box is not None else None

        if vis is None:
            vis = init_vis()

        vis.clear_geometries()

        vis.add_geometry(axis)
        vis.add_geometry(grid_map)

        pcd = o3d.geometry.PointCloud()

        if left_hand_coordinate:
            processed_pcd[:, 1:2] = -processed_pcd[:, 1:2]
            if projected_others_pcds is not None:
                for projected_pcd in projected_others_pcds:
                    projected_pcd[:, 1:2] = -projected_pcd[:, 1:2]

        pcd.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])

        origin_lidar_intcolor = color_encoding(processed_pcd[:, 2], mode='constant')
        pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

        vis.add_geometry(pcd)

        if projected_others_pcds is not None:
            for projected_pcd in projected_others_pcds:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(projected_pcd[:, :3])
                blue_color = np.array([0, 0, 1.0])  # RGB值范围[0,1]
                pcd.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(pcd.points), 1)))  # 复制颜色到所有点

                vis.add_geometry(pcd)

        if ego_pred_box is not None and ego_pred_box.size > 0:
            # print(f'ego_pred_box.size = {ego_pred_box.shape[0]}')
            oabbs_pred = bbx2oabb(ego_pred_box, color=(1, 0, 0), left_hand_coordinate=left_hand_coordinate)
            for oabb in oabbs_pred:
                # pass
                vis.add_geometry(oabb)

        if fused_pred_box is not None and fused_pred_box.size > 0:
            # print(f'fused_pred_box.size = {fused_pred_box.shape[0]}')
            if ego_pred_box is not None and ego_pred_box.size > 0:
                more_pred_box_count = fused_pred_box.shape[0] - ego_pred_box.shape[0]
                if more_pred_box_count > 0:
                    print(f'fusion method get {more_pred_box_count} more pred box')
            oabbs_pred = bbx2oabb(fused_pred_box, color=(0, 1, 0), left_hand_coordinate=left_hand_coordinate)
            for oabb in oabbs_pred:
                vis.add_geometry(oabb)

        # oabbs_gt = get_oabbs_gt(self.shared_info, color=(0, 0, 1))
        # for oabb in oabbs_gt:
        #     self.vis.add_geometry(oabb)

        if others_oabbs is not None:
            for oabb in others_oabbs:
                vis.add_geometry(oabb)

        view_control = vis.get_view_control()

        # cam = view_control.convert_to_pinhole_camera_parameters()
        #
        # cam.extrinsic = np.array([[1, 0, 0, 0],  # 调整相机位置
        #                           [0, -1, 0, 0],
        #                           [0, 0, -1, 100],  # Z值增大=拉远相机
        #                           [0, 0, 0, 1]])
        # extrinsic = np.array(cam.extrinsic)
        # extrinsic[2, 3] = 10
        # cam.extrinsic = extrinsic
        # view_control.convert_from_pinhole_camera_parameters(cam)
        # view_control.set_lookat([0, 0, 0])  # 焦点在原点
        # self.vis.reset_view_point(True)

        view_control.set_lookat([0, 0, 0])  # 焦点在原点
        view_control.set_up([0, 1, 0])  # Y轴向上
        view_control.set_front([0, 0, 1])  # 摄像头朝向Z轴负方向 (从100看0)
        view_control.set_zoom(vis_zoom)  # 调整缩放因子，让整个点云在视野中

        # 渲染场景
        vis.poll_events()
        vis.update_renderer()

        pcd_img = np.asarray(vis.capture_screen_float_buffer(do_render=True))

        vis.destroy_window()

        return pcd_img
