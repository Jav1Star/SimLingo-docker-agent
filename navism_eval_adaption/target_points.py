from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from shapely.geometry import Point

from navsim.common.dataclasses import Scene
from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction


class TargetPointBuilder:
    def __init__(self, target_distance: float = 7.5, search_depth: int = 30, map_radius: float = 50.0):
        self.target_distance = target_distance
        self.search_depth = search_depth
        self.map_radius = map_radius
        self.vehicle_parameters = get_pacifica_parameters()

    def build(self, scene: Scene) -> List[np.ndarray]:
        current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        current_ego_state = ego_status_to_ego_state(
            current_frame.ego_status,
            self.vehicle_parameters,
            TimePoint(int(current_frame.timestamp)),
        )

        route_lane_dict = self._load_route_lane_dict(
            scene=scene,
            current_ego_state=current_ego_state,
        )
        centerline = self._build_centerline(
            scene=scene,
            current_ego_state=current_ego_state,
            route_lane_state=route_lane_dict,
        )
        centerline_progress = np.asarray(
            [centerline.project(Point(state.x, state.y)) for state in centerline.discrete_path],
            dtype=np.float64,
        )
        ego_pose = frame.ego_status.ego_pose
        ego_progress = float(centerline.project(Point(ego_pose[0], ego_pose[1])))
        first_idx = int(np.searchsorted(centerline_progress, ego_progress + self.target_distance, side="left"))
        first_idx = min(first_idx, len(centerline.discrete_path) - 1)
        second_idx = min(first_idx + 1, len(centerline.discrete_path) - 1)

            points_global = np.array(
                [
                    [centerline.discrete_path[first_idx].x, centerline.discrete_path[first_idx].y],
                    [centerline.discrete_path[second_idx].x, centerline.discrete_path[second_idx].y],
                ],
                dtype=np.float32,
            )
            target_points.append(self._to_local_xy(points_global, ego_pose))

        return target_points

    def _load_route_lane_dict(self, scene: Scene, frame, current_ego_state):
        route_roadblock_ids = list(dict.fromkeys(frame.roadblock_ids))
        route_roadblock_dict = {}
        for roadblock_id in route_roadblock_ids:
            block = scene.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            block = block or scene.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if block is None:
                continue
            route_roadblock_dict[block.id] = block
        if not route_roadblock_dict:
            return None

        corrected_ids = route_roadblock_correction(current_ego_state.rear_axle, scene.map_api, route_roadblock_dict)
        if not corrected_ids:
            corrected_ids = list(route_roadblock_dict.keys())

        route_lane_dict = {}
        route_roadblocks = []
        for roadblock_id in corrected_ids:
            block = scene.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            block = block or scene.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if block is None:
                continue
            route_roadblocks.append(block)
            for lane in block.interior_edges:
                route_lane_dict[lane.id] = lane
        if not route_roadblocks or not route_lane_dict:
            return None
        return route_roadblocks, route_lane_dict

    def _build_centerline(self, scene: Scene, current_ego_state, route_lane_state):
        route_roadblocks, route_lane_dict = route_lane_state
        drivable_area_map = PDMDrivableMap.from_simulation(
            map_api=scene.map_api,
            ego_state=current_ego_state,
            map_radius=self.map_radius,
        )
        starting_lane = self._get_starting_lane(
            ego_state=current_ego_state,
            route_lane_dict=route_lane_dict,
            drivable_area_map=drivable_area_map,
        )
        if starting_lane is None:
            return None

        roadblock_ids = [roadblock.id for roadblock in route_roadblocks]
        start_matches = np.flatnonzero(np.array(roadblock_ids) == starting_lane.get_roadblock_id())
        start_idx = int(start_matches[0]) if len(start_matches) else 0
        roadblock_window = route_roadblocks[start_idx : start_idx + self.search_depth]
        if not roadblock_window:
            roadblock_window = route_roadblocks[start_idx:]
        if not roadblock_window:
            return None

        route_plan, _ = Dijkstra(starting_lane, list(route_lane_dict.keys())).search(roadblock_window[-1])
        if not route_plan:
            route_plan = [starting_lane]

        centerline_discrete_path = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)
        return PDMPath(centerline_discrete_path) if centerline_discrete_path else None

    def _get_starting_lane(self, ego_state, route_lane_dict, drivable_area_map) -> LaneGraphEdgeMapObject:
        ego_position = ego_state.rear_axle.array
        ego_heading = ego_state.rear_axle.heading
        ego_point = Point(*ego_position)

        intersecting_lanes = drivable_area_map.intersects(ego_point)
        candidates = []
        heading_errors = []
        for lane_id in intersecting_lanes:
            if lane_id not in route_lane_dict:
                continue
            lane = route_lane_dict[lane_id]
            lane_states = np.array([state.array for state in lane.baseline_path.discrete_path], dtype=np.float64)
            lane_distances = np.linalg.norm(ego_position[None, :] - lane_states, axis=-1)
            heading_error = lane.baseline_path.discrete_path[int(np.argmin(lane_distances))].heading - ego_heading
            candidates.append(lane)
            heading_errors.append(abs(normalize_angle(heading_error)))
        if candidates:
            return candidates[int(np.argmin(np.asarray(heading_errors)))]

        starting_lane = None
        closest_distance = np.inf
        for lane in route_lane_dict.values():
            if lane.contains_point(ego_state.center):
                return lane
            distance = lane.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_lane = lane
                closest_distance = distance
        return starting_lane

    def _to_local_xy(self, points_global: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        dx = points_global[:, 0] - ego_pose[0]
        dy = points_global[:, 1] - ego_pose[1]
        cos_yaw = np.cos(ego_pose[2])
        sin_yaw = np.sin(ego_pose[2])
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return np.stack([local_x, local_y], axis=-1).astype(np.float32)

    def _default_target_points(self) -> np.ndarray:
        return np.asarray(
            [
                [self.target_distance, 0.0],
                [self.target_distance + 1.0, 0.0],
            ],
            dtype=np.float32,
        )
