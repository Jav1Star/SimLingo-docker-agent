import numpy as np
from typing import Optional

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, NAVSIM_INTERVAL_LENGTH, Scene, SensorConfig

from .bootstrap import SmartAssignerRuntime
from .input_builder import DrivingInputBuilder
from .target_points import TargetPointBuilder
from .trajectory_adapter import predictions_to_trajectory


class NavsimSmartAssignerAgent(AbstractAgent):
    requires_scene = True

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1),
        use_route_centerline: bool = True,
        max_speed_mps: float = 6.0,
        max_accel_mps2: float = 2.0,
        max_decel_mps2: float = 4.0,
        process_only_new_history_frames: bool = True,
    ):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_route_centerline = use_route_centerline
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_decel_mps2 = max_decel_mps2
        self.process_only_new_history_frames = process_only_new_history_frames

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self.runtime = SmartAssignerRuntime(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self.target_point_builder = TargetPointBuilder()
        self.input_builder = DrivingInputBuilder(self.runtime)
        self._active_log_name = None
        self._active_log_time_origin_s: Optional[float] = None
        self._last_processed_timestamp_s: Optional[float] = None

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[0, 1, 2, 3],
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene):
        log_name = scene.scene_metadata.log_name
        route_key = log_name

        if self._active_log_name != log_name:
            self.runtime.model.budget_assigner.reset()
            self._active_log_name = log_name
            self._active_log_time_origin_s = None
            self._last_processed_timestamp_s = None

        target_points_by_frame = self.target_point_builder.build(scene)
        reference_route = (
            self.target_point_builder.build_route_polyline(scene)
            if self.use_route_centerline
            else None
        )

        pred_speed_wps = None
        pred_route = None
        for step_idx in range(scene.scene_metadata.num_history_frames):
            history_frame = scene.frames[step_idx]
            timestamp_s = self._relative_frame_timestamp_s(
                history_frame,
                fallback_s=step_idx * NAVSIM_INTERVAL_LENGTH,
            )
            if (
                self.process_only_new_history_frames
                and self._last_processed_timestamp_s is not None
                and timestamp_s <= self._last_processed_timestamp_s + 1e-4
            ):
                continue
            model_input = self.input_builder.build(
                agent_input=agent_input,
                target_points_by_frame=target_points_by_frame,
                step_idx=step_idx,
                global_ego_pose=history_frame.ego_status.ego_pose,
                timestamp_s=timestamp_s,
            )
            pred_speed_wps, pred_route, _ = self.runtime.model(
                model_input,
                budget=None,
                route_keys=[route_key],
            )
            self._last_processed_timestamp_s = timestamp_s

        if pred_speed_wps is None or pred_route is None:
            final_idx = scene.scene_metadata.num_history_frames - 1
            history_frame = scene.frames[final_idx]
            self.runtime.model.budget_assigner.reset(route_keys=[route_key])
            timestamp_s = self._relative_frame_timestamp_s(
                history_frame,
                fallback_s=final_idx * NAVSIM_INTERVAL_LENGTH,
            )
            model_input = self.input_builder.build(
                agent_input=agent_input,
                target_points_by_frame=target_points_by_frame,
                step_idx=final_idx,
                global_ego_pose=history_frame.ego_status.ego_pose,
                timestamp_s=timestamp_s,
            )
            pred_speed_wps, pred_route, _ = self.runtime.model(
                model_input,
                budget=None,
                route_keys=[route_key],
            )
            self._last_processed_timestamp_s = timestamp_s

        return predictions_to_trajectory(
            speed_wps=pred_speed_wps[0],
            route=pred_route[0],
            speed_dt=float(self.runtime.cfg.model.smart_assigner.t_lap),
            current_speed_mps=float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity)),
            reference_route=reference_route,
            max_speed_mps=self.max_speed_mps,
            max_accel_mps2=self.max_accel_mps2,
            max_decel_mps2=self.max_decel_mps2,
        )

    def _relative_frame_timestamp_s(self, frame, fallback_s: float) -> float:
        raw_timestamp = getattr(frame, "timestamp", None)
        if raw_timestamp is None:
            timestamp_s = float(fallback_s)
        else:
            timestamp_s = float(raw_timestamp) * 1e-6

        if self._active_log_time_origin_s is None:
            self._active_log_time_origin_s = timestamp_s
        return float(timestamp_s - self._active_log_time_origin_s)
