import numpy as np
import torch
from typing import Optional
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory


def _cumulative_arc_length(points_xy: np.ndarray) -> np.ndarray:
    deltas = np.diff(points_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)], axis=0)


def _decode_speed_knots(speed_xy: np.ndarray, speed_dt: float, current_speed_mps: float) -> np.ndarray:
    current_speed = np.float32(max(0.0, float(current_speed_mps)))
    speed_xy = np.asarray(speed_xy, dtype=np.float32)
    if speed_xy.size == 0 or speed_dt <= 0.0:
        return np.array([current_speed], dtype=np.float32)

    speed_polyline = np.concatenate([np.zeros((1, 2), dtype=np.float32), speed_xy], axis=0)
    if len(speed_polyline) < 2:
        return np.array([current_speed], dtype=np.float32)

    segment_speed = np.linalg.norm(speed_polyline[1:] - speed_polyline[:-1], axis=-1) / np.float32(speed_dt)
    return np.concatenate([np.array([current_speed], dtype=np.float32), segment_speed.astype(np.float32)], axis=0)


def _extend_speed_profile(
    v_knots: np.ndarray,
    src_dt: float = 0.2,
    dst_dt: float = 0.1,
    horizon: float = 4.0,
    max_speed_mps: Optional[float] = None,
    max_accel_mps2: Optional[float] = None,
    max_decel_mps2: Optional[float] = None,
) -> np.ndarray:
    v_knots = np.asarray(v_knots, dtype=np.float32)
    if dst_dt <= 0.0 or horizon <= 0.0:
        return np.zeros((0,), dtype=np.float32)

    num_steps = int(round(horizon / dst_dt))
    if num_steps <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(v_knots) == 0:
        return np.zeros((num_steps,), dtype=np.float32)
    if len(v_knots) == 1:
        v_dst = np.full((num_steps,), v_knots[0], dtype=np.float32)
        v_dst = _limit_speed_profile(
            v_dst,
            dst_dt=dst_dt,
            max_speed_mps=max_speed_mps,
            max_accel_mps2=max_accel_mps2,
            max_decel_mps2=max_decel_mps2,
        )
        return np.cumsum(v_dst * np.float32(dst_dt), dtype=np.float32)

    t_src = np.arange(len(v_knots), dtype=np.float32) * np.float32(src_dt)
    t_dst = np.arange(dst_dt, horizon + 1e-6, dst_dt, dtype=np.float32)

    v_dst = np.interp(np.minimum(t_dst, t_src[-1]), t_src, v_knots).astype(np.float32)

    future_mask = t_dst > t_src[-1]
    if np.any(future_mask):
        v0 = float(v_knots[-1])
        a0 = float((v_knots[-1] - v_knots[-2]) / src_dt) if len(v_knots) >= 2 and src_dt > 0.0 else 0.0
        ramp = 0.6
        tau = t_dst[future_mask] - t_src[-1]
        tau1 = np.minimum(tau, ramp)
        v_dst[future_mask] = v0 + a0 * tau1 - 0.5 * a0 * (tau1**2) / ramp

        cruise_mask = t_dst > t_src[-1] + ramp
        if np.any(cruise_mask):
            v_cruise = max(0.0, v0 + 0.5 * a0 * ramp)
            v_dst[cruise_mask] = v_cruise

    v_dst = np.maximum(v_dst, 0.0).astype(np.float32)
    v_dst = _limit_speed_profile(
        v_dst,
        dst_dt=dst_dt,
        max_speed_mps=max_speed_mps,
        max_accel_mps2=max_accel_mps2,
        max_decel_mps2=max_decel_mps2,
    )
    return np.cumsum(v_dst * np.float32(dst_dt), dtype=np.float32)


def _limit_speed_profile(
    speeds: np.ndarray,
    dst_dt: float,
    max_speed_mps: Optional[float],
    max_accel_mps2: Optional[float],
    max_decel_mps2: Optional[float],
) -> np.ndarray:
    speeds = np.asarray(speeds, dtype=np.float32).copy()
    if speeds.size == 0:
        return speeds
    if max_speed_mps is not None:
        speeds = np.minimum(speeds, np.float32(max(0.0, float(max_speed_mps))))
    if dst_dt <= 0.0:
        return np.maximum(speeds, 0.0).astype(np.float32)

    accel_limit = None if max_accel_mps2 is None else max(0.0, float(max_accel_mps2)) * float(dst_dt)
    decel_limit = None if max_decel_mps2 is None else max(0.0, float(max_decel_mps2)) * float(dst_dt)
    for idx in range(1, len(speeds)):
        lo = 0.0 if decel_limit is None else max(0.0, float(speeds[idx - 1]) - decel_limit)
        hi = float("inf") if accel_limit is None else float(speeds[idx - 1]) + accel_limit
        speeds[idx] = np.float32(np.clip(float(speeds[idx]), lo, hi))
    return np.maximum(speeds, 0.0).astype(np.float32)


def _as_numpy_xy(points) -> Optional[np.ndarray]:
    if points is None:
        return None
    if isinstance(points, torch.Tensor):
        points_xy = points.detach().float().cpu().numpy()
    else:
        points_xy = np.asarray(points)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.ndim != 2 or points_xy.shape[-1] != 2 or len(points_xy) == 0:
        return None
    if not np.isfinite(points_xy).all():
        return None
    return points_xy


def _interpolate_polyline(points_xy: np.ndarray, arc_length: np.ndarray, query_arc: np.ndarray) -> np.ndarray:
    if len(points_xy) == 1:
        return np.repeat(points_xy, len(query_arc), axis=0)

    query_arc = np.asarray(query_arc, dtype=np.float32)
    clipped_arc = np.clip(query_arc, arc_length[0], arc_length[-1])
    x = np.interp(clipped_arc, arc_length, points_xy[:, 0]).astype(np.float32)
    y = np.interp(clipped_arc, arc_length, points_xy[:, 1]).astype(np.float32)
    interpolated = np.stack([x, y], axis=-1)

    extra_mask = query_arc > arc_length[-1]
    if np.any(extra_mask):
        tail = points_xy[-1] - points_xy[-2]
        tail_norm = float(np.linalg.norm(tail))
        if tail_norm < 1e-4:
            tail_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tail_dir = (tail / tail_norm).astype(np.float32)
        extension = (query_arc[extra_mask] - arc_length[-1])[:, None] * tail_dir[None, :]
        interpolated[extra_mask] = points_xy[-1][None, :] + extension

    return interpolated.astype(np.float32)


def _compute_heading_smooth(points_xy: np.ndarray) -> np.ndarray:
    num_points = len(points_xy)
    headings = np.zeros((num_points,), dtype=np.float32)
    if num_points <= 1:
        return headings
    if num_points == 2:
        heading = np.float32(np.arctan2(points_xy[1, 1] - points_xy[0, 1], points_xy[1, 0] - points_xy[0, 0]))
        headings[:] = heading
        return headings

    deltas = points_xy[2:] - points_xy[:-2]
    headings[1:-1] = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
    headings[0] = headings[1]
    headings[-1] = headings[-2]

    headings = np.unwrap(headings.astype(np.float64))
    if num_points >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        padded = np.pad(headings, (1, 1), mode="edge")
        headings = np.convolve(padded, kernel, mode="valid")

    return headings.astype(np.float32)


def predictions_to_trajectory(
    speed_wps: torch.Tensor,
    route: torch.Tensor,
    speed_dt: float = 0.2,
    time_horizon: float = 4.0,
    trajectory_dt: float = 0.1,
    current_speed_mps: float = 0.0,
    reference_route: Optional[np.ndarray] = None,
    max_speed_mps: Optional[float] = None,
    max_accel_mps2: Optional[float] = None,
    max_decel_mps2: Optional[float] = None,
) -> Trajectory:
    route_xy = _as_numpy_xy(reference_route)
    if route_xy is None:
        route_xy = _as_numpy_xy(route)
    if route_xy is None:
        route_xy = np.zeros((1, 2), dtype=np.float32)
    speed_xy = speed_wps.detach().float().cpu().numpy().astype(np.float32)

    route_polyline = np.concatenate([np.zeros((1, 2), dtype=np.float32), route_xy], axis=0)

    route_arc = _cumulative_arc_length(route_polyline)
    if float(route_arc[-1]) < 1e-4:
        route_polyline = np.concatenate([np.zeros((1, 2), dtype=np.float32), speed_xy], axis=0)
        route_arc = _cumulative_arc_length(route_polyline)

    speed_knots = _decode_speed_knots(speed_xy=speed_xy, speed_dt=speed_dt, current_speed_mps=current_speed_mps)
    target_arc = _extend_speed_profile(
        speed_knots,
        src_dt=speed_dt,
        dst_dt=trajectory_dt,
        horizon=time_horizon,
        max_speed_mps=max_speed_mps,
        max_accel_mps2=max_accel_mps2,
        max_decel_mps2=max_decel_mps2,
    )
    target_xy = _interpolate_polyline(route_polyline, route_arc, target_arc)
    heading = _compute_heading_smooth(target_xy)
    poses = np.concatenate([target_xy, heading[:, None]], axis=-1).astype(np.float32)

    return Trajectory(
        poses=poses,
        trajectory_sampling=TrajectorySampling(time_horizon=time_horizon, interval_length=trajectory_dt),
    )
