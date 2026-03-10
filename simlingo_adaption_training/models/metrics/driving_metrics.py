import logging
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor

from simlingo_adaption_training.utils.custom_types import DrivingInput


logger = logging.getLogger(__name__)


class DrivingMetricsComputer:
    def __init__(self, owner):
        self.owner = owner

    def _pad_position_lists(self, pos_lists, device):
        if len(pos_lists) == 0:
            return None
        max_len = max(len(x) for x in pos_lists)
        if max_len == 0:
            return torch.full((len(pos_lists), 1), -1, device=device, dtype=torch.long)
        out = torch.full((len(pos_lists), max_len), -1, device=device, dtype=torch.long)
        for idx, positions in enumerate(pos_lists):
            if len(positions) > 0:
                out[idx, :len(positions)] = torch.tensor(positions, device=device, dtype=torch.long)
        return out

    def _pad_coord_lists(self, coord_lists, device):
        if len(coord_lists) == 0:
            return None
        max_len = max(len(x) for x in coord_lists)
        if max_len == 0:
            return torch.full((len(coord_lists), 1, 2), -1.0, device=device, dtype=torch.float32)
        out = torch.full((len(coord_lists), max_len, 2), -1.0, device=device, dtype=torch.float32)
        for idx, coords in enumerate(coord_lists):
            if len(coords) > 0:
                out[idx, :len(coords)] = torch.tensor(coords, device=device, dtype=torch.float32)
        return out

    def _build_rule_based_token_positions(self, adaptor_dict: Dict, driving_input: DrivingInput):
        perm = adaptor_dict["perm"]
        inv_perm = perm.argsort(-1)
        split_sizes = adaptor_dict["split_sizes"].tolist()
        language_len = int(split_sizes[0])
        driving_len = int(split_sizes[1]) if len(split_sizes) > 1 else 0
        driving_start = language_len

        language_ids = adaptor_dict["language__ids"]
        img_context_token_id = self.owner.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        visual_pos_lists = []
        visual_coord_lists = []
        num_patches = int(driving_input.camera_images.size(2))
        for b_idx in range(language_ids.size(0)):
            visual_orig = torch.nonzero(language_ids[b_idx] == img_context_token_id, as_tuple=False).squeeze(-1)
            visual_new = inv_perm[b_idx, visual_orig].tolist() if visual_orig.numel() > 0 else []
            visual_pos_lists.append(visual_new)

            coords = []
            if len(visual_new) > 0:
                tokens_per_patch = len(visual_new) // num_patches
                side = int(math.sqrt(tokens_per_patch))
                for token_rank in range(len(visual_new)):
                    patch_id = token_rank // tokens_per_patch
                    local_idx = token_rank % tokens_per_patch
                    local_r = local_idx // side
                    local_c = local_idx % side
                    global_c = patch_id * side + local_c
                    r_norm = local_r / max(side - 1, 1)
                    c_norm = global_c / max(num_patches * side - 1, 1)
                    coords.append([r_norm, c_norm])
            visual_coord_lists.append(coords)

        order = list(self.owner.adaptors.driving.order)
        sizes = self.owner.adaptors.driving.sizes
        path_size = int(sizes.get("route", 0))
        speed_size = int(sizes.get("speed_wps", 0))
        if "route" in order and order.index("route") == 0:
            path_start = driving_start
            speed_start = driving_start + path_size
        else:
            path_start = driving_start
            speed_start = driving_start

        path_pos_lists = []
        speed_pos_lists = []
        for b_idx in range(inv_perm.size(0)):
            if path_size > 0 and driving_len > 0:
                path_orig = torch.arange(path_start, path_start + path_size, device=inv_perm.device)
                path_new = inv_perm[b_idx, path_orig].tolist()
            else:
                path_new = []
            if speed_size > 0 and driving_len > 0:
                speed_orig = torch.arange(speed_start, speed_start + speed_size, device=inv_perm.device)
                speed_new = inv_perm[b_idx, speed_orig].tolist()
            else:
                speed_new = []
            path_pos_lists.append(path_new)
            speed_pos_lists.append(speed_new)

        visual_positions = self._pad_position_lists(visual_pos_lists, inv_perm.device)
        visual_coords = self._pad_coord_lists(visual_coord_lists, inv_perm.device)
        path_positions = self._pad_position_lists(path_pos_lists, inv_perm.device)
        speed_positions = self._pad_position_lists(speed_pos_lists, inv_perm.device)
        return visual_positions, visual_coords, path_positions, speed_positions

    def run_rule_based_probe(
        self,
        adaptor_dict: Dict,
        driving_input: DrivingInput,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Optional[Tensor],
        cache_position: Optional[Tensor],
        latency_value: Optional[Tensor],
        route_keys: Optional[List[str]] = None,
    ):
        (
            visual_token_positions,
            visual_token_coords,
            waypoint_path_token_positions,
            waypoint_speed_token_positions,
        ) = self._build_rule_based_token_positions(adaptor_dict, driving_input)

        history_state = self._prepare_probe_history_state(route_keys, inputs_embeds)
        probe_metrics = self.owner.language_model.model.probe_forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            latency=latency_value,
            visual_token_positions=visual_token_positions,
            visual_token_coords=visual_token_coords,
            waypoint_path_token_positions=waypoint_path_token_positions,
            waypoint_speed_token_positions=waypoint_speed_token_positions,
            history_state=history_state,
            history_alpha=self.owner.probe_history_alpha,
        )
        if route_keys is not None:
            self._commit_probe_history_state(route_keys, history_state)
        return probe_metrics

    def _prepare_probe_history_state(self, route_keys: Optional[List[str]], inputs_embeds: Tensor):
        if route_keys is None:
            return self.owner.probe_history_state

        batch_size = inputs_embeds.size(0)
        if len(route_keys) != batch_size:
            raise ValueError(f"route_keys length ({len(route_keys)}) must match batch size ({batch_size})")
        hidden_size = inputs_embeds.size(-1)
        v_hist = torch.zeros((batch_size, hidden_size), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        v_hist_valid = torch.zeros(batch_size, device=inputs_embeds.device, dtype=torch.bool)

        for b_idx, route_key in enumerate(route_keys):
            if route_key is None:
                continue
            prev_state = self.owner.probe_history_state_by_route.get(route_key, None)
            if not prev_state:
                continue
            prev_hist = prev_state.get("v_hist", None)
            if isinstance(prev_hist, torch.Tensor) and prev_hist.numel() == hidden_size:
                v_hist[b_idx] = prev_hist.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                v_hist_valid[b_idx] = True

        return {"v_hist": v_hist, "v_hist_valid": v_hist_valid}

    def _commit_probe_history_state(self, route_keys: List[str], history_state: Dict[str, Tensor]):
        if not isinstance(history_state, dict):
            return
        hist = history_state.get("v_hist")
        valid = history_state.get("v_hist_valid")
        if not isinstance(hist, torch.Tensor) or hist.ndim != 2:
            return

        if isinstance(valid, torch.Tensor) and valid.numel() == hist.size(0):
            valid_mask = valid.to(device=hist.device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(hist.size(0), device=hist.device, dtype=torch.bool)

        for b_idx, route_key in enumerate(route_keys):
            if route_key is None or not bool(valid_mask[b_idx]):
                continue
            self.owner.probe_history_state_by_route[route_key] = {
                "v_hist": hist[b_idx].detach().clone(),
            }

    def compute_latency_from_probe_metrics(self, metrics, fallback_latency):
        waypoint_entropy = metrics.get("waypoint_entropy", {}) if isinstance(metrics, dict) else {}
        history_similarity = metrics.get("history_similarity", {}) if isinstance(metrics, dict) else {}

        entropy_mean = waypoint_entropy.get("mean_spatial_entropy")
        sim_in = history_similarity.get("sim_in")
        if entropy_mean is None or sim_in is None:
            return fallback_latency

        if not isinstance(entropy_mean, torch.Tensor):
            if isinstance(fallback_latency, torch.Tensor):
                entropy_mean = torch.full_like(fallback_latency, float(entropy_mean))
            else:
                entropy_mean = torch.tensor(float(entropy_mean))
        if not isinstance(sim_in, torch.Tensor):
            if isinstance(fallback_latency, torch.Tensor):
                sim_in = torch.full_like(fallback_latency, float(sim_in))
            else:
                sim_in = torch.tensor(float(sim_in))

        decided_latency = self.owner.probe_entropy_weight * entropy_mean + (1.0 - self.owner.probe_entropy_weight) * sim_in
        return decided_latency.clamp(0.0, 1.0)

    def to_python_metrics(self, outputs):
        metrics = outputs if isinstance(outputs, dict) else getattr(outputs, "rule_based_metrics", None)
        if metrics is None:
            return {}

        def _to_python(value):
            if isinstance(value, dict):
                return {k: _to_python(v) for k, v in value.items()}
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                return float(tensor.item()) if tensor.numel() == 1 else tensor.tolist()
            return value

        return _to_python(metrics)

    def _transform_prev_waypoints_to_curr_frame(self, prev_waypoints: Tensor, delta_xy_prev_frame: Tensor, delta_yaw: Tensor):
        centered = prev_waypoints - delta_xy_prev_frame[:, None, :]
        cos_yaw = torch.cos(delta_yaw)[:, None]
        sin_yaw = torch.sin(delta_yaw)[:, None]
        x_prev = centered[..., 0]
        y_prev = centered[..., 1]
        x_curr = cos_yaw * x_prev + sin_yaw * y_prev
        y_curr = -sin_yaw * x_prev + cos_yaw * y_prev
        return torch.stack([x_curr, y_curr], dim=-1)

    def _time_align_prev_waypoints(self, prev_waypoints_curr_frame: Tensor, delta_tau: Tensor):
        batch_size, waypoint_count, _ = prev_waypoints_curr_frame.shape
        base_idx = torch.arange(waypoint_count, device=prev_waypoints_curr_frame.device, dtype=prev_waypoints_curr_frame.dtype)
        j_star = base_idx[None, :] + (delta_tau[:, None] / self.owner.decision_shift_t_lap)
        k = torch.floor(j_star).long()
        alpha = j_star - k.to(j_star.dtype)
        k_next = k + 1
        valid_mask = (k >= 0) & (k_next < waypoint_count)

        k = k.clamp(0, waypoint_count - 1)
        k_next = k_next.clamp(0, waypoint_count - 1)
        k_idx = k.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        k_next_idx = k_next.unsqueeze(-1).expand(batch_size, waypoint_count, 2)
        wp_k = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_idx)
        wp_k_next = torch.gather(prev_waypoints_curr_frame, dim=1, index=k_next_idx)
        aligned = (1.0 - alpha.unsqueeze(-1)) * wp_k + alpha.unsqueeze(-1) * wp_k_next
        return aligned, valid_mask

    def _compute_decision_shift(self, current_waypoints: Tensor, aligned_prev_waypoints: Tensor, valid_mask: Tensor):
        point_error = torch.linalg.norm(current_waypoints - aligned_prev_waypoints, dim=-1)
        valid_float = valid_mask.to(point_error.dtype)
        valid_count = valid_float.sum(dim=-1)
        error_sum = (point_error * valid_float).sum(dim=-1)
        e_mean = torch.where(
            valid_count > 0,
            error_sum / valid_count,
            torch.full_like(error_sum, float("nan")),
        )
        e_norm = e_mean / (e_mean + 1.0)
        return e_mean, e_norm

    def _decision_shift_for_waypoint_set(
        self,
        current_waypoints: Optional[Tensor],
        prev_waypoints: Optional[Tensor],
        delta_xy_prev_frame: Tensor,
        delta_yaw: Tensor,
        delta_tau: Tensor,
    ):
        if current_waypoints is None or prev_waypoints is None:
            return {"e_mean": None, "e_norm": None}
        prev_curr_frame = self._transform_prev_waypoints_to_curr_frame(prev_waypoints, delta_xy_prev_frame, delta_yaw)
        prev_aligned, valid_mask = self._time_align_prev_waypoints(prev_curr_frame, delta_tau)
        e_mean, e_norm = self._compute_decision_shift(current_waypoints, prev_aligned, valid_mask)
        return {"e_mean": e_mean, "e_norm": e_norm}

    def _update_decision_shift_state(
        self,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        ego_xy: Tensor,
        ego_yaw: Tensor,
        timestamp: Tensor,
    ):
        self.owner.decision_shift_state = {
            "speed_wps": None if current_speed_wps is None else current_speed_wps.detach(),
            "route": None if current_route is None else current_route.detach(),
            "ego_xy": ego_xy.detach(),
            "ego_yaw": ego_yaw.detach(),
            "timestamp": timestamp.detach(),
        }

    def compute_decision_shift_metrics(
        self,
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        route_keys: Optional[List[str]] = None,
    ):
        if route_keys is not None:
            return self._compute_decision_shift_metrics_by_route(
                driving_input=driving_input,
                current_speed_wps=current_speed_wps,
                current_route=current_route,
                route_keys=route_keys,
            )

        default_out = {
            "speed_wps": {"e_mean": None, "e_norm": None},
            "route": {"e_mean": None, "e_norm": None},
            "delta_tau": None,
            "t_lap": self.owner.decision_shift_t_lap,
        }

        ego_xy = driving_input.ego_xy
        ego_yaw = driving_input.ego_yaw
        timestamp = driving_input.timestamp
        if ego_xy is None or ego_yaw is None or timestamp is None:
            if not self.owner._decision_shift_warning_emitted:
                logger.warning("Skip decision_shift in rule_based mode: missing ego_xy/ego_yaw/timestamp in DrivingInput.")
                self.owner._decision_shift_warning_emitted = True
            return default_out

        prev_state = self.owner.decision_shift_state
        if not prev_state:
            self._update_decision_shift_state(current_speed_wps, current_route, ego_xy, ego_yaw, timestamp)
            return default_out

        prev_ego_xy = prev_state["ego_xy"]
        prev_ego_yaw = prev_state["ego_yaw"]
        prev_timestamp = prev_state["timestamp"]

        dx_global = ego_xy[:, 0] - prev_ego_xy[:, 0]
        dy_global = ego_xy[:, 1] - prev_ego_xy[:, 1]
        cos_prev = torch.cos(prev_ego_yaw)
        sin_prev = torch.sin(prev_ego_yaw)
        delta_x_prev = cos_prev * dx_global + sin_prev * dy_global
        delta_y_prev = -sin_prev * dx_global + cos_prev * dy_global
        delta_xy_prev_frame = torch.stack([delta_x_prev, delta_y_prev], dim=-1)
        delta_yaw = ego_yaw - prev_ego_yaw
        delta_tau = timestamp - prev_timestamp

        out = {
            "speed_wps": self._decision_shift_for_waypoint_set(
                current_waypoints=current_speed_wps,
                prev_waypoints=prev_state.get("speed_wps"),
                delta_xy_prev_frame=delta_xy_prev_frame,
                delta_yaw=delta_yaw,
                delta_tau=delta_tau,
            ),
            "route": self._decision_shift_for_waypoint_set(
                current_waypoints=current_route,
                prev_waypoints=prev_state.get("route"),
                delta_xy_prev_frame=delta_xy_prev_frame,
                delta_yaw=delta_yaw,
                delta_tau=delta_tau,
            ),
            "delta_tau": delta_tau,
            "t_lap": torch.full_like(delta_tau, float(self.owner.decision_shift_t_lap)),
        }
        self._update_decision_shift_state(current_speed_wps, current_route, ego_xy, ego_yaw, timestamp)
        return out

    def _compute_decision_shift_metrics_by_route(
        self,
        driving_input: DrivingInput,
        current_speed_wps: Optional[Tensor],
        current_route: Optional[Tensor],
        route_keys: List[str],
    ):
        batch_size = len(route_keys)
        if driving_input.ego_xy is not None and driving_input.ego_xy.size(0) != batch_size:
            raise ValueError(
                f"route_keys length ({batch_size}) must match ego batch size ({driving_input.ego_xy.size(0)})"
            )
        device = None
        for tensor in (current_speed_wps, current_route, driving_input.ego_xy):
            if tensor is not None:
                device = tensor.device
                break
        if device is None:
            device = torch.device("cpu")

        speed_e_mean = torch.full((batch_size,), float("nan"), device=device)
        speed_e_norm = torch.full((batch_size,), float("nan"), device=device)
        route_e_mean = torch.full((batch_size,), float("nan"), device=device)
        route_e_norm = torch.full((batch_size,), float("nan"), device=device)
        delta_tau_out = torch.full((batch_size,), float("nan"), device=device)

        ego_xy = driving_input.ego_xy
        ego_yaw = driving_input.ego_yaw
        timestamp = driving_input.timestamp
        if ego_xy is None or ego_yaw is None or timestamp is None:
            if not self.owner._decision_shift_warning_emitted:
                logger.warning("Skip decision_shift in rule_based mode: missing ego_xy/ego_yaw/timestamp in DrivingInput.")
                self.owner._decision_shift_warning_emitted = True
            return {
                "speed_wps": {"e_mean": speed_e_mean, "e_norm": speed_e_norm},
                "route": {"e_mean": route_e_mean, "e_norm": route_e_norm},
                "delta_tau": delta_tau_out,
                "t_lap": torch.full((batch_size,), float(self.owner.decision_shift_t_lap), device=device),
            }

        for b_idx, route_key in enumerate(route_keys):
            if route_key is None:
                continue

            ego_xy_i = ego_xy[b_idx]
            ego_yaw_i = ego_yaw[b_idx]
            timestamp_i = timestamp[b_idx]
            if (
                not torch.isfinite(ego_xy_i).all()
                or not torch.isfinite(ego_yaw_i)
                or not torch.isfinite(timestamp_i)
            ):
                continue

            prev_state = self.owner.decision_shift_state_by_route.get(route_key, None)
            current_speed_i = None if current_speed_wps is None else current_speed_wps[b_idx]
            current_route_i = None if current_route is None else current_route[b_idx]

            if prev_state:
                prev_ego_xy = prev_state["ego_xy"]
                prev_ego_yaw = prev_state["ego_yaw"]
                prev_timestamp = prev_state["timestamp"]

                dx_global = ego_xy_i[0] - prev_ego_xy[0]
                dy_global = ego_xy_i[1] - prev_ego_xy[1]
                cos_prev = torch.cos(prev_ego_yaw)
                sin_prev = torch.sin(prev_ego_yaw)
                delta_x_prev = cos_prev * dx_global + sin_prev * dy_global
                delta_y_prev = -sin_prev * dx_global + cos_prev * dy_global
                delta_xy_prev_frame = torch.stack([delta_x_prev, delta_y_prev], dim=-1).unsqueeze(0)
                delta_yaw = (ego_yaw_i - prev_ego_yaw).unsqueeze(0)
                delta_tau = (timestamp_i - prev_timestamp).unsqueeze(0)
                delta_tau_out[b_idx] = delta_tau[0]

                speed_metrics = self._decision_shift_for_waypoint_set(
                    current_waypoints=None if current_speed_i is None else current_speed_i.unsqueeze(0),
                    prev_waypoints=None if prev_state.get("speed_wps") is None else prev_state["speed_wps"].unsqueeze(0),
                    delta_xy_prev_frame=delta_xy_prev_frame,
                    delta_yaw=delta_yaw,
                    delta_tau=delta_tau,
                )
                route_metrics = self._decision_shift_for_waypoint_set(
                    current_waypoints=None if current_route_i is None else current_route_i.unsqueeze(0),
                    prev_waypoints=None if prev_state.get("route") is None else prev_state["route"].unsqueeze(0),
                    delta_xy_prev_frame=delta_xy_prev_frame,
                    delta_yaw=delta_yaw,
                    delta_tau=delta_tau,
                )

                if speed_metrics["e_mean"] is not None:
                    speed_e_mean[b_idx] = speed_metrics["e_mean"][0]
                    speed_e_norm[b_idx] = speed_metrics["e_norm"][0]
                if route_metrics["e_mean"] is not None:
                    route_e_mean[b_idx] = route_metrics["e_mean"][0]
                    route_e_norm[b_idx] = route_metrics["e_norm"][0]

            self.owner.decision_shift_state_by_route[route_key] = {
                "speed_wps": None if current_speed_i is None else current_speed_i.detach().clone(),
                "route": None if current_route_i is None else current_route_i.detach().clone(),
                "ego_xy": ego_xy_i.detach().clone(),
                "ego_yaw": ego_yaw_i.detach().clone(),
                "timestamp": timestamp_i.detach().clone(),
            }

        return {
            "speed_wps": {"e_mean": speed_e_mean, "e_norm": speed_e_norm},
            "route": {"e_mean": route_e_mean, "e_norm": route_e_norm},
            "delta_tau": delta_tau_out,
            "t_lap": torch.full((batch_size,), float(self.owner.decision_shift_t_lap), device=device),
        }
