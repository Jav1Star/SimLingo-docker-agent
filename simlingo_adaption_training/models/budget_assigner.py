from collections import deque
import copy
import random
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from torch import nn

from simlingo_adaption_training.models.metrics.driving_metrics import DrivingMetricsComputer
from simlingo_adaption_training.utils.custom_types import DrivingInput


class BudgetAssigner(nn.Module):
    def __init__(
        self,
        mode: str = "fixed",
        fixed_budget: float = 1.0,
        rule_based_cfg: Optional[Dict[str, Any]] = None,
        random_min: float = 0.25,
        random_max: float = 1.0,
        history_alpha: float = 0.6,
        decision_shift_t_lap: float = 0.2,
    ) -> None:
        """Initialize budget policy, per-route state, and per-step runtime caches."""
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.fixed_budget = float(np.clip(fixed_budget, 0.0, 1.0))
        self.random_min = float(min(random_min, random_max))
        self.random_max = float(max(random_min, random_max))
        self.history_alpha = float(np.clip(history_alpha, 0.0, 1.0))
        self.decision_shift_t_lap = float(decision_shift_t_lap)

        self.rule_based_cfg = None
        if self.mode == "rule_based":
            if not rule_based_cfg:
                raise ValueError("rule_based mode requires non-empty rule_based_cfg")
            self.rule_based_cfg = copy.deepcopy(rule_based_cfg)
        self.state_by_route: Dict[str, Dict[str, Any]] = {}
        self.prev_decision_state_by_route: Dict[str, Dict[str, Any]] = {}
        self.last_decision_info: List[Dict[str, Any]] = []
        self.last_update_info: List[Dict[str, Any]] = []
        self.last_scene_metrics: Dict[str, Any] = {}
        self.metrics_computer = DrivingMetricsComputer(self)

        # Runtime states (one LLM forward step)
        self.scheduler: Optional[nn.Module] = None
        self._runtime_ref_tensor: Optional[torch.Tensor] = None
        self._runtime_budget_values: Optional[torch.Tensor] = None
        self._runtime_budget_token_position: Optional[torch.LongTensor] = None
        self._runtime_execution_plan: Optional[torch.Tensor] = None

    def set_policy(
        self,
        mode: str,
        fixed_budget: Optional[float] = None,
        rule_based_cfg: Optional[Dict[str, Any]] = None,
        reset_state: bool = True,
    ) -> None:
        """Update policy config; optionally reset route/runtime states."""
        self.mode = str(mode).strip().lower()

        if fixed_budget is not None:
            self.fixed_budget = float(np.clip(fixed_budget, 0.0, 1.0))

        if self.mode == "rule_based":
            if not rule_based_cfg:
                raise ValueError("rule_based mode requires non-empty rule_based_cfg")
            self.rule_based_cfg = copy.deepcopy(rule_based_cfg)
        elif rule_based_cfg is not None:
            self.rule_based_cfg = copy.deepcopy(rule_based_cfg)

        if reset_state:
            self.reset()

    def build_scheduler(self, scheduler_cfg: Any, language_model_config: Any) -> Optional[nn.Module]:
        """Instantiate scheduler with LLM-aligned dims and save to `self.scheduler`."""
        if scheduler_cfg is None:
            self.scheduler = None
            return None

        cfg = copy.deepcopy(scheduler_cfg)
        aligned_vals = {
            "num_hidden_layers": int(language_model_config.num_hidden_layers),
            "num_attention_heads": int(language_model_config.num_attention_heads),
            "hidden_size": int(language_model_config.hidden_size),
        }

        for key, value in aligned_vals.items():
            setattr(cfg, key, value)

        self.scheduler = hydra.utils.instantiate(cfg, _recursive_=False)
        return self.scheduler

    def _clear_runtime_context(self) -> None:
        """Clear per-step runtime caches (`_runtime_*`) kept for one forward pass."""
        self._runtime_ref_tensor = None
        self._runtime_budget_values = None
        self._runtime_budget_token_position = None
        self._runtime_execution_plan = None

    def on_llm_forward_start(self) -> None:
        """Mark execution plan stale; it will be rebuilt on first prefix boundary layer."""
        self._runtime_execution_plan = None

    def reset(self, route_keys: Optional[List[str]] = None) -> None:
        """Reset route states; also clears runtime caches."""
        if route_keys is None:
            self.state_by_route = {}
            self.prev_decision_state_by_route = {}
            self.last_decision_info = []
            self.last_update_info = []
            self.last_scene_metrics = {}
            self._clear_runtime_context()
            return
        for route_key in route_keys:
            key = self._normalize_route_key(route_key)
            self.state_by_route.pop(key, None)
            self.prev_decision_state_by_route.pop(key, None)
        self._clear_runtime_context()

    def bind_runtime_reference(self, ref_tensor: torch.Tensor) -> None:
        """Bind step reference tensor to infer runtime device/dtype for budget tensors.

        Saved state: `_runtime_ref_tensor`.
        """
        self._runtime_ref_tensor = ref_tensor

    def set_runtime_budget_values(self, budget_tensor: torch.Tensor) -> None:
        """Save decided budget values for current step (`_runtime_budget_values`)."""
        self._runtime_budget_values = budget_tensor.detach()

    def get_runtime_budget_values(self, batch_size: int) -> torch.Tensor:
        """Return current step budget tensor normalized to runtime ref device/dtype.

        Read/Write state: `_runtime_ref_tensor`, `_runtime_budget_values`.
        """
        budget_tensor = self._to_budget_tensor(
            self._runtime_budget_values,
            batch_size,
            ref_tensor=self._runtime_ref_tensor,
        )
        self._runtime_budget_values = budget_tensor.detach()
        return budget_tensor

    def encode_budget_token(self, budget_tensor: torch.Tensor) -> torch.Tensor:
        """Encode budget scalar(s) into budget token embeddings via scheduler encoder."""
        self.set_runtime_budget_values(budget_tensor)
        return self.scheduler.budget_encoding(budget_tensor)

    def _resolve_budget_token_position(self, adaptor_dict: Dict[str, torch.Tensor]) -> torch.LongTensor:
        """Locate budget token position after adaptor permutation for each sample."""
        perm = adaptor_dict["perm"]
        inv_perm = perm.argsort(-1)
        budget_orig_indices = adaptor_dict["budget_orig_indices"].to(device=perm.device, dtype=torch.long)
        return inv_perm[:, budget_orig_indices][:, 0].long()

    def prepare_llm_runtime(self, adaptor_dict: Dict[str, torch.Tensor]) -> None:
        """Prepare LLM-time caches: budget token position and execution plan reset."""
        self._runtime_budget_token_position = self._resolve_budget_token_position(adaptor_dict)
        self._runtime_execution_plan = None

    def build_execution_plan_on_prefix_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Optional[torch.Tensor]:
        """Build full execution plan once at prefix boundary and cache it."""
        if self.scheduler is None:
            return None
        if self._runtime_budget_values is None or self._runtime_budget_token_position is None:
            return None
        if layer_idx != int(num_prefix_layers):
            return self._runtime_execution_plan

        budget_pos = self._runtime_budget_token_position.to(device=hidden_states.device, dtype=torch.long)
        if budget_pos.ndim == 0:
            budget_pos = budget_pos.unsqueeze(0)
        if budget_pos.size(0) == 1:
            budget_pos = budget_pos.expand(hidden_states.size(0))
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        budget_token_feat = hidden_states[batch_indices, budget_pos]

        budget_values = self._to_budget_tensor(
            self._runtime_budget_values,
            hidden_states.size(0),
            ref_tensor=hidden_states,
            out_dtype=torch.float32,
        )
        self._runtime_execution_plan = self.scheduler(
            budget_token_feat.contiguous(), budget_values
        ).transpose(0, 1)
        return self._runtime_execution_plan

    def get_execution_plan(self) -> Optional[torch.Tensor]:
        """Return cached execution plan built at prefix boundary."""
        return self._runtime_execution_plan

    def get_drop_states_from_execution_plan(
        self,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Optional[torch.Tensor]:
        """Read one layer drop-state from cached execution plan."""
        plan = self._runtime_execution_plan
        if plan is None:
            return None
        if layer_idx < int(num_prefix_layers) or layer_idx >= int(plan.size(0)):
            return None
        return plan[layer_idx]

    def get_drop_states_for_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        num_prefix_layers: int,
    ) -> Optional[torch.Tensor]:
        """Build/read scheduler execution plan and return drop states for one decoder layer.

        Read/Write state: `_runtime_budget_values`, `_runtime_budget_token_position`, `_runtime_execution_plan`.
        """
        if self.scheduler is None:
            return None

        if self._runtime_budget_values is None or self._runtime_budget_token_position is None:
            return None

        self.build_execution_plan_on_prefix_layer(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            num_prefix_layers=num_prefix_layers,
        )
        return self.get_drop_states_from_execution_plan(
            layer_idx=layer_idx,
            num_prefix_layers=num_prefix_layers,
        )

    def budget_decide_before_llm(
        self,
        batch_size: int,
        route_keys: List[str],
    ) -> torch.Tensor:
        """Decide budget before LLM and persist per-route + runtime budget states.

        Read/Write state: `state_by_route`, `last_decision_info`, `_runtime_budget_values`.
        """
        budgets: List[float] = []
        decision_rows: List[Dict[str, Any]] = []

        for route_key in route_keys:
            state = self._get_state(route_key)
            if self.mode == "fixed":
                value = self.fixed_budget
                state["value"] = value
                state["used_budget"] = value
                state["phase"] = "fixed"
            elif self.mode == "random":
                value = float(random.uniform(self.random_min, self.random_max))
                value = float(np.clip(value, 0.0, 1.0))
                state["value"] = value
                state["used_budget"] = value
                state["phase"] = "random"
            else:
                value = self._rule_based_pre_decide(state)

            budgets.append(value)
            decision_rows.append(
                {
                    "route_key": route_key,
                    "mode": self.mode,
                    "phase": state["phase"],
                    "used_budget": float(state["used_budget"]),
                    "value": float(state["value"]),
                    "base_budget": state["base_budget"],
                    "instant_budget": state["instant_budget"],
                    "target_budget": state["target_budget"],
                    "frame_count": int(state["frame_count"]),
                    "safe_counter": int(state["safe_counter"]),
                }
            )

        budget_tensor = self._to_budget_tensor(
            budgets,
            batch_size,
            ref_tensor=self._runtime_ref_tensor,
        )
        self.last_decision_info = decision_rows
        self.set_runtime_budget_values(budget_tensor)
        return budget_tensor

    def budget_info_update_after_llm(
        self,
        route_keys: List[str],
        driving_input: DrivingInput,
        current_speed_wps: Optional[torch.Tensor],
        current_route: Optional[torch.Tensor],
        adaptor_dict: Optional[Dict[str, torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Update post-LLM novelty/shift history and expose scene-difficulty metrics.

        Read/Write state: per-route `prev_*`, `v_hist`, and `last_update_info`.
        """
        novelty_metrics = self.metrics_computer.compute_novelty_metrics(
            route_keys=route_keys,
            adaptor_dict=adaptor_dict,
            inputs_embeds=inputs_embeds,
            tokenizer=tokenizer,
        )
        novelty_list = novelty_metrics["novelty"]
        sim_in_list = novelty_metrics["history_similarity"]["sim_in"]

        decision_shift = {
            "speed_wps": {"e_mean": [None] * len(route_keys), "e_norm": [None] * len(route_keys)},
            "route": {"e_mean": [None] * len(route_keys), "e_norm": [None] * len(route_keys)},
            "delta_tau": [None] * len(route_keys),
            "t_lap": [self.decision_shift_t_lap] * len(route_keys),
        }
        if self.mode == "rule_based":
            decision_shift_raw = self.metrics_computer.compute_decision_shift_metrics(
                driving_input=driving_input,
                current_speed_wps=current_speed_wps,
                current_route=current_route,
                route_keys=route_keys,
            )
            decision_shift = self.metrics_computer.to_python_metrics(decision_shift_raw)

        update_rows: List[Dict[str, Any]] = []
        used_budget_list: List[float] = []
        phase_list: List[str] = []
        base_budget_list: List[Optional[float]] = []
        instant_budget_list: List[Optional[float]] = []
        target_budget_list: List[Optional[float]] = []

        speed_shift_list = decision_shift["speed_wps"]["e_norm"]
        route_shift_list = decision_shift["route"]["e_norm"]

        for idx, route_key in enumerate(route_keys):
            state = self._get_state(route_key)
            speed_curr = speed_shift_list[idx]
            route_curr = route_shift_list[idx]
            novelty_curr = novelty_list[idx]

            if novelty_curr is not None and not np.isnan(novelty_curr):
                state["prev_novelty"] = float(novelty_curr)
            if speed_curr is not None and not np.isnan(speed_curr):
                state["prev_decision_shift_speed"] = float(speed_curr)
            if route_curr is not None and not np.isnan(route_curr):
                state["prev_decision_shift_route"] = float(route_curr)

            used_budget = float(state["used_budget"])
            used_budget_list.append(float(used_budget))
            phase_list.append(str(state["phase"]))
            base_budget_list.append(state["base_budget"])
            instant_budget_list.append(state["instant_budget"])
            target_budget_list.append(state["target_budget"])

            update_rows.append(
                {
                    "route_key": route_key,
                    "novelty": novelty_curr,
                    "history_similarity": sim_in_list[idx],
                    "decision_shift_speed": speed_curr,
                    "decision_shift_route": route_curr,
                    "used_budget": float(used_budget),
                    "phase": str(state["phase"]),
                    "base_budget": state["base_budget"],
                    "instant_budget": state["instant_budget"],
                    "target_budget": state["target_budget"],
                    "frame_count": int(state["frame_count"]),
                    "safe_counter": int(state["safe_counter"]),
                }
            )

        out = {
            "novelty": novelty_list,
            "history_similarity": {
                "sim_in": sim_in_list,
                "alpha": float(self.history_alpha),
            },
            "decision_shift": decision_shift,
            "used_budget": used_budget_list,
            "budget_info": {
                "mode": self.mode,
                "phase": phase_list,
                "base_budget": base_budget_list,
                "instant_budget": instant_budget_list,
                "target_budget": target_budget_list,
            },
        }
        self.last_scene_metrics = out
        self.last_update_info = update_rows
        return out

    def get_eval_budget(self) -> Dict[str, Any]:
        """Return compact eval snapshot with current mode and latest decision/update info."""
        return {
            "mode": self.mode,
            "fixed_budget": float(self.fixed_budget),
            "last_decision": self.last_decision_info,
            "last_update": self.last_update_info,
            "last_scene_metrics": self.last_scene_metrics,
        }

    def _rule_based_pre_decide(self, state: Dict[str, Any]) -> float:
        """Rule-based pre-LLM budget update using previous-step novelty and decision-shift."""
        cfg = self.rule_based_cfg
        n_prev = float(state["prev_novelty"]) if state["prev_novelty"] is not None else 0.0
        s_prev = float(state["prev_decision_shift_speed"]) if state["prev_decision_shift_speed"] is not None else 0.0
        r_prev = float(state["prev_decision_shift_route"]) if state["prev_decision_shift_route"] is not None else 0.0

        n_hat = self._normalize_metric(n_prev, cfg["normalization"]["novelty"])
        s_hat = self._normalize_metric(s_prev, cfg["normalization"]["speed_shift"])
        r_hat = self._normalize_metric(r_prev, cfg["normalization"]["route_shift"])

        state["window_novelty"].append(n_hat)
        state["window_speed_shift"].append(s_hat)
        state["window_route_shift"].append(r_hat)
        state["frame_count"] += 1

        inst_weights = cfg["weights"]["inst"]
        u_t = float(
            np.clip(
                inst_weights["novelty"] * n_hat
                + inst_weights["speed_shift"] * s_hat
                + inst_weights["route_shift"] * r_hat,
                0.0,
                1.0,
            )
        )
        inst_budget = float(np.clip(cfg["inst_offset"] + cfg["inst_scale"] * u_t, 0.0, 1.0))
        state["instant_budget"] = float(inst_budget)

        used_budget = float(state["value"])
        next_budget = float(used_budget)
        target_budget = None

        k_warmup = int(cfg["k_warmup"])
        frame_count = int(state["frame_count"])
        if frame_count <= k_warmup:
            next_budget = 1.0
            phase = "warmup"
        elif state["base_budget"] is None:
            g_init = self._compute_base_g_score(
                state["window_novelty"],
                state["window_speed_shift"],
                state["window_route_shift"],
                cfg,
            )
            state["base_budget"] = float(np.clip(cfg["base_offset"] + cfg["base_scale"] * g_init, 0.0, 1.0))
            next_budget = float(state["base_budget"])
            phase = "adaptive"
        else:
            g_tilde = self._compute_base_g_score(
                state["window_novelty"],
                state["window_speed_shift"],
                state["window_route_shift"],
                cfg,
            )
            tilde_base_budget = float(np.clip(cfg["base_offset"] + cfg["base_scale"] * g_tilde, 0.0, 1.0))
            state["base_budget"] = float(
                np.clip(
                    (1.0 - cfg["eta"]) * state["base_budget"] + cfg["eta"] * tilde_base_budget,
                    0.0,
                    1.0,
                )
            )
            target_budget = max(float(state["base_budget"]), float(inst_budget))

            if target_budget > used_budget:
                next_budget = float(target_budget)
                state["safe_counter"] = 0
            else:
                safe_condition = ((s_hat + r_hat) * 0.5 <= cfg["safe_threshold"]) and (
                    target_budget <= float(state["base_budget"])
                )
                state["safe_counter"] = state["safe_counter"] + 1 if safe_condition else 0
                if state["safe_counter"] > int(cfg["safe_count_threshold"]):
                    next_budget = max(float(state["base_budget"]), used_budget - cfg["decay_step"])
                else:
                    next_budget = float(used_budget)
            phase = "adaptive"

        state["value"] = float(np.clip(next_budget, 0.0, 1.0))
        state["used_budget"] = float(state["value"])
        state["phase"] = phase
        state["target_budget"] = None if target_budget is None else float(target_budget)
        return float(state["value"])

    def _get_state(self, route_key: str) -> Dict[str, Any]:
        """Get or lazily create per-route state entry in `state_by_route`."""
        key = self._normalize_route_key(route_key)
        if key not in self.state_by_route:
            self.state_by_route[key] = self._init_state()
        return self.state_by_route[key]

    def _init_state(self) -> Dict[str, Any]:
        """Create default per-route state container for budget decision/update."""
        k_warmup = int(self.rule_based_cfg["k_warmup"]) if self.mode == "rule_based" else 1
        init_budget = 1.0 if self.mode == "rule_based" else self.fixed_budget
        return {
            "value": float(init_budget),
            "used_budget": float(init_budget),
            "base_budget": None,
            "instant_budget": None,
            "target_budget": None,
            "phase": "warmup" if self.mode == "rule_based" else self.mode,
            "frame_count": 0,
            "safe_counter": 0,
            "prev_novelty": 0.0,
            "prev_decision_shift_speed": 0.0,
            "prev_decision_shift_route": 0.0,
            "window_novelty": deque(maxlen=max(k_warmup, 1)),
            "window_speed_shift": deque(maxlen=max(k_warmup, 1)),
            "window_route_shift": deque(maxlen=max(k_warmup, 1)),
            "v_hist": None,
        }

    @staticmethod
    def _normalize_route_key(route_key: Optional[str]) -> str:
        """Normalize route key to stable string; empty route id is invalid."""
        if route_key is None or str(route_key).strip() == "":
            raise ValueError("route_key must be a non-empty string")
        return str(route_key)

    @classmethod
    def _resolve_route_keys(cls, route_keys: Optional[List[str]], batch_size: int) -> List[str]:
        """Validate and normalize route key list to batch size."""
        if route_keys is None:
            raise ValueError("route_keys is required")
        if len(route_keys) != batch_size:
            raise ValueError(f"route_keys length ({len(route_keys)}) must match batch size ({batch_size})")
        return [cls._normalize_route_key(k) for k in route_keys]

    @staticmethod
    def _to_budget_tensor(
        budget: Any,
        batch_size: int,
        ref_tensor: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert scalar/list/tensor budget to shape [B] on ref tensor's device/dtype."""
        target_device = ref_tensor.device
        if out_dtype is not None:
            target_dtype = out_dtype
        elif torch.is_floating_point(ref_tensor):
            target_dtype = ref_tensor.dtype
        else:
            target_dtype = torch.float32

        budget_tensor = torch.as_tensor(budget, device=target_device, dtype=target_dtype)
        if budget_tensor.ndim == 0:
            return budget_tensor.repeat(batch_size)
        return budget_tensor.reshape(batch_size)

    def _normalize_metric(self, value: float, quantile_cfg: Dict[str, Any]) -> float:
        """Normalize metric with configured q10/q90 and clamp to [0, 1]."""
        q10 = float(quantile_cfg["q10"])
        q90 = float(quantile_cfg["q90"])
        denom = max(q90 - q10, 1e-6)
        return float(np.clip((float(value) - q10) / denom, 0.0, 1.0))

    def _compute_base_g_score(
        self,
        novelty_vals: deque,
        speed_vals: deque,
        route_vals: deque,
        cfg: Dict[str, Any],
    ) -> float:
        """Compute base complexity score from window means/maxima."""
        n = np.asarray(list(novelty_vals), dtype=np.float32)
        s = np.asarray(list(speed_vals), dtype=np.float32)
        r = np.asarray(list(route_vals), dtype=np.float32)
        if n.size == 0 or s.size == 0 or r.size == 0:
            return 0.0

        base_mean_weights = cfg["weights"]["base_mean"]
        base_max_weights = cfg["weights"]["base_max"]
        g_t = (
            base_mean_weights["novelty"] * float(n.mean())
            + base_mean_weights["speed_shift"] * float(s.mean())
            + base_mean_weights["route_shift"] * float(r.mean())
            + base_max_weights["novelty"] * float(n.max())
            + base_max_weights["speed_shift"] * float(s.max())
            + base_max_weights["route_shift"] * float(r.max())
        )
        return float(np.clip(g_t, 0.0, 1.0))
