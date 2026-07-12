from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig

from simlingo_adaption_training.models.budget_assigner import BaseBudgetAssigner, build_budget_assigner
from utils.logger_utils import get_logger


logger = get_logger(__name__)


SCHEDULER_NUM_PREFIX_LAYERS = 10

DEFAULT_RULE_BASED_CFG = {
    "k_warmup": 5,
    "eta": 0.03,
    "safe_threshold": 0.33,
    "safe_count_threshold": 2,
    "decay_step": 0.1,
    "base_offset": 0.25,
    "base_scale": 0.75,
    "inst_offset": 0.5,
    "inst_scale": 0.5,
    "weights": {
        "base_mean": {"novelty": 0.29, "speed_shift": 0.25, "route_shift": 0.19},
        "base_max": {"novelty": 0.09, "speed_shift": 0.11, "route_shift": 0.08},
        "inst": {"novelty": 0.42, "speed_shift": 0.33, "route_shift": 0.25},
    },
    "normalization": {
        "novelty": {"q10": 0.0001102686, "q90": 0.0131252408},
        "speed_shift": {"q10": 0.0030981766, "q90": 0.7412269711},
        "route_shift": {"q10": 0.1959435195, "q90": 0.7455124855},
    },
}


class SchedulerRuntime:
    """Owns budget policy and execution-plan generation in two scheduler phases."""

    def __init__(self) -> None:
        self._model_variant: Optional[str] = None
        self._assigner: BaseBudgetAssigner | None = None
        self._scheduler_target: Optional[str] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float32
        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._hidden_size: Optional[int] = None
        self._num_hidden_layers: Optional[int] = None
        self._num_attention_heads: Optional[int] = None
        self._num_prefix_layers: Optional[int] = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._assigner is not None and self._assigner.scheduler is not None

    def load_model(
        self,
        model_variant: str,
        *,
        checkpoint_path: Optional[str] = None,
        mode: str = "rule_based",
        fixed_budget: float = 0.8,
        history_alpha: float = 0.6,
        decision_shift_t_lap: float = 0.2,
        scheduler_target: str,
        tau: float = 5.0,
        is_hard: bool = True,
        threshold: float = 0.5,
        bias: bool = True,
        num_prefix_layers: int = SCHEDULER_NUM_PREFIX_LAYERS,
        rule_based_cfg_json: Optional[str] = None,
    ) -> None:
        num_prefix_layers = int(num_prefix_layers)

        if self.is_loaded and self._model_variant == model_variant:
            return

        with self._state_lock:
            if self.is_loaded and self._model_variant == model_variant:
                return

            logger.info("Loading SimLingo scheduler from %s", model_variant)
            llm_config = AutoConfig.from_pretrained(model_variant, trust_remote_code=True)
            llm_cfg = getattr(llm_config, "llm_config", llm_config)

            rule_based_cfg = self._resolve_rule_based_cfg(mode, rule_based_cfg_json)
            assigner = build_budget_assigner(
                mode=mode,
                fixed_budget=fixed_budget,
                rule_based_cfg=rule_based_cfg,
                history_alpha=history_alpha,
                decision_shift_t_lap=decision_shift_t_lap,
            )

            scheduler_cfg = OmegaConf.create(
                {
                    "_target_": scheduler_target,
                    "tau": float(tau),
                    "is_hard": bool(is_hard),
                    "threshold": float(threshold),
                    "bias": bool(bias),
                    "num_prefix_layers": int(num_prefix_layers),
                }
            )
            language_cfg = SimpleNamespace(
                hidden_size=int(llm_cfg.hidden_size),
                num_hidden_layers=int(llm_cfg.num_hidden_layers),
                num_attention_heads=int(llm_cfg.num_attention_heads),
            )
            scheduler_module = assigner.build_scheduler(scheduler_cfg, language_cfg)
            if scheduler_module is None:
                raise RuntimeError("Failed to build scheduler module")
            scheduler_module.to(device=self._device, dtype=self._dtype)
            scheduler_module.eval()

            if checkpoint_path:
                self._load_scheduler_weights(
                    checkpoint_path=checkpoint_path,
                    scheduler_module=scheduler_module,
                )

            self._model_variant = model_variant
            self._assigner = assigner
            self._scheduler_target = scheduler_target
            self._hidden_size = int(llm_cfg.hidden_size)
            self._num_hidden_layers = int(llm_cfg.num_hidden_layers)
            self._num_attention_heads = int(llm_cfg.num_attention_heads)
            self._num_prefix_layers = int(num_prefix_layers)
            logger.info(
                "Scheduler loaded: hidden_size=%s num_hidden_layers=%s num_attention_heads=%s num_prefix_layers=%s mode=%s",
                self._hidden_size,
                self._num_hidden_layers,
                self._num_attention_heads,
                self._num_prefix_layers,
                mode,
            )

    def _resolve_rule_based_cfg(self, mode: str, rule_based_cfg_json: Optional[str]) -> Optional[dict[str, Any]]:
        if str(mode).strip().lower() != "rule_based":
            return None
        if not rule_based_cfg_json:
            return copy.deepcopy(DEFAULT_RULE_BASED_CFG)
        loaded = json.loads(rule_based_cfg_json)
        if not isinstance(loaded, dict):
            raise ValueError("SCHEDULER_RULE_BASED_CFG_JSON must decode to a JSON object")
        return loaded

    def _load_scheduler_weights(
        self,
        *,
        checkpoint_path: str,
        scheduler_module: torch.nn.Module,
    ) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SCHEDULER_CHECKPOINT_PATH does not exist: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint must be a state_dict or contain a state_dict key")

        prefixes = ("budget_assigner.scheduler.", "scheduler.")
        scheduler_state: dict[str, Any] = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    bare_key = key[len(prefix):]
                    scheduler_state.setdefault(bare_key, value)
                    break

        if not scheduler_state:
            logger.warning("No scheduler weights found with prefixes %s", prefixes)
            return

        scheduler_state = self._filter_compatible_state_dict(
            scheduler_state,
            scheduler_module,
            module_name="scheduler",
        )
        if not scheduler_state:
            logger.warning("No compatible scheduler weights found with prefixes %s", prefixes)
            return

        missing, unexpected = scheduler_module.load_state_dict(scheduler_state, strict=False)
        logger.info(
            "Loaded scheduler partial weights: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

    def _filter_compatible_state_dict(
        self,
        state_dict: dict[str, Any],
        module: torch.nn.Module,
        *,
        module_name: str,
    ) -> dict[str, Any]:
        target_state = module.state_dict()
        compatible: dict[str, Any] = {}
        skipped: list[str] = []
        for key, value in state_dict.items():
            target_value = target_state.get(key)
            if target_value is None:
                skipped.append(f"{key}: unexpected")
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                skipped.append(f"{key}: checkpoint{tuple(value.shape)} != runtime{tuple(target_value.shape)}")
                continue
            compatible[key] = value

        if skipped:
            preview = "; ".join(skipped[:5])
            logger.warning(
                "Skipped %d incompatible %s weights while forcing num_prefix_layers=%d: %s",
                len(skipped),
                module_name,
                int(getattr(module, "num_prefix_layers", SCHEDULER_NUM_PREFIX_LAYERS)),
                preview,
            )
        return compatible

    def _require_loaded(self) -> BaseBudgetAssigner:
        if self._assigner is None or self._assigner.scheduler is None:
            raise RuntimeError("scheduler model has not been loaded yet")
        return self._assigner

    def run(self, payload: dict[str, Any], phase: str = "budget") -> dict[str, Any]:
        assigner = self._require_loaded()
        phase = str(phase).strip().lower()
        if phase == "budget":
            return self._run_budget_phase(assigner, payload)
        if phase == "plan":
            return self._run_plan_phase(assigner, payload)
        if phase == "decision_update":
            return self._run_decision_update_phase(assigner, payload)
        raise ValueError(f"Unsupported scheduler phase: {phase}")

    def _run_budget_phase(self, assigner: BaseBudgetAssigner, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_payload = payload.get("encoded_payload") if isinstance(payload.get("encoded_payload"), dict) else payload
        if not isinstance(encoded_payload, dict):
            raise ValueError("scheduler budget phase requires an 'encoded_payload' object")

        tokenized = encoded_payload.get("tokenized")
        if not isinstance(tokenized, dict):
            raise ValueError("encoded_payload must contain a 'tokenized' object")
        input_ids = tokenized.get("input_ids")
        visual_embeds = encoded_payload.get("visual_embeds")
        if input_ids is None:
            raise ValueError("encoded_payload.tokenized.input_ids is required")
        if visual_embeds is None:
            raise ValueError("encoded_payload.visual_embeds is required")

        input_ids_np = np.asarray(input_ids, dtype=np.int64)
        visual_embeds_np = np.asarray(visual_embeds)
        if input_ids_np.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, S], got {input_ids_np.shape}")
        batch_size = int(input_ids_np.shape[0])
        route_keys = self._resolve_route_keys(encoded_payload, batch_size)

        visual_summary = self._summarize_visual_embeds(visual_embeds_np, batch_size)
        novelty_metrics = self._update_scene_state(assigner, route_keys, visual_summary)

        ref_tensor = torch.zeros((batch_size, 1), device=self._device, dtype=self._dtype)
        assigner.bind_runtime_reference(ref_tensor)

        with self._model_lock:
            with torch.no_grad():
                budget_tensor = assigner.budget_decide_before_llm(
                    batch_size=batch_size,
                    route_keys=route_keys,
                ).to(device=self._device, dtype=torch.float32)

        return {
            "status": "success",
            "phase": "budget",
            "encoded_payload": encoded_payload,
            "budget_value": budget_tensor.detach().float().cpu().numpy(),
            "scheduler_meta": {
                "mode": assigner.mode,
                "route_keys": route_keys,
                "novelty": novelty_metrics["novelty"],
                "history_similarity": novelty_metrics["history_similarity"],
                "decision": assigner.last_decision_info,
                "eval_budget": assigner.get_eval_budget(),
            },
        }

    def _run_plan_phase(self, assigner: BaseBudgetAssigner, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_payload = payload.get("encoded_payload")
        budget_value = payload.get("budget_value")
        prefix_feature = payload.get("budget_token_prefix_feature")
        if not isinstance(encoded_payload, dict):
            raise ValueError("scheduler plan phase requires an 'encoded_payload' object")
        if budget_value is None:
            raise ValueError("scheduler plan phase requires 'budget_value'")
        if prefix_feature is None:
            raise ValueError("scheduler plan phase requires 'budget_token_prefix_feature'")

        route_keys = self._resolve_route_keys(encoded_payload, batch_size=np.asarray(budget_value).reshape(-1).shape[0] or 1)
        budget_value_np = self._validate_budget_value(np.asarray(budget_value, dtype=np.float32), expected_batch=len(route_keys))
        prefix_feature_np = np.asarray(prefix_feature, dtype=np.float32)
        if prefix_feature_np.ndim != 2 or prefix_feature_np.shape[0] != len(route_keys):
            raise ValueError(
                f"budget_token_prefix_feature must have shape [B, H], got {prefix_feature_np.shape}"
            )
        if self._hidden_size is not None and prefix_feature_np.shape[1] != self._hidden_size:
            raise ValueError(
                f"budget_token_prefix_feature hidden size ({prefix_feature_np.shape[1]}) must match scheduler hidden size "
                f"({self._hidden_size})"
            )

        with self._model_lock:
            with torch.no_grad():
                prefix_feature_tensor = torch.as_tensor(prefix_feature_np, device=self._device, dtype=self._dtype)
                budget_tensor = torch.as_tensor(budget_value_np, device=self._device, dtype=torch.float32)
                execution_plan = assigner.scheduler(prefix_feature_tensor.contiguous(), budget_tensor)
                assigner._runtime_execution_plan = execution_plan.transpose(0, 1)
                plan_debug = self._build_plan_debug(
                    assigner=assigner,
                    prefix_feature_tensor=prefix_feature_tensor,
                    budget_tensor=budget_tensor,
                    execution_plan=execution_plan,
                )

        return {
            "status": "success",
            "phase": "plan",
            "encoded_payload": encoded_payload,
            "budget_value": budget_value_np,
            "execution_plan": execution_plan.detach().float().cpu().numpy(),
            "plan_meta": {
                "num_hidden_layers": self._num_hidden_layers,
                "num_attention_heads": self._num_attention_heads,
                "num_prefix_layers": self._num_prefix_layers,
                "scheduler_target": self._scheduler_target,
                "route_keys": route_keys,
                "llm_prefix_meta": payload.get("meta"),
                "llm_prefix_feature_stats": payload.get("budget_token_prefix_feature_stats"),
                "scheduler_budget_meta": payload.get("scheduler_meta"),
                "scheduler_plan_debug": plan_debug,
            },
        }

    def _run_decision_update_phase(self, assigner: BaseBudgetAssigner, payload: dict[str, Any]) -> dict[str, Any]:
        llm_payload = payload.get("llm_payload") if isinstance(payload.get("llm_payload"), dict) else payload
        if not isinstance(llm_payload, dict):
            raise ValueError("scheduler decision_update phase requires an 'llm_payload' object")

        speed_wps = llm_payload.get("speed_wps")
        route = llm_payload.get("route")
        if speed_wps is None:
            raise ValueError("llm_payload.speed_wps is required")
        if route is None:
            raise ValueError("llm_payload.route is required")

        speed_np = np.asarray(speed_wps, dtype=np.float32)
        route_np = np.asarray(route, dtype=np.float32)
        if speed_np.ndim < 3 or speed_np.shape[-1] != 2:
            raise ValueError(f"llm_payload.speed_wps must have shape [B, N, 2], got {speed_np.shape}")
        if route_np.ndim < 3 or route_np.shape[-1] != 2:
            raise ValueError(f"llm_payload.route must have shape [B, N, 2], got {route_np.shape}")
        batch_size = int(speed_np.shape[0])
        if route_np.shape[0] != batch_size:
            raise ValueError("llm_payload.route batch size must match speed_wps batch size")

        route_keys = self._resolve_route_keys(llm_payload, batch_size=batch_size)
        runtime_context = self._resolve_runtime_context(payload, llm_payload)
        ego_xy, ego_yaw, timestamp = self._extract_motion_context(runtime_context, batch_size)

        with self._model_lock:
            with torch.no_grad():
                speed_tensor = torch.as_tensor(speed_np, device=self._device, dtype=torch.float32)
                route_tensor = torch.as_tensor(route_np, device=self._device, dtype=torch.float32)
                driving_input = SimpleNamespace(
                    ego_xy=ego_xy,
                    ego_yaw=ego_yaw,
                    timestamp=timestamp,
                )
                metrics_computer = getattr(assigner, "metrics_computer", None)
                if metrics_computer is None:
                    decision_shift = {
                        "speed_wps": {"e_mean": [None] * batch_size, "e_norm": [None] * batch_size},
                        "route": {"e_mean": [None] * batch_size, "e_norm": [None] * batch_size},
                        "delta_tau": [None] * batch_size,
                        "t_lap": [float(assigner.decision_shift_t_lap)] * batch_size,
                    }
                else:
                    decision_shift_raw = metrics_computer.compute_decision_shift_metrics(
                        driving_input=driving_input,
                        current_speed_wps=speed_tensor,
                        current_route=route_tensor,
                        route_keys=route_keys,
                    )
                    decision_shift = metrics_computer.to_python_metrics(decision_shift_raw)

                update_rows = self._apply_decision_shift_to_budget_state(
                    assigner=assigner,
                    route_keys=route_keys,
                    decision_shift=decision_shift,
                )

        assigner.last_scene_metrics = {
            **(assigner.last_scene_metrics or {}),
            "decision_shift": decision_shift,
        }
        assigner.last_update_info = update_rows
        out = {
            "status": "success",
            "phase": "decision_update",
            "route_key": llm_payload.get("route_key"),
            "frame_id": llm_payload.get("frame_id"),
            "route_keys": route_keys,
            "decision_shift": decision_shift,
            "decision_update": update_rows,
            "eval_budget": assigner.get_eval_budget(),
        }
        return out

    def _build_plan_debug(
        self,
        *,
        assigner: BaseBudgetAssigner,
        prefix_feature_tensor: torch.Tensor,
        budget_tensor: torch.Tensor,
        execution_plan: torch.Tensor,
        max_values: int = 64,
    ) -> dict[str, Any]:
        scheduler = assigner.scheduler
        debug: dict[str, Any] = {
            "budget_value": budget_tensor.detach().float().cpu().tolist(),
            "prefix_feature_stats": self._tensor_stats(prefix_feature_tensor),
            "execution_plan_shape": list(execution_plan.shape),
        }

        layer_mask = (execution_plan.detach().float().sum(dim=(2, 3)) > 0).to(dtype=torch.int64)
        debug["layer_active_mask"] = layer_mask.cpu().tolist()
        debug["active_layer_count"] = layer_mask.sum(dim=1).cpu().tolist()

        if scheduler is None or not hasattr(scheduler, "mlp_head"):
            debug["logits_available"] = False
            return debug

        logits = scheduler.mlp_head(prefix_feature_tensor.contiguous()).detach().float().cpu()
        debug["logits_available"] = True
        debug["logits_shape"] = list(logits.shape)
        debug["logits"] = logits.tolist()
        debug["logits_stats"] = self._tensor_stats(logits)
        flat_logits = logits.reshape(-1)
        debug["logits_first_values"] = flat_logits[:max_values].tolist()

        num_prefix_layers = int(getattr(scheduler, "num_prefix_layers", self._num_prefix_layers or 0))
        num_hidden_layers = int(getattr(scheduler, "num_hidden_layers", self._num_hidden_layers or logits.size(-1)))
        sub_layer_count = max(num_hidden_layers - num_prefix_layers, 0)
        raw_units = torch.floor(budget_tensor.detach().float().cpu() * num_hidden_layers) - num_prefix_layers
        units = torch.clamp(raw_units, min=0, max=sub_layer_count).to(dtype=torch.long)
        debug["quantized_budget_units"] = units.tolist()
        debug["quantized_budget_raw_units"] = raw_units.tolist()
        debug["num_prefix_layers"] = num_prefix_layers
        debug["num_hidden_layers"] = num_hidden_layers

        if logits.ndim == 2 and logits.size(1) == sub_layer_count:
            topk_rows: list[dict[str, Any]] = []
            for batch_idx in range(logits.size(0)):
                k = int(units[batch_idx].item())
                row = logits[batch_idx]
                if k <= 0:
                    topk_rows.append(
                        {
                            "batch_index": batch_idx,
                            "k": k,
                            "sub_layer_indices": [],
                            "layer_indices": [],
                            "values": [],
                        }
                    )
                    continue
                values, indices = torch.topk(row, k=k, largest=True, sorted=True)
                topk_rows.append(
                    {
                        "batch_index": batch_idx,
                        "k": k,
                        "sub_layer_indices": indices.tolist(),
                        "layer_indices": (indices + num_prefix_layers).tolist(),
                        "values": values.tolist(),
                    }
                )
            debug["topk"] = topk_rows
        else:
            debug["topk"] = None
            debug["topk_note"] = "logits shape does not match SimpleScheduler_L sub-layer layout"
        return debug

    def _tensor_stats(self, value: torch.Tensor, *, max_items: int = 8) -> dict[str, Any]:
        tensor = value.detach().float().cpu()
        finite = torch.isfinite(tensor)
        flat = tensor.reshape(-1)
        finite_flat = flat[torch.isfinite(flat)]
        stats: dict[str, Any] = {
            "shape": list(tensor.shape),
            "finite": bool(finite.all().item()) if finite.numel() else True,
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "inf_count": int(torch.isinf(tensor).sum().item()),
            "first_values": flat[:max_items].tolist(),
        }
        if finite_flat.numel() == 0:
            stats.update({"mean": None, "std": None, "min": None, "max": None, "l2_norm": None, "abs_mean": None})
            return stats
        stats.update(
            {
                "mean": float(finite_flat.mean().item()),
                "std": float(finite_flat.std(unbiased=False).item()),
                "min": float(finite_flat.min().item()),
                "max": float(finite_flat.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(finite_flat).item()),
                "abs_mean": float(finite_flat.abs().mean().item()),
            }
        )
        if tensor.ndim >= 2:
            per_batch = tensor.reshape(tensor.shape[0], -1)
            stats["per_batch_l2_norm"] = torch.linalg.vector_norm(per_batch, dim=1).tolist()
            stats["per_batch_mean"] = per_batch.mean(dim=1).tolist()
            stats["per_batch_std"] = per_batch.std(dim=1, unbiased=False).tolist()
        return stats

    def _validate_budget_value(self, budget_value: np.ndarray, expected_batch: int) -> np.ndarray:
        if budget_value.ndim == 0:
            budget_value = np.repeat(budget_value.reshape(1), expected_batch)
        budget_value = budget_value.reshape(-1)
        if budget_value.shape[0] != expected_batch:
            raise ValueError(f"budget_value length ({budget_value.shape[0]}) must match batch size ({expected_batch})")
        return budget_value.astype(np.float32)

    def _resolve_runtime_context(self, payload: dict[str, Any], llm_payload: dict[str, Any]) -> dict[str, Any]:
        runtime_context = payload.get("runtime_context")
        if not isinstance(runtime_context, dict):
            runtime_context = llm_payload.get("runtime_context")
        if not isinstance(runtime_context, dict):
            plan_meta = llm_payload.get("scheduler_plan_meta")
            if isinstance(plan_meta, dict):
                runtime_context = plan_meta.get("runtime_context")
        if not isinstance(runtime_context, dict):
            raise ValueError("decision_update requires runtime_context with ego_xy, ego_yaw, and timestamp")
        return runtime_context

    def _extract_motion_context(
        self,
        runtime_context: dict[str, Any],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego_xy = runtime_context.get("ego_xy")
        ego_yaw = runtime_context.get("ego_yaw")
        timestamp = runtime_context.get("timestamp")
        if ego_xy is None or ego_yaw is None or timestamp is None:
            raise ValueError("runtime_context must contain ego_xy, ego_yaw, and timestamp")

        ego_xy_tensor = torch.as_tensor(np.asarray(ego_xy, dtype=np.float32), device=self._device, dtype=torch.float32)
        if ego_xy_tensor.ndim == 1:
            ego_xy_tensor = ego_xy_tensor.unsqueeze(0)
        if ego_xy_tensor.shape == (1, 2) and batch_size > 1:
            ego_xy_tensor = ego_xy_tensor.repeat(batch_size, 1)
        if ego_xy_tensor.shape != (batch_size, 2):
            raise ValueError(f"runtime_context.ego_xy must have shape [B, 2], got {tuple(ego_xy_tensor.shape)}")

        ego_yaw_tensor = torch.as_tensor(np.asarray(ego_yaw, dtype=np.float32), device=self._device, dtype=torch.float32).reshape(-1)
        timestamp_tensor = torch.as_tensor(np.asarray(timestamp, dtype=np.float32), device=self._device, dtype=torch.float32).reshape(-1)
        if ego_yaw_tensor.numel() == 1 and batch_size > 1:
            ego_yaw_tensor = ego_yaw_tensor.repeat(batch_size)
        if timestamp_tensor.numel() == 1 and batch_size > 1:
            timestamp_tensor = timestamp_tensor.repeat(batch_size)
        if ego_yaw_tensor.numel() != batch_size:
            raise ValueError(f"runtime_context.ego_yaw must have length {batch_size}, got {ego_yaw_tensor.numel()}")
        if timestamp_tensor.numel() != batch_size:
            raise ValueError(f"runtime_context.timestamp must have length {batch_size}, got {timestamp_tensor.numel()}")
        return ego_xy_tensor, ego_yaw_tensor, timestamp_tensor

    def _apply_decision_shift_to_budget_state(
        self,
        *,
        assigner: BaseBudgetAssigner,
        route_keys: list[str],
        decision_shift: dict[str, Any],
    ) -> list[dict[str, Any]]:
        speed_shift_list = decision_shift.get("speed_wps", {}).get("e_norm", [None] * len(route_keys))
        route_shift_list = decision_shift.get("route", {}).get("e_norm", [None] * len(route_keys))
        delta_tau_list = decision_shift.get("delta_tau", [None] * len(route_keys))

        update_rows: list[dict[str, Any]] = []
        for idx, route_key in enumerate(route_keys):
            state = assigner.state_by_route.get(route_key)
            if state is None:
                state = assigner._get_state(route_key) if hasattr(assigner, "_get_state") else None
            if state is None:
                raise RuntimeError(f"failed to initialize scheduler state for route {route_key}")

            speed_curr = speed_shift_list[idx] if idx < len(speed_shift_list) else None
            route_curr = route_shift_list[idx] if idx < len(route_shift_list) else None
            if speed_curr is not None and not np.isnan(speed_curr):
                state["prev_decision_shift_speed"] = float(speed_curr)
            if route_curr is not None and not np.isnan(route_curr):
                state["prev_decision_shift_route"] = float(route_curr)

            update_rows.append(
                {
                    "route_key": route_key,
                    "decision_shift_speed": speed_curr,
                    "decision_shift_route": route_curr,
                    "delta_tau": delta_tau_list[idx] if idx < len(delta_tau_list) else None,
                    "used_budget": float(state["used_budget"]),
                    "phase": str(state["phase"]),
                    "base_budget": state["base_budget"],
                    "instant_budget": state["instant_budget"],
                    "target_budget": state["target_budget"],
                    "frame_count": int(state["frame_count"]),
                    "safe_counter": int(state["safe_counter"]),
                }
            )
        return update_rows

    def _resolve_route_keys(self, encoded_payload: dict[str, Any], batch_size: int) -> list[str]:
        route_keys = encoded_payload.get("route_keys")
        if route_keys is not None:
            if not isinstance(route_keys, list) or len(route_keys) != batch_size:
                raise ValueError("encoded_payload.route_keys must be a list with length equal to batch size")
            return [str(key) for key in route_keys]

        route_key = encoded_payload.get("route_key")
        if route_key is not None:
            if batch_size == 1:
                return [str(route_key)]
            return [str(route_key) for _ in range(batch_size)]

        frame_id = encoded_payload.get("frame_id")
        if batch_size == 1 and frame_id is not None:
            return [f"frame::{frame_id}"]
        raise ValueError("encoded_payload must contain route_key or route_keys")

    def _summarize_visual_embeds(self, visual_embeds: np.ndarray, batch_size: int) -> torch.Tensor:
        visual_tensor = torch.as_tensor(visual_embeds, device=self._device, dtype=torch.float32)
        if visual_tensor.ndim < 2:
            raise ValueError(f"visual_embeds must have at least 2 dimensions, got {tuple(visual_tensor.shape)}")

        hidden_size = visual_tensor.size(-1)
        flat_tokens = visual_tensor.reshape(-1, hidden_size)
        if flat_tokens.size(0) % batch_size != 0:
            raise ValueError(
                f"visual_embeds token count ({flat_tokens.size(0)}) is not divisible by batch size ({batch_size})"
            )
        return flat_tokens.reshape(batch_size, -1, hidden_size).mean(dim=1)

    def _update_scene_state(
        self,
        assigner: BaseBudgetAssigner,
        route_keys: list[str],
        visual_summary: torch.Tensor,
    ) -> dict[str, Any]:
        novelty_list: list[float] = []
        sim_in_list: list[float] = []

        for idx, route_key in enumerate(route_keys):
            state = assigner.state_by_route.get(route_key)
            if state is None:
                state = assigner._get_state(route_key) if hasattr(assigner, "_get_state") else None
            if state is None:
                raise RuntimeError(f"failed to initialize scheduler state for route {route_key}")

            v_global = visual_summary[idx].detach().float()
            prev_hist = state.get("v_hist")
            if prev_hist is None:
                sim_in = 1.0
                updated_hist = v_global
            else:
                prev_hist = prev_hist.to(device=v_global.device, dtype=v_global.dtype)
                denom = float((v_global.norm(p=2) * prev_hist.norm(p=2)).item())
                cos = float(torch.dot(v_global, prev_hist).item() / max(denom, 1e-8))
                sim_in = float(torch.clamp(torch.tensor(0.5 * (1.0 + cos)), 0.0, 1.0).item())
                updated_hist = assigner.history_alpha * v_global + (1.0 - assigner.history_alpha) * prev_hist

            novelty = float(torch.clamp(torch.tensor(1.0 - sim_in), 0.0, 1.0).item())
            state["v_hist"] = updated_hist.detach()
            state["prev_novelty"] = novelty
            novelty_list.append(novelty)
            sim_in_list.append(sim_in)

        metrics = {
            "novelty": novelty_list,
            "history_similarity": {"sim_in": sim_in_list, "alpha": float(assigner.history_alpha)},
        }
        assigner.last_scene_metrics = metrics
        return metrics


scheduler_runtime = SchedulerRuntime()
