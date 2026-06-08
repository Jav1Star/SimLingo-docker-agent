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
        num_prefix_layers: int = 2,
        rule_based_cfg_json: Optional[str] = None,
    ) -> None:
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

        missing, unexpected = scheduler_module.load_state_dict(scheduler_state, strict=False)
        logger.info(
            "Loaded scheduler partial weights: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

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
            },
        }

    def _validate_budget_value(self, budget_value: np.ndarray, expected_batch: int) -> np.ndarray:
        if budget_value.ndim == 0:
            budget_value = np.repeat(budget_value.reshape(1), expected_batch)
        budget_value = budget_value.reshape(-1)
        if budget_value.shape[0] != expected_batch:
            raise ValueError(f"budget_value length ({budget_value.shape[0]}) must match batch size ({expected_batch})")
        return budget_value.astype(np.float32)

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
