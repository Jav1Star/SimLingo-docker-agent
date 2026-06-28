from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoConfig

from simlingo_adaption_training.models.adaptors.adaptors import DrivingAdaptor
from simlingo_adaption_training.models.language_model.llm import LLM
from utils.logger_utils import get_logger


logger = get_logger(__name__)


LLM_PREFIX_NUM_LAYERS = 2
SCHEDULER_NUM_PREFIX_LAYERS = 10

DRIVING_SPECIAL_TOKENS = [
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    "<TARGET_POINT>",
]


class LLMRuntime:
    """Owns the two-stage language executor and driving output heads."""

    def __init__(self) -> None:
        self._model_variant: Optional[str] = None
        self._tokenizer: Any | None = None
        self._llm: LLM | None = None
        self._driving_adaptor: DrivingAdaptor | None = None
        self._budget_encoder: torch.nn.Module | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._hidden_size: Optional[int] = None
        self._num_hidden_layers: Optional[int] = None
        self._num_attention_heads: Optional[int] = None
        self._num_prefix_layers: Optional[int] = None
        self._scheduler_num_prefix_layers: Optional[int] = None
        self._img_context_token_id: Optional[int] = None
        self._model_param_dtype: Optional[torch.dtype] = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return (
            self._llm is not None
            and self._driving_adaptor is not None
            and self._tokenizer is not None
            and self._budget_encoder is not None
        )

    def load_model(
        self,
        model_variant: str,
        *,
        checkpoint_path: Optional[str] = None,
        speed_wps_mode: str = "2d",
        predict_route_as_wps: bool = True,
        use_lora: bool = True,
        adaption_train: bool = False,
        lora_alpha: int = 64,
        lora_r: int = 32,
        lora_dropout: float = 0.1,
        num_prefix_layers: int = LLM_PREFIX_NUM_LAYERS,
        scheduler_num_prefix_layers: int = SCHEDULER_NUM_PREFIX_LAYERS,
        scheduler_target: str,
        scheduler_tau: float = 5.0,
        scheduler_is_hard: bool = True,
        scheduler_threshold: float = 0.5,
        scheduler_bias: bool = True,
    ) -> None:
        num_prefix_layers = int(num_prefix_layers)
        scheduler_num_prefix_layers = int(scheduler_num_prefix_layers)

        if self.is_loaded and self._model_variant == model_variant:
            return

        with self._state_lock:
            if self.is_loaded and self._model_variant == model_variant:
                return

            logger.info("Loading SimLingo two-stage LLM from %s", model_variant)
            llm = LLM(
                variant=model_variant,
                lora=use_lora,
                lora_alpha=lora_alpha,
                lora_r=lora_r,
                lora_dropout=lora_dropout,
                adaption_train=bool(adaption_train),
                num_prefix_layers=num_prefix_layers,
                cache_dir=None,
            )
            llm.get_lora_model()
            llm.model.to(device=self._device, dtype=self._dtype)
            llm.model.eval()
            llm.eval()

            tokenizer = llm.tokenizer
            self._ensure_padding_token(tokenizer)
            tokenizer.add_special_tokens({"additional_special_tokens": DRIVING_SPECIAL_TOKENS})
            tokenizer.padding_side = "left"

            llm.config.num_prefix_layers = num_prefix_layers
            llm.model.config.num_prefix_layers = num_prefix_layers

            driving_adaptor = DrivingAdaptor(
                llm.hidden_size,
                speed_wps_mode=speed_wps_mode,
                predict_route_as_wps=predict_route_as_wps,
            )
            driving_adaptor.to(device=self._device, dtype=self._dtype)
            driving_adaptor.eval()

            budget_encoder = self._build_budget_encoder(
                model_variant=model_variant,
                scheduler_target=scheduler_target,
                tau=scheduler_tau,
                is_hard=scheduler_is_hard,
                threshold=scheduler_threshold,
                bias=scheduler_bias,
                num_prefix_layers=scheduler_num_prefix_layers,
            )
            budget_encoder.to(device=self._device, dtype=torch.float32)
            budget_encoder.eval()

            if checkpoint_path:
                self._load_llm_weights(
                    checkpoint_path=checkpoint_path,
                    llm=llm,
                    driving_adaptor=driving_adaptor,
                    budget_encoder=budget_encoder,
                )

            self._model_variant = model_variant
            self._tokenizer = tokenizer
            self._llm = llm
            self._driving_adaptor = driving_adaptor
            self._budget_encoder = budget_encoder
            self._hidden_size = int(llm.hidden_size)
            self._num_hidden_layers = int(llm.config.num_hidden_layers)
            self._num_attention_heads = int(llm.config.num_attention_heads)
            self._num_prefix_layers = int(num_prefix_layers)
            self._scheduler_num_prefix_layers = int(scheduler_num_prefix_layers)
            self._img_context_token_id = int(tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))
            self._model_param_dtype = next(llm.model.parameters()).dtype
            logger.info(
                (
                    "Two-stage LLM loaded: hidden_size=%s num_hidden_layers=%s "
                    "num_attention_heads=%s llm_num_prefix_layers=%s scheduler_num_prefix_layers=%s"
                ),
                self._hidden_size,
                self._num_hidden_layers,
                self._num_attention_heads,
                self._num_prefix_layers,
                self._scheduler_num_prefix_layers,
            )

    def _ensure_padding_token(self, tokenizer: Any) -> None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            return

        if getattr(tokenizer, "eos_token_id", None) is not None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer pad_token was missing; using eos_token as padding token.")
            return

        if getattr(tokenizer, "unk_token_id", None) is not None and getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info("Tokenizer pad_token was missing; using unk_token as padding token.")
            return

        raise ValueError(
            "Tokenizer does not define pad_token_id and has no eos_token/unk_token fallback. "
            "Please provide a tokenizer with a valid padding token."
        )

    def _build_budget_encoder(
        self,
        *,
        model_variant: str,
        scheduler_target: str,
        tau: float,
        is_hard: bool,
        threshold: float,
        bias: bool,
        num_prefix_layers: int,
    ) -> torch.nn.Module:
        llm_config = AutoConfig.from_pretrained(model_variant, trust_remote_code=True)
        llm_cfg = getattr(llm_config, "llm_config", llm_config)
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
        scheduler_module = hydra.utils.instantiate(
            scheduler_cfg,
            num_hidden_layers=language_cfg.num_hidden_layers,
            num_attention_heads=language_cfg.num_attention_heads,
            hidden_size=language_cfg.hidden_size,
            _recursive_=False,
        )
        return scheduler_module

    def _load_llm_weights(
        self,
        *,
        checkpoint_path: str,
        llm: LLM,
        driving_adaptor: DrivingAdaptor,
        budget_encoder: torch.nn.Module,
    ) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LLM_CHECKPOINT_PATH does not exist: {path}")

        logger.info("Loading language-side weights from %s", path)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint must be a state_dict or contain a state_dict key")

        language_prefix = "language_model."
        language_state = {
            key[len(language_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(language_prefix)
        }
        if language_state:
            missing, unexpected = llm.load_state_dict(language_state, strict=False)
            logger.info(
                "Loaded language model partial weights: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No language model weights found with prefix %s", language_prefix)

        driving_prefix = "adaptors.driving."
        driving_state = {
            key[len(driving_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(driving_prefix)
        }
        if driving_state:
            missing, unexpected = driving_adaptor.load_state_dict(driving_state, strict=False)
            logger.info(
                "Loaded driving adaptor partial weights: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No driving adaptor weights found with prefix %s", driving_prefix)

        scheduler_prefixes = ("budget_assigner.scheduler.", "scheduler.")
        scheduler_state: dict[str, Any] = {}
        for key, value in state_dict.items():
            for prefix in scheduler_prefixes:
                if key.startswith(prefix):
                    scheduler_state.setdefault(key[len(prefix):], value)
                    break
        if scheduler_state:
            scheduler_state = self._filter_compatible_state_dict(
                scheduler_state,
                budget_encoder,
                module_name="budget encoder",
            )
        if scheduler_state:
            missing, unexpected = budget_encoder.load_state_dict(scheduler_state, strict=False)
            logger.info(
                "Loaded budget encoder partial weights: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No compatible scheduler weights found with prefixes %s", scheduler_prefixes)

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

    def _require_loaded(self) -> tuple[LLM, DrivingAdaptor, Any, torch.nn.Module]:
        if self._llm is None or self._driving_adaptor is None or self._tokenizer is None or self._budget_encoder is None:
            raise RuntimeError("LLM model has not been loaded yet")
        return self._llm, self._driving_adaptor, self._tokenizer, self._budget_encoder

    def run(self, payload: dict[str, Any], phase: str = "final") -> dict[str, Any]:
        llm, driving_adaptor, _tokenizer, budget_encoder = self._require_loaded()
        phase = str(phase).strip().lower()
        if phase not in {"prefix", "final"}:
            raise ValueError(f"Unsupported llm phase: {phase}")

        encoded_payload = payload.get("encoded_payload")
        if not isinstance(encoded_payload, dict):
            raise ValueError("payload must contain an 'encoded_payload' object")

        tokenized = encoded_payload.get("tokenized")
        if not isinstance(tokenized, dict):
            raise ValueError("encoded_payload must contain a 'tokenized' object")

        input_ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
        visual_embeds = encoded_payload.get("visual_embeds")
        budget_value = payload.get("budget_value")
        execution_plan = payload.get("execution_plan")
        if input_ids is None:
            raise ValueError("encoded_payload.tokenized.input_ids is required")
        if attention_mask is None:
            raise ValueError("encoded_payload.tokenized.attention_mask is required")
        if visual_embeds is None:
            raise ValueError("encoded_payload.visual_embeds is required")
        if budget_value is None:
            raise ValueError("payload.budget_value is required")
        if phase == "final" and execution_plan is None:
            raise ValueError("payload.execution_plan is required for llm final phase")

        input_ids_np = np.asarray(input_ids, dtype=np.int64)
        attention_mask_np = np.asarray(attention_mask, dtype=np.bool_)
        visual_embeds_np = np.asarray(visual_embeds)
        budget_value_np = np.asarray(budget_value, dtype=np.float32)
        execution_plan_np = None if execution_plan is None else np.asarray(execution_plan)

        if input_ids_np.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, S], got {input_ids_np.shape}")
        if attention_mask_np.shape != input_ids_np.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids shape, got {attention_mask_np.shape} vs {input_ids_np.shape}"
            )
        budget_value_np = self._validate_budget_value(budget_value_np, batch_size=input_ids_np.shape[0])
        if execution_plan_np is not None:
            self._validate_execution_plan(execution_plan_np, batch_size=input_ids_np.shape[0])

        input_ids_tensor = torch.as_tensor(input_ids_np, device=self._device, dtype=torch.long)
        attention_mask_tensor = torch.as_tensor(attention_mask_np, device=self._device, dtype=torch.bool)
        budget_value_tensor = torch.as_tensor(budget_value_np, device=self._device, dtype=torch.float32)
        execution_plan_tensor = None if execution_plan_np is None else torch.as_tensor(execution_plan_np, device=self._device)

        with self._model_lock:
            with torch.no_grad():
                language_inputs = self._embed_input_ids(llm, input_ids_tensor)
                language_inputs = self._replace_waypoint_tokens(
                    language_inputs=language_inputs,
                    input_ids=input_ids_tensor,
                    waypoint_embeds=encoded_payload.get("waypoint_embeds"),
                )
                language_inputs = self._replace_visual_tokens(
                    language_inputs=language_inputs,
                    input_ids=input_ids_tensor,
                    visual_embeds=visual_embeds_np,
                )

                budget_inputs = budget_encoder.budget_encoding(budget_value_tensor).to(device=self._device, dtype=language_inputs.dtype)
                if budget_inputs.ndim == 2:
                    budget_inputs = budget_inputs.unsqueeze(1)
                budget_mask = torch.ones((input_ids_tensor.size(0), 1), dtype=torch.bool, device=self._device)

                driving_inputs, driving_mask = self._build_driving_inputs(driving_adaptor, input_ids_tensor.size(0))
                inputs_embeds, inputs_mask, perm, split_sizes, budget_position = self._merge_inputs(
                    language_inputs=language_inputs,
                    language_mask=attention_mask_tensor,
                    budget_inputs=budget_inputs,
                    budget_mask=budget_mask,
                    driving_inputs=driving_inputs,
                    driving_mask=driving_mask,
                )

                model_kwargs: dict[str, Any] = {
                    "inputs_embeds": inputs_embeds.to(dtype=self._model_param_dtype or inputs_embeds.dtype),
                    "attention_mask": inputs_mask,
                    "output_hidden_states": True,
                    "return_dict": True,
                    "execution_plan": execution_plan_tensor,
                }
                if phase == "prefix":
                    model_kwargs.update(
                        {
                            "stop_at_layer": self._num_prefix_layers,
                            "skip_logits": True,
                            "use_cache": False,
                        }
                    )

                outputs = llm.model(**model_kwargs)

        route_key = encoded_payload.get("route_key")
        frame_id = encoded_payload.get("frame_id")
        runtime_context = encoded_payload.get("runtime_context", {})

        if phase == "prefix":
            prefix_hidden_states = outputs.hidden_states[self._num_prefix_layers]
            batch_indices = torch.arange(prefix_hidden_states.size(0), device=prefix_hidden_states.device)
            budget_token_prefix_feature = prefix_hidden_states[batch_indices, budget_position]
            return {
                "status": "success",
                "phase": "prefix",
                "route_key": route_key,
                "frame_id": frame_id,
                "encoded_payload": encoded_payload,
                "budget_value": budget_value_np,
                "budget_token_prefix_feature": budget_token_prefix_feature.detach().float().cpu().numpy(),
                "runtime_context": runtime_context,
                "meta": {
                    "model_variant": self._model_variant,
                    "hidden_size": self._hidden_size,
                    "num_hidden_layers": self._num_hidden_layers,
                    "num_attention_heads": self._num_attention_heads,
                    "num_prefix_layers": self._num_prefix_layers,
                    "scheduler_num_prefix_layers": self._scheduler_num_prefix_layers,
                    "device": self._device,
                    "dtype": str(self._dtype),
                },
            }

        features = outputs.hidden_states[-1]
        adaptor_features = self._split_outputs(
            outputs=features,
            perm=perm,
            split_sizes=split_sizes,
        )
        driving_features = adaptor_features["driving"]
        predictions = driving_adaptor.get_predictions(driving_features)

        speed_wps = predictions.get("speed_wps")
        route = predictions.get("route")

        return {
            "status": "success",
            "phase": "final",
            "route_key": route_key,
            "frame_id": frame_id,
            "llm_payload": {
                "route_key": route_key,
                "frame_id": frame_id,
                "speed_wps": None if speed_wps is None else speed_wps.detach().float().cpu().numpy(),
                "route": None if route is None else route.detach().float().cpu().numpy(),
                "driving_features": driving_features.detach().float().cpu().numpy(),
                "execution_plan_applied": execution_plan_np,
                "budget_value": budget_value_np,
                "runtime_context": runtime_context,
                "meta": {
                    "model_variant": self._model_variant,
                    "hidden_size": self._hidden_size,
                    "num_hidden_layers": self._num_hidden_layers,
                    "num_attention_heads": self._num_attention_heads,
                    "num_prefix_layers": self._num_prefix_layers,
                    "scheduler_num_prefix_layers": self._scheduler_num_prefix_layers,
                    "device": self._device,
                    "dtype": str(self._dtype),
                },
            },
        }

    def _validate_budget_value(self, budget_value: np.ndarray, *, batch_size: int) -> np.ndarray:
        if budget_value.ndim == 0:
            budget_value = np.repeat(budget_value.reshape(1), batch_size)
        budget_value = budget_value.reshape(-1)
        if budget_value.shape[0] != batch_size:
            raise ValueError(f"budget_value length ({budget_value.shape[0]}) must match batch size ({batch_size})")
        return budget_value.astype(np.float32)

    def _validate_execution_plan(self, execution_plan: np.ndarray, *, batch_size: int) -> None:
        if execution_plan.ndim != 4:
            raise ValueError(
                "execution_plan must have shape [B, num_hidden_layers, 2, num_attention_heads], "
                f"got {execution_plan.shape}"
            )
        if execution_plan.shape[0] != batch_size:
            raise ValueError(
                f"execution_plan batch size ({execution_plan.shape[0]}) must match input batch size ({batch_size})"
            )
        if self._num_hidden_layers is not None and execution_plan.shape[1] != self._num_hidden_layers:
            raise ValueError(
                f"execution_plan layer count ({execution_plan.shape[1]}) must match model num_hidden_layers "
                f"({self._num_hidden_layers})"
            )
        if execution_plan.shape[2] != 2:
            raise ValueError(f"execution_plan third dimension must be 2, got {execution_plan.shape[2]}")
        if self._num_attention_heads is not None and execution_plan.shape[3] != self._num_attention_heads:
            raise ValueError(
                f"execution_plan head count ({execution_plan.shape[3]}) must match model num_attention_heads "
                f"({self._num_attention_heads})"
            )

    def _embed_input_ids(self, llm: LLM, input_ids: torch.Tensor) -> torch.Tensor:
        embed_tokens = llm.model.get_input_embeddings()
        safe_input_ids = input_ids.clamp(min=0, max=embed_tokens.num_embeddings - 1)
        return embed_tokens(safe_input_ids)

    def _replace_visual_tokens(
        self,
        *,
        language_inputs: torch.Tensor,
        input_ids: torch.Tensor,
        visual_embeds: np.ndarray,
    ) -> torch.Tensor:
        if self._img_context_token_id is None:
            raise RuntimeError("IMG_CONTEXT token id has not been initialized")

        vit_embeds = torch.as_tensor(visual_embeds, device=language_inputs.device, dtype=language_inputs.dtype)
        if vit_embeds.ndim < 2:
            raise ValueError(f"visual_embeds must have at least 2 dimensions, got {tuple(vit_embeds.shape)}")

        hidden_size = language_inputs.size(-1)
        vit_embeds = vit_embeds.reshape(-1, hidden_size)

        flat_inputs = language_inputs.reshape(-1, hidden_size)
        flat_ids = input_ids.reshape(-1)
        selected = flat_ids == self._img_context_token_id
        num_required = int(selected.sum().item())
        if vit_embeds.size(0) < num_required:
            raise ValueError(
                f"Not enough visual tokens for <IMG_CONTEXT>: required={num_required}, got={vit_embeds.size(0)}"
            )
        if vit_embeds.size(0) > num_required:
            logger.warning(
                "Vision token count mismatch for <IMG_CONTEXT>: required=%s got=%s; truncating extra tokens.",
                num_required,
                vit_embeds.size(0),
            )
        flat_inputs[selected] = vit_embeds[:num_required]
        return flat_inputs.reshape_as(language_inputs)

    def _replace_waypoint_tokens(
        self,
        *,
        language_inputs: torch.Tensor,
        input_ids: torch.Tensor,
        waypoint_embeds: Any,
    ) -> torch.Tensor:
        if not waypoint_embeds:
            return language_inputs
        if not isinstance(waypoint_embeds, dict):
            raise ValueError("encoded_payload.waypoint_embeds must be a mapping of batch_id -> token_id -> embeds")

        seq_len = input_ids.size(1)
        for batch_key, token_map in waypoint_embeds.items():
            if not isinstance(token_map, dict):
                raise ValueError("encoded_payload.waypoint_embeds values must be mappings of token_id -> embeds")
            batch_idx = int(batch_key)
            if batch_idx < 0 or batch_idx >= input_ids.size(0):
                raise ValueError(f"waypoint batch index out of range: {batch_idx}")
            for token_key, embeds in token_map.items():
                coords_embeds = torch.as_tensor(embeds, device=language_inputs.device, dtype=language_inputs.dtype)
                if coords_embeds.ndim != 2 or coords_embeds.size(-1) != language_inputs.size(-1):
                    raise ValueError(
                        f"waypoint embeds for batch {batch_idx}, token {token_key} must have shape [N, {language_inputs.size(-1)}], "
                        f"got {tuple(coords_embeds.shape)}"
                    )
                token_id = int(token_key)
                positions = (input_ids[batch_idx] == token_id).nonzero(as_tuple=False).flatten()
                if positions.numel() == 0:
                    raise ValueError(
                        f"waypoint token id {token_id} was not found in input_ids for batch {batch_idx}"
                    )
                start = int(positions[0].item())
                end = start + coords_embeds.size(0)
                if end > seq_len:
                    raise ValueError(
                        f"waypoint embeds for batch {batch_idx}, token {token_key} exceed sequence length: "
                        f"start={start}, len={coords_embeds.size(0)}, seq_len={seq_len}"
                    )
                language_inputs[batch_idx, start:end] = coords_embeds
        return language_inputs

    def _build_driving_inputs(
        self,
        driving_adaptor: DrivingAdaptor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = None
        for input_type in driving_adaptor.order:
            query_embed = driving_adaptor.queries[input_type]
            expanded = query_embed.expand(batch_size, -1, -1)
            inputs = expanded if inputs is None else torch.cat((inputs, expanded), dim=1)
        if inputs is None:
            raise RuntimeError("Driving adaptor produced no query tokens")
        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool, device=inputs.device)
        return inputs, inputs_mask

    def _merge_inputs(
        self,
        *,
        language_inputs: torch.Tensor,
        language_mask: torch.Tensor,
        budget_inputs: torch.Tensor,
        budget_mask: torch.Tensor,
        driving_inputs: torch.Tensor,
        driving_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat((language_inputs, budget_inputs, driving_inputs), dim=1)
        inputs_mask = torch.cat((language_mask, budget_mask, driving_mask), dim=1)
        split_sizes = torch.as_tensor(
            [language_inputs.size(1), budget_inputs.size(1), driving_inputs.size(1)],
            device=inputs.device,
            dtype=torch.long,
        )
        budget_orig_index = torch.as_tensor(language_inputs.size(1), device=inputs.device, dtype=torch.long)
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None]
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1)
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        perm = rand_perm.gather(1, valid_perm)
        inv_perm = perm.argsort(-1)
        budget_position = inv_perm[:, budget_orig_index].long()
        return inputs[arange, perm], inputs_mask[arange, perm], perm, split_sizes, budget_position

    def _split_outputs(
        self,
        *,
        outputs: torch.Tensor,
        perm: torch.Tensor,
        split_sizes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inv_perm = perm.argsort(-1)
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]
        outputs = outputs[arange, inv_perm]
        language_size, budget_size, driving_size = [int(x) for x in split_sizes.tolist()]
        language_features, budget_features, driving_features = outputs.split(
            [language_size, budget_size, driving_size],
            dim=1,
        )
        return {
            "language": language_features,
            "budget": budget_features,
            "driving": driving_features,
        }


llm_runtime = LLMRuntime()
