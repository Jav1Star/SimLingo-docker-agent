from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from simlingo_adaption_training.models.adaptors.adaptors import WaypointInputAdaptor
from simlingo_adaption_training.models.encoder.internvl2_model import LingoInternVLModel
from utils.logger_utils import get_logger


logger = get_logger(__name__)


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


class EncoderRuntime:
    """Owns only tokenizer, vision encoder, and lightweight input adaptors."""

    def __init__(self) -> None:
        self._model_variant: Optional[str] = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._image_encoder: LingoInternVLModel | None = None
        self._wp_encoder: WaypointInputAdaptor | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._num_image_token: Optional[int] = None
        self._hidden_size: Optional[int] = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._image_encoder is not None and self._tokenizer is not None

    @property
    def num_image_token(self) -> int:
        if self._num_image_token is None:
            raise RuntimeError("encoder has not been loaded yet")
        return self._num_image_token

    def load_model(self, model_variant: str, checkpoint_path: Optional[str] = None) -> None:
        if self.is_loaded and self._model_variant == model_variant:
            return
        with self._state_lock:
            if self.is_loaded and self._model_variant == model_variant:
                return

            logger.info("Loading SimLingo encoder from %s", model_variant)
            processor = AutoProcessor.from_pretrained(model_variant, trust_remote_code=True)
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            self._ensure_padding_token(tokenizer)
            tokenizer.add_special_tokens({"additional_special_tokens": DRIVING_SPECIAL_TOKENS})
            tokenizer.padding_side = "left"

            image_encoder = LingoInternVLModel(model_variant)
            image_encoder.processor = processor
            image_encoder.model.language_model = None
            image_encoder.model.to(device=self._device, dtype=self._dtype)
            image_encoder.model.eval()

            config = AutoConfig.from_pretrained(model_variant, trust_remote_code=True)
            image_size = config.force_image_size or config.vision_config.image_size
            patch_size = config.vision_config.patch_size
            num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
            hidden_size = int(config.llm_config.hidden_size)

            wp_encoder = WaypointInputAdaptor(token_size=hidden_size, hidden_size=256, hidden_size2=512)
            wp_encoder.to(device=self._device, dtype=self._dtype)
            wp_encoder.eval()

            if checkpoint_path:
                self._load_encoder_weights(
                    checkpoint_path=checkpoint_path,
                    image_encoder=image_encoder,
                    wp_encoder=wp_encoder,
                )

            self._model_variant = model_variant
            self._processor = processor
            self._tokenizer = tokenizer
            self._image_encoder = image_encoder
            self._wp_encoder = wp_encoder
            self._num_image_token = num_image_token
            self._hidden_size = hidden_size
            logger.info("Encoder loaded: hidden_size=%s num_image_token=%s", hidden_size, num_image_token)

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

    def _load_encoder_weights(
        self,
        *,
        checkpoint_path: str,
        image_encoder: LingoInternVLModel,
        wp_encoder: WaypointInputAdaptor,
    ) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ENCODER_CHECKPOINT_PATH does not exist: {path}")

        logger.info("Loading encoder-side weights from %s", path)
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint must be a state_dict or contain a state_dict key")

        image_prefix = "vision_model.image_encoder.model."
        image_state = {
            key[len(image_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(image_prefix)
        }
        if image_state:
            missing, unexpected = image_encoder.model.load_state_dict(image_state, strict=False)
            logger.info(
                "Loaded image encoder partial weights: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No image encoder weights found with prefix %s", image_prefix)

        wp_prefix = "wp_encoder."
        wp_state = {
            key[len(wp_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(wp_prefix)
        }
        if wp_state:
            missing, unexpected = wp_encoder.load_state_dict(wp_state, strict=False)
            logger.info(
                "Loaded waypoint encoder partial weights: missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("No waypoint encoder weights found with prefix %s", wp_prefix)

    def _require_loaded(self) -> tuple[Any, LingoInternVLModel, WaypointInputAdaptor]:
        if self._tokenizer is None or self._image_encoder is None or self._wp_encoder is None:
            raise RuntimeError("encoder model has not been loaded yet")
        return self._tokenizer, self._image_encoder, self._wp_encoder

    def tokenize(
        self,
        prompt_texts: list[str],
        *,
        add_special_tokens: bool = False,
        expand_image_token: bool = True,
        num_patches: int = 2,
    ) -> dict[str, np.ndarray | list[str]]:
        tokenizer, _, _ = self._require_loaded()
        prompts = [
            self._expand_image_placeholder(text, num_patches=num_patches) if expand_image_token else text
            for text in prompt_texts
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        input_ids = encoded["input_ids"].cpu().numpy().astype(np.int64)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("Tokenizer padding token is not configured after initialization.")
        attention_mask = (encoded["input_ids"] != pad_token_id).cpu().numpy().astype(np.bool_)
        img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        visual_positions = [np.where(row == img_context_id)[0].astype(np.int64) for row in input_ids]
        return {
            "prompt_texts": prompts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "visual_token_positions": visual_positions,
        }

    def encode_visual(self, camera_images: np.ndarray) -> np.ndarray:
        _, image_encoder, _ = self._require_loaded()
        pixel_values = torch.as_tensor(camera_images, device=self._device)
        if not torch.is_floating_point(pixel_values):
            pixel_values = pixel_values.float()
        pixel_values = pixel_values.to(dtype=self._dtype)

        if pixel_values.ndim == 6:
            batch_size, time_steps, num_patches, channels, height, width = pixel_values.shape
            if time_steps != 1:
                raise ValueError("encoder_agent currently supports T=1 camera input")
            pixel_values = pixel_values.reshape(batch_size * num_patches, channels, height, width)
        elif pixel_values.ndim != 4:
            raise ValueError("camera_images must have shape [B,T,NP,C,H,W] or [B*NP,C,H,W]")

        with self._model_lock:
            with torch.no_grad():
                visual_embeds = image_encoder.model.extract_feature(pixel_values)
        return visual_embeds.detach().float().cpu().numpy()

    def encode_waypoints(self, placeholder_values: Any) -> dict[str, Any]:
        _, _, wp_encoder = self._require_loaded()
        if placeholder_values is None:
            return {}

        batches = self._normalize_placeholder_batches(placeholder_values)
        encoded: dict[str, Any] = {}
        for batch_key, item in batches:
            encoded[batch_key] = {}
            for token_id, coords in item.items():
                try:
                    coords_tensor = torch.as_tensor(coords, device=self._device, dtype=self._dtype)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid waypoint coordinates for batch {batch_key}, token {token_id}: {exc}"
                    ) from exc
                if coords_tensor.ndim != 2 or coords_tensor.size(-1) != 2:
                    raise ValueError(
                        f"Waypoint coordinates for batch {batch_key}, token {token_id} "
                        f"must have shape [N, 2], got {tuple(coords_tensor.shape)}"
                    )
                with torch.no_grad():
                    embeds = wp_encoder(coords_tensor.unsqueeze(0)).squeeze(0)
                encoded[batch_key][str(token_id)] = embeds.detach().float().cpu().numpy()
        return encoded

    def _normalize_placeholder_batches(self, placeholder_values: Any) -> list[tuple[str, dict[Any, Any]]]:
        if isinstance(placeholder_values, dict):
            if not placeholder_values:
                return []
            if all(hasattr(value, "items") for value in placeholder_values.values()):
                return [(str(batch_key), value) for batch_key, value in placeholder_values.items()]
            if all(not hasattr(value, "items") for value in placeholder_values.values()):
                return [("0", placeholder_values)]
            raise ValueError(
                "placeholder_values dict has mixed value types; expected either "
                "{token_id: coords} or {batch_id: {token_id: coords}}."
            )

        if isinstance(placeholder_values, (list, tuple)):
            batches: list[tuple[str, dict[Any, Any]]] = []
            for batch_idx, item in enumerate(placeholder_values):
                if item is None:
                    batches.append((str(batch_idx), {}))
                    continue
                if not hasattr(item, "items"):
                    raise ValueError(
                        f"placeholder_values[{batch_idx}] must be a mapping of token_id -> coords, "
                        f"got {type(item).__name__}."
                    )
                batches.append((str(batch_idx), item))
            return batches

        raise ValueError(
            "placeholder_values must be a list/tuple of mappings, a mapping of token_id -> coords, "
            f"or a mapping of batch_id -> mapping. Got {type(placeholder_values).__name__}."
        )

    def encode(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt_texts = payload.get("prompt_texts")
        tokenized = {}
        if prompt_texts is not None:
            tokenized = self.tokenize(
                list(prompt_texts),
                add_special_tokens=bool(payload.get("add_special_tokens", False)),
                expand_image_token=bool(payload.get("expand_image_token", True)),
                num_patches=int(payload.get("num_patches", 2)),
            )
        elif "input_ids" in payload:
            tokenized = {
                "input_ids": np.asarray(payload["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(payload.get("attention_mask"), dtype=np.bool_),
                "prompt_texts": payload.get("prompt_texts", []),
                "visual_token_positions": payload.get("visual_token_positions", []),
            }

        visual_embeds = None
        if payload.get("camera_images") is not None:
            visual_embeds = self.encode_visual(np.asarray(payload["camera_images"]))

        waypoint_embeds = self.encode_waypoints(payload.get("placeholder_values"))

        return {
            "agent": "simlingo-encoder-agent",
            "route_key": payload.get("route_key"),
            "frame_id": payload.get("frame_id"),
            "tokenized": tokenized,
            "visual_embeds": visual_embeds,
            "waypoint_embeds": waypoint_embeds,
            "runtime_context": payload.get("runtime_context", {}),
            "meta": {
                "model_variant": self._model_variant,
                "hidden_size": self._hidden_size,
                "num_image_token": self._num_image_token,
                "device": self._device,
                "dtype": str(self._dtype),
            },
        }

    def _expand_image_placeholder(self, text: str, *, num_patches: int) -> str:
        if "<image>" not in text:
            text = "<image>\n" + text
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * num_patches + "</img>"
        return text.replace("<image>", image_tokens, 1)


def save_artifacts(payload: dict[str, Any], artifact_dir: str, frame_id: str) -> dict[str, Any]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame_dir = root / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    def _save(value: Any, name: str) -> Any:
        if isinstance(value, np.ndarray):
            path = frame_dir / f"{name}.npy"
            np.save(path, value)
            return {
                "uri": f"file://{path}",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {str(key): _save(item, f"{name}_{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [_save(item, f"{name}_{idx}") for idx, item in enumerate(value)]
        return value

    return _save(payload, "payload")


encoder_runtime = EncoderRuntime()
