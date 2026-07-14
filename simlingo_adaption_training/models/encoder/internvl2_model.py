from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoModel

from simlingo_adaption_training.models.adaptors.adaptors import rebuild_packed_adaptor_inputs
from simlingo_adaption_training.models.encoder.visual_token_pruner import VisualTokenPruner


class LingoInternVLModel(nn.Module):
    def __init__(self, variant, token_prune=None, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(variant, trust_remote_code=True)
        try:
            self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
        except Exception:
            self.num_embeddings = self.model.language_model.vocab_size
        self.use_global_img = None
        self.processor = None
        self.visual_token_pruner = self._build_visual_token_pruner(token_prune)

    def _build_visual_token_pruner(self, token_prune) -> Optional[VisualTokenPruner]:
        """关键调用点：只在显式开启时创建 prune 模块，默认保持原始路径。"""
        if token_prune is None:
            return None

        mode = str(getattr(token_prune, "mode", "off")).strip().lower()
        if mode == "off":
            return None

        return VisualTokenPruner(
            mode=mode,
            prune_ratio=float(getattr(token_prune, "prune_ratio", 0.0)),
            min_keep=int(getattr(token_prune, "min_keep", 1)),
        )

    def replace_placeholder_tokens(
        self,
        adaptor_dict: Dict[str, torch.Tensor],
        pixel_values: Optional[torch.FloatTensor],
        placeholder_values: Optional[List[dict]],
        wp_encoder: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """关键调用点：视觉与 waypoint 占位替换统一收敛到 image encoder。"""
        if self.processor is None:
            raise ValueError("processor must be set before replacing placeholder tokens")

        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        inputs_embeds = adaptor_dict["language_inputs"]
        input_ids = adaptor_dict["language__ids"]
        inputs_embeds_dtype = inputs_embeds.dtype
        inputs_embeds_work = inputs_embeds.to(dtype=torch.float32)

        # 先替换 waypoint 相关占位，避免后续视觉写回覆盖文本段。
        smallest_added_id = self.tokenizer.additional_special_tokens_ids[0]
        special_ids = torch.tensor(
            list(set(input_ids[(input_ids >= smallest_added_id)].tolist())),
            device=input_ids.device,
        )
        special_ids = special_ids[special_ids != self.img_context_token_id]
        special_ids = special_ids.view(-1, 1, 1)

        if special_ids.size(0) > 0 and placeholder_values:
            wp_encoder_dtype = wp_encoder.mlp[0].weight.dtype
            replacement_specs = []
            for batch_idx in range(input_ids.size(0)):
                for special_id in special_ids.view(-1).tolist():
                    positions = torch.nonzero(
                        input_ids[batch_idx] == int(special_id),
                        as_tuple=False,
                    ).squeeze(-1)
                    if positions.numel() == 0:
                        continue
                    replacement_specs.append(
                        (
                            batch_idx,
                            int(positions[0].item()),
                            torch.tensor(
                                placeholder_values[batch_idx][int(special_id)],
                                device=input_ids.device,
                                dtype=wp_encoder_dtype,
                            ),
                        )
                    )

            if replacement_specs:
                coords_length_org = [len(spec[2]) for spec in replacement_specs]
                coords = torch.cat([spec[2] for spec in replacement_specs], dim=0)
                wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)
                wp_embeds = torch.split(wp_embeds, coords_length_org)

                with torch.autocast(device_type=inputs_embeds_work.device.type, enabled=False):
                    for (batch_idx, start, _), wp_embed in zip(replacement_specs, wp_embeds):
                        end = start + wp_embed.size(0)
                        inputs_embeds_work[batch_idx, start:end] = wp_embed.to(dtype=inputs_embeds_work.dtype)

        visual_keep_mask = None
        visual_keep_indices = None
        visual_scores = None

        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
            _, _, embed_dim = inputs_embeds_work.shape
            batch_size, time_steps, num_patches, channels, height, width = pixel_values.shape
            if time_steps != 1:
                raise ValueError("Only one frame is supported for now")

            pixel_values = pixel_values.view(batch_size, num_patches, channels, height, width)
            pixel_values = pixel_values.reshape(batch_size * num_patches, channels, height, width)

            image_features = self.model.extract_feature(pixel_values)
            vit_embeds = image_features.reshape(batch_size, -1, embed_dim)

            if self.visual_token_pruner is not None:
                vit_embeds, visual_keep_mask, visual_keep_indices, visual_scores = self.visual_token_pruner(vit_embeds)
                adaptor_dict["visual_token_keep_mask"] = visual_keep_mask
                adaptor_dict["visual_token_keep_indices"] = visual_keep_indices
                adaptor_dict["visual_token_scores"] = visual_scores

            if visual_keep_indices is None:
                visual_keep_indices = torch.arange(
                    vit_embeds.size(1),
                    device=vit_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)

            (
                inputs_embeds_work,
                adaptor_dict["language__ids"],
                adaptor_dict["language_inputs_mask"],
                compact_language_ids_mask,
            ) = self._replace_and_compact_visual_tokens(
                language_inputs=inputs_embeds_work,
                input_ids=input_ids,
                language_mask=adaptor_dict["language_inputs_mask"],
                language_ids_mask=adaptor_dict.get("language__ids_mask"),
                vit_embeds=vit_embeds.to(
                    device=inputs_embeds_work.device,
                    dtype=inputs_embeds_work.dtype,
                ),
                visual_keep_indices=visual_keep_indices,
            )
            if compact_language_ids_mask is not None:
                adaptor_dict["language__ids_mask"] = compact_language_ids_mask

        adaptor_dict["language_inputs"] = inputs_embeds_work.to(dtype=inputs_embeds_dtype)
        adaptor_order = [key for key in ("language", "driving", "budget") if f"{key}_inputs" in adaptor_dict]
        rebuild_packed_adaptor_inputs(adaptor_dict, adaptor_order)
        return adaptor_dict

    def _replace_and_compact_visual_tokens(
        self,
        language_inputs: torch.Tensor,
        input_ids: torch.Tensor,
        language_mask: torch.Tensor,
        language_ids_mask: Optional[torch.Tensor],
        vit_embeds: torch.Tensor,
        visual_keep_indices: torch.Tensor,
    ):
        """关键调用点：真正减少 LLM 序列长度必须在这里同步压缩 language 段。"""
        compact_inputs = []
        compact_ids = []
        compact_masks = []
        compact_ids_masks = [] if language_ids_mask is not None else None

        for batch_idx in range(input_ids.size(0)):
            visual_positions = torch.nonzero(
                input_ids[batch_idx] == self.img_context_token_id,
                as_tuple=False,
            ).squeeze(-1)
            if visual_positions.numel() < visual_keep_indices.size(1):
                raise ValueError(
                    "visual token count mismatch between placeholder positions and keep indices: "
                    f"{visual_positions.numel()} vs {visual_keep_indices.size(1)}"
                )

            kept_positions = visual_positions[visual_keep_indices[batch_idx]]
            updated_inputs = language_inputs[batch_idx].clone()
            updated_inputs[kept_positions] = vit_embeds[batch_idx]

            keep_token_mask = torch.ones(
                input_ids.size(1),
                device=input_ids.device,
                dtype=torch.bool,
            )
            keep_token_mask[visual_positions] = False
            keep_token_mask[kept_positions] = True

            compact_inputs.append(updated_inputs[keep_token_mask])
            compact_ids.append(input_ids[batch_idx][keep_token_mask])
            compact_masks.append(language_mask[batch_idx][keep_token_mask])
            if compact_ids_masks is not None:
                compact_ids_masks.append(language_ids_mask[batch_idx][keep_token_mask])

        compact_language_ids_mask = None
        if compact_ids_masks is not None:
            compact_language_ids_mask = torch.stack(compact_ids_masks, dim=0)

        return (
            torch.stack(compact_inputs, dim=0),
            torch.stack(compact_ids, dim=0),
            torch.stack(compact_masks, dim=0),
            compact_language_ids_mask,
        )
