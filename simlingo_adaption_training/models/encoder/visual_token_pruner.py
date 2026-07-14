import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VisualTokenPruner(nn.Module):
    """按固定配置对视觉 token 打分并生成 keep mask。"""

    def __init__(
        self,
        mode: str = "off",
        prune_ratio: float = 0.0,
        min_keep: int = 1,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.prune_ratio = float(prune_ratio)
        self.min_keep = int(min_keep)
        self._validate_mode(self.mode)
        self._validate_prune_ratio(self.prune_ratio)

    def forward(
        self,
        image_features: torch.Tensor,
        mode: Optional[str] = None,
        prune_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 masked features、keep mask 和分数。"""
        if image_features.dim() != 3:
            raise ValueError(f"image_features should be [B, N, D], got {image_features.shape}")

        mode = self.mode if mode is None else str(mode).strip().lower()
        prune_ratio = self.prune_ratio if prune_ratio is None else float(prune_ratio)
        self._validate_mode(mode)
        self._validate_prune_ratio(prune_ratio)

        batch_size, num_tokens, _ = image_features.shape
        keep_num = self._compute_keep_num(
            num_tokens=num_tokens,
            prune_ratio=prune_ratio,
            min_keep=self.min_keep,
        )

        if mode == "off" or prune_ratio <= 0.0 or keep_num >= num_tokens:
            keep_mask = torch.ones(
                batch_size,
                num_tokens,
                device=image_features.device,
                dtype=torch.bool,
            )
            scores = torch.zeros(
                batch_size,
                num_tokens,
                device=image_features.device,
                dtype=image_features.dtype,
            )
            return image_features, keep_mask, scores

        if mode == "random":
            keep_mask, scores = self._random_select(image_features, keep_num)
        elif mode == "prune2drive":
            keep_mask, scores = self._prune2drive_select(image_features, keep_num)
        else:
            raise ValueError(f"Unsupported visual token prune mode: {mode}")

        masked_features = image_features.clone()
        masked_features[~keep_mask] = 0.0
        return masked_features, keep_mask, scores

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in {"off", "random", "prune2drive"}:
            raise ValueError(f"Unsupported visual token prune mode: {mode}")

    @staticmethod
    def _validate_prune_ratio(prune_ratio: float) -> None:
        if not 0.0 <= prune_ratio <= 1.0:
            raise ValueError(f"prune_ratio should be in [0, 1], got {prune_ratio}")

    @staticmethod
    def _compute_keep_num(
        num_tokens: int,
        prune_ratio: float,
        min_keep: int,
    ) -> int:
        keep_num = int(math.ceil(num_tokens * (1.0 - prune_ratio)))
        keep_num = max(int(min_keep), keep_num)
        return min(num_tokens, keep_num)

    @staticmethod
    def _random_select(
        image_features: torch.Tensor,
        keep_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = image_features.shape
        scores = torch.rand(
            batch_size,
            num_tokens,
            device=image_features.device,
            dtype=image_features.dtype,
        )
        topk_idx = torch.topk(scores, k=min(keep_num, num_tokens), dim=1).indices
        keep_mask = torch.zeros(
            batch_size,
            num_tokens,
            device=image_features.device,
            dtype=torch.bool,
        )
        keep_mask.scatter_(1, topk_idx, True)
        return keep_mask, scores

    @staticmethod
    def _prune2drive_select(
        image_features: torch.Tensor,
        keep_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = image_features.shape
        keep_masks = []
        all_scores = []

        for batch_idx in range(batch_size):
            feats = image_features[batch_idx]
            if keep_num >= num_tokens:
                keep_masks.append(torch.ones(num_tokens, device=feats.device, dtype=torch.bool))
                all_scores.append(torch.zeros(num_tokens, device=feats.device, dtype=feats.dtype))
                continue

            normalized = F.normalize(feats, p=2, dim=-1, eps=1e-6)
            cosine_sim = torch.matmul(normalized, normalized.t())
            distance_matrix = 1.0 - cosine_sim
            distance_matrix.fill_diagonal_(float("inf"))

            selected_mask = torch.zeros(num_tokens, device=feats.device, dtype=torch.bool)
            selected_indices = torch.empty(keep_num, device=feats.device, dtype=torch.long)
            scores = torch.zeros(num_tokens, device=feats.device, dtype=feats.dtype)

            for i in range(keep_num):
                if i == 0:
                    current_scores = distance_matrix.min(dim=1).values
                    current_scores[selected_mask] = float("-inf")
                else:
                    selected_distances = distance_matrix[selected_mask, :]
                    current_scores = selected_distances.min(dim=0).values
                    current_scores[selected_mask] = float("-inf")

                next_idx = torch.argmax(current_scores)
                selected_indices[i] = next_idx
                selected_mask[next_idx] = True
                scores = current_scores

            keep_mask = torch.zeros(num_tokens, device=feats.device, dtype=torch.bool)
            keep_mask[selected_indices] = True
            keep_masks.append(keep_mask)
            all_scores.append(scores)

        return torch.stack(keep_masks, dim=0), torch.stack(all_scores, dim=0)
