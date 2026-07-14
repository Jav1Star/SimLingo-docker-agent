import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VisualTokenPruner(nn.Module):
    """按固定配置对视觉 token 打分并返回真实保留结果。"""

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回保留后的特征、keep mask、keep indices 和分数。"""
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
            keep_indices = torch.arange(
                num_tokens,
                device=image_features.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            scores = torch.zeros(
                batch_size,
                num_tokens,
                device=image_features.device,
                dtype=image_features.dtype,
            )
            return image_features, keep_mask, keep_indices, scores

        if mode == "random":
            scores = self._random_score(image_features)
        elif mode == "prune2drive":
            scores = self._prune2drive_score(image_features)
        else:
            raise ValueError(f"Unsupported visual token prune mode: {mode}")

        keep_mask, keep_indices = self._select_keep_tokens(scores, keep_num)
        kept_features = image_features.gather(
            1,
            keep_indices.unsqueeze(-1).expand(-1, -1, image_features.size(-1)),
        )
        return kept_features, keep_mask, keep_indices, scores

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
    def _random_score(image_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = image_features.shape
        return torch.rand(
            batch_size,
            num_tokens,
            device=image_features.device,
            dtype=image_features.dtype,
        )

    @staticmethod
    def _select_keep_tokens(
        scores: torch.Tensor,
        keep_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens = scores.shape
        topk_idx = torch.topk(scores, k=min(keep_num, num_tokens), dim=1).indices
        keep_indices = topk_idx.sort(dim=1).values
        keep_mask = torch.zeros(
            batch_size,
            num_tokens,
            device=scores.device,
            dtype=torch.bool,
        )
        keep_mask.scatter_(1, keep_indices, True)
        return keep_mask, keep_indices

    @staticmethod
    def _prune2drive_score(image_features: torch.Tensor) -> torch.Tensor:
        """关键调用点：在线推理改用 O(BND) 打分，避免原始 N^2 选点吞掉延迟。"""
        normalized = F.normalize(image_features, p=2, dim=-1, eps=1e-6)
        global_anchor = F.normalize(normalized.mean(dim=1, keepdim=True), p=2, dim=-1, eps=1e-6)
        cosine_to_anchor = (normalized * global_anchor).sum(dim=-1)
        return 1.0 - cosine_to_anchor
