"""Write compact, per-route behavior samples for remote SimLingo evaluation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class BehaviorWindowLogger:
    """Aggregate inference timing and layer masks into fixed-size JSON windows."""

    def __init__(
        self,
        output_dir: Path,
        *,
        route_id: str,
        route_key: str,
        prune_ratio: float | None,
        mode: str,
        fixed_budget: float,
        predict_language_enabled: bool,
        predict_language_max_new_tokens: int,
        predict_language_stride: int,
        window_size: int = 50,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = int(window_size)
        self.parameters = {
            "route_id": str(route_id),
            "route_key": str(route_key),
            "prune_ratio": prune_ratio,
            "mode": str(mode),
            "fixed_budget": float(fixed_budget),
            "predict_language_enabled": bool(predict_language_enabled),
            "predict_language_max_new_tokens": int(predict_language_max_new_tokens),
            "predict_language_stride": int(predict_language_stride),
        }
        self._records: list[dict[str, Any]] = []
        self._durations: list[float] = []
        self._window_start = 0

    def add_step(
        self,
        *,
        step: int,
        timestamp: float,
        remote_inference_sec: float,
        layer_active_mask: list[int] | None,
    ) -> None:
        mask = None if layer_active_mask is None else [int(value) for value in layer_active_mask]
        self._records.append(
            {
                "step": int(step),
                "timestamp": float(timestamp),
                "layer_active_mask": mask,
            }
        )
        self._durations.append(float(remote_inference_sec))
        if len(self._records) >= self.window_size:
            self._flush(complete=True)

    def finalize(self) -> None:
        if self._records:
            self._flush(complete=False)

    def _flush(self, *, complete: bool) -> None:
        window_end = self._window_start + self.window_size
        sampled_at_unix = time.time()
        payload = {
            "schema_version": 1,
            **self.parameters,
            "window_step_start": self._window_start,
            "window_step_end": window_end,
            "window_complete": bool(complete),
            "sample_count": len(self._records),
            "first_observed_step": self._records[0]["step"],
            "last_observed_step": self._records[-1]["step"],
            "avg_remote_inference_sec": sum(self._durations) / len(self._durations),
            "steps": self._records,
            "sampled_at_unix": sampled_at_unix,
            "sampled_at": self._format_unix_ts(sampled_at_unix),
        }
        destination = self.output_dir / f"step_{self._window_start}_{window_end}.json"
        temporary = destination.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2, ensure_ascii=False)
            outfile.write("\n")
        os.replace(temporary, destination)
        self._records = []
        self._durations = []
        self._window_start = window_end

    @staticmethod
    def _format_unix_ts(timestamp: float) -> str:
        whole = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp))
        millis = int((timestamp % 1.0) * 1000)
        zone = time.strftime("%z", time.localtime(timestamp))
        return f"{whole}.{millis:03d}{zone}"
