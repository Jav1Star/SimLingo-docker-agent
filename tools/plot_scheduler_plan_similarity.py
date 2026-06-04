#!/usr/bin/env python3
"""Plot per-route scheduler-plan cosine-similarity heatmaps from eval metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read metric_info.json files produced during Bench2Drive eval and plot "
            "frame-by-frame cosine-similarity heatmaps for scheduler execution plans."
        )
    )
    parser.add_argument(
        "--viz-root",
        type=Path,
        required=True,
        help="Root directory that contains route visualization folders, e.g. eval_results/.../viz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <viz-root>/scheduler_plan_similarity.",
    )
    parser.add_argument(
        "--metric-glob",
        default="**/metric_info.json",
        help="Glob pattern under --viz-root for metric files.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch item index inside logged flattened_by_batch. Eval normally uses 0.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame when plotting very long routes.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Uniformly subsample to at most this many frames before computing the matrix.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the cosine-similarity matrix and frame ids as .npy files.",
    )
    return parser.parse_args()


def route_label_from_metric_path(metric_path: Path) -> str:
    parts = metric_path.parts
    if "debug_viz" in parts:
        idx = parts.index("debug_viz")
        if idx > 0:
            route_dir = parts[idx - 1]
            return route_dir
    return metric_path.parent.parent.name


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "route"


def load_metric_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def frame_sort_key(item: Tuple[str, object]) -> float:
    key, _ = item
    try:
        return int(key)
    except ValueError:
        return math.inf


def extract_plan_vector(frame_data: dict, batch_index: int) -> Optional[np.ndarray]:
    eval_budget = frame_data.get("eval_budget")
    if not isinstance(eval_budget, dict):
        return None
    execution_plan = eval_budget.get("execution_plan")
    if not isinstance(execution_plan, dict):
        return None

    flattened_by_batch = execution_plan.get("flattened_by_batch")
    if not isinstance(flattened_by_batch, list) or batch_index >= len(flattened_by_batch):
        return None

    vector = np.asarray(flattened_by_batch[batch_index], dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def iter_frame_vectors(metric_info: dict, batch_index: int) -> Tuple[np.ndarray, np.ndarray]:
    frame_ids = []
    vectors = []
    for frame_key, frame_data in sorted(metric_info.items(), key=frame_sort_key):
        if not isinstance(frame_data, dict):
            continue
        vector = extract_plan_vector(frame_data, batch_index)
        if vector is None:
            continue
        frame_ids.append(int(frame_key))
        vectors.append(vector)

    if not vectors:
        return np.asarray([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)

    dims = {v.size for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent scheduler vector lengths: {sorted(dims)}")
    return np.asarray(frame_ids, dtype=np.int64), np.stack(vectors, axis=0)


def subsample(frame_ids: np.ndarray, vectors: np.ndarray, frame_stride: int, max_frames: Optional[int]):
    if frame_stride > 1:
        frame_ids = frame_ids[::frame_stride]
        vectors = vectors[::frame_stride]

    if max_frames is not None and frame_ids.size > max_frames:
        keep = np.linspace(0, frame_ids.size - 1, max_frames).round().astype(np.int64)
        frame_ids = frame_ids[keep]
        vectors = vectors[keep]
    return frame_ids, vectors


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def plot_heatmap(matrix: np.ndarray, frame_ids: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_count = matrix.shape[0]
    fig_size = min(max(frame_count / 40.0, 5.0), 18.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=160)
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="viridis", origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")

    tick_count = min(8, frame_count)
    if tick_count > 0:
        tick_positions = np.linspace(0, frame_count - 1, tick_count).round().astype(int)
        tick_labels = [str(int(frame_ids[pos])) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def find_metric_files(viz_root: Path, metric_glob: str) -> List[Path]:
    return sorted(path for path in viz_root.glob(metric_glob) if path.is_file())


def main() -> None:
    args = parse_args()

    global np
    import numpy as np_module

    np = np_module
    viz_root = args.viz_root.expanduser().resolve()
    out_dir = (args.out_dir or (viz_root / "scheduler_plan_similarity")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_files = find_metric_files(viz_root, args.metric_glob)
    if not metric_files:
        raise FileNotFoundError(f"No metric_info.json files found under {viz_root} with glob {args.metric_glob!r}")

    generated = 0
    skipped = 0
    for metric_file in metric_files:
        route_label = safe_name(route_label_from_metric_path(metric_file))
        try:
            metric_info = load_metric_json(metric_file)
        except ValueError as exc:
            print(f"[skip] {metric_file}: {exc}")
            skipped += 1
            continue
        frame_ids, vectors = iter_frame_vectors(metric_info, args.batch_index)
        frame_ids, vectors = subsample(frame_ids, vectors, args.frame_stride, args.max_frames)

        if vectors.shape[0] < 2:
            print(f"[skip] {metric_file}: fewer than 2 frames with scheduler execution_plan")
            skipped += 1
            continue

        matrix = cosine_similarity_matrix(vectors)
        png_path = out_dir / f"{route_label}_scheduler_plan_cosine_heatmap.png"
        plot_heatmap(matrix, frame_ids, f"Route {route_label} scheduler-plan cosine similarity", png_path)

        if args.save_npy:
            np.save(out_dir / f"{route_label}_scheduler_plan_cosine.npy", matrix)
            np.save(out_dir / f"{route_label}_frames.npy", frame_ids)

        print(f"[ok] {metric_file} -> {png_path}")
        generated += 1

    print(f"Generated {generated} heatmap(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
