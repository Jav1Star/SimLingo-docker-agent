#!/usr/bin/env python3
"""Build an mp4 video from numbered scene-frame images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert scene-frame images to an mp4 video.")
    parser.add_argument("--imgs-dir", type=str, required=True, help="Directory containing frame images.")
    parser.add_argument("--output", type=str, default="", help="Output mp4 path. Defaults to <imgs-dir>/../scene_frames.mp4")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument("--metric-json", type=str, default="", help="Optional metric_info.json used to overlay execution plan.")
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group(0)) if match else 10**12, path.name)


def collect_frames(imgs_dir: Path) -> List[Path]:
    frames = sorted(
        (path for path in imgs_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in {imgs_dir}")
    return frames


def extract_frame_step(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if not match:
        raise ValueError(f"Frame filename must contain step number: {path}")
    return int(match.group(0))


def extract_layer_mask(frame_data: dict) -> Optional[List[int]]:
    eval_budget = frame_data.get("eval_budget", {})
    if not isinstance(eval_budget, dict):
        return None

    path_mask_hard = eval_budget.get("path_mask_hard")
    if isinstance(path_mask_hard, list) and path_mask_hard:
        return [int(value) for value in path_mask_hard]

    execution_plan = eval_budget.get("execution_plan", {})
    if not isinstance(execution_plan, dict):
        return None
    full_mask = execution_plan.get("full_layer_mask_by_batch")
    if isinstance(full_mask, list) and full_mask and isinstance(full_mask[0], list):
        return [int(value) for value in full_mask[0]]
    return None


def load_layer_masks(metric_json: Path) -> Dict[int, List[int]]:
    with metric_json.open("r", encoding="utf-8") as handle:
        metric_info = json.load(handle)
    if not isinstance(metric_info, dict):
        raise ValueError(f"metric-json must contain a JSON object: {metric_json}")

    masks: Dict[int, List[int]] = {}
    for step_key, frame_data in metric_info.items():
        if not isinstance(frame_data, dict):
            continue
        try:
            step = int(step_key)
        except ValueError:
            continue
        mask = extract_layer_mask(frame_data)
        if mask is not None:
            masks[step] = mask
    if not masks:
        raise ValueError(f"No execution plan masks found in {metric_json}")
    return masks


def draw_execution_plan(frame, mask: List[int]) -> None:
    total_layers = len(mask)
    if total_layers == 0:
        return

    height, width = frame.shape[:2]
    bar_h = 24
    margin_x = 18
    margin_y = 14
    gap = 2
    cell_w = max(3, (width - 2 * margin_x - gap * (total_layers - 1)) // total_layers)
    active_color = (230, 120, 35)
    inactive_color = (145, 145, 145)
    border_color = (245, 245, 245)

    # 顶部半透明底色保证不同场景亮度下 execution plan 都可读。
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, margin_y * 2 + bar_h), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, dst=frame)

    for layer_idx, selected in enumerate(mask):
        x0 = margin_x + layer_idx * (cell_w + gap)
        x1 = min(width - margin_x, x0 + cell_w)
        y0 = margin_y
        y1 = min(height - 1, y0 + bar_h)
        color = active_color if int(selected) else inactive_color
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=-1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), border_color, thickness=1)


def main() -> None:
    args = parse_args()
    imgs_dir = Path(args.imgs_dir).expanduser().resolve()
    if not imgs_dir.is_dir():
        raise NotADirectoryError(f"imgs-dir is not a directory: {imgs_dir}")
    if args.fps <= 0:
        raise ValueError("fps must be > 0")

    output_path = Path(args.output).expanduser().resolve() if args.output else imgs_dir.parent / "scene_frames.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(imgs_dir)
    layer_masks = load_layer_masks(Path(args.metric_json).expanduser().resolve()) if args.metric_json else None
    first_frame = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError(f"Failed to read first frame: {frames[0]}")
    if layer_masks is not None:
        first_mask = layer_masks.get(extract_frame_step(frames[0]))
        if first_mask is None:
            raise ValueError(f"No execution plan mask for first frame: {frames[0]}")
        draw_execution_plan(first_frame, first_mask)

    height, width = first_frame.shape[:2]
    # mp4v 在当前环境最少依赖，直接写出可被常见播放器读取的 mp4。
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame size mismatch: {frame_path}")
            if layer_masks is not None:
                step = extract_frame_step(frame_path)
                mask = layer_masks.get(step)
                if mask is None:
                    raise ValueError(f"No execution plan mask for frame step {step}: {frame_path}")
                draw_execution_plan(frame, mask)
            writer.write(frame)
    finally:
        writer.release()

    print(f"[images-to-video] frames={len(frames)} fps={args.fps} output={output_path}")


if __name__ == "__main__":
    main()
