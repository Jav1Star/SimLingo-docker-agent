#!/usr/bin/env python3
"""Turn saved SimLingo scene frames and their scheduler log into an MP4.

The annotated scene frames already contain the camera image, target point,
predicted routes and driving-frame description.  This tool adds the per-frame
24-layer execution state from ``remote_scheduler_plan.jsonl`` and encodes the
result as a video.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a video from SimLingo scene frames with the 24-layer activation plan."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        help="Evaluation run directory containing scene_frames/ and metric/.",
    )
    source.add_argument(
        "--imgs-dir",
        help="Frame directory (normally scene_frames/annotated).",
    )
    parser.add_argument("--output", default="", help="Output MP4 path.")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS. By default (0), infer real-time FPS from driving-log timestamps.",
    )
    parser.add_argument(
        "--scheduler-jsonl",
        default="",
        help="remote_scheduler_plan.jsonl path; auto-detected with --run-dir.",
    )
    parser.add_argument(
        "--metric-json",
        default="",
        help="Optional metric_info.json fallback for older evaluation outputs.",
    )
    parser.add_argument(
        "--allow-missing-plan",
        action="store_true",
        help="Keep frames with no matching layer plan and display N/A.",
    )
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 10**12, path.name)


def frame_step(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    if not match:
        raise ValueError(f"Frame filename has no numeric step: {path}")
    return int(match.group())


def collect_frames(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Frame directory does not exist: {directory}")
    frames = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No PNG/JPEG frames found in {directory}")
    return frames


def _binary_mask(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or not value:
        return None
    if isinstance(value[0], list):
        value = value[0]
    if not value or any(isinstance(item, (dict, list)) for item in value):
        return None
    return [int(bool(item)) for item in value]


def extract_layer_mask(record: Dict[str, Any]) -> Optional[List[int]]:
    """Read both current split-agent logs and older metric_info layouts."""
    candidates: Iterable[Any] = (
        record.get("path_mask_hard"),
        record.get("layer_active_mask"),
        record.get("full_layer_mask_by_batch"),
        record.get("scheduler_plan_debug", {}).get("layer_active_mask")
        if isinstance(record.get("scheduler_plan_debug"), dict)
        else None,
    )
    for candidate in candidates:
        mask = _binary_mask(candidate)
        if mask is not None:
            return mask

    # Recursion is deliberately restricted to known containers so large feature
    # arrays/logits cannot accidentally be interpreted as a layer mask.
    for key in ("eval_budget", "execution_plan", "scheduler_plan_meta", "scheduler_budget_meta"):
        child = record.get(key)
        if isinstance(child, dict):
            mask = extract_layer_mask(child)
            if mask is not None:
                return mask
    return None


def load_jsonl_masks(path: Path) -> Dict[int, List[int]]:
    masks: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or "step" not in record:
                continue
            mask = extract_layer_mask(record)
            if mask is not None:
                masks[int(record["step"])] = mask
    return masks


def load_metric_masks(path: Path) -> Dict[int, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Metric JSON must contain an object: {path}")
    masks: Dict[int, List[int]] = {}
    for key, record in data.items():
        if not isinstance(record, dict):
            continue
        try:
            step = int(key)
        except (TypeError, ValueError):
            continue
        mask = extract_layer_mask(record)
        if mask is not None:
            masks[step] = mask
    return masks


def load_frame_timestamps(
    metric_path: Optional[Path], scheduler_path: Optional[Path]
) -> Dict[int, float]:
    """Load simulation timestamps, preferring the compact metric_info mapping."""
    timestamps: Dict[int, float] = {}
    if metric_path is not None and metric_path.is_file():
        with metric_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for key, record in data.items():
                if not isinstance(record, dict):
                    continue
                update = record.get("eval_budget", {}).get("last_update", {})
                timestamp = update.get("timestamp") if isinstance(update, dict) else None
                if timestamp is not None:
                    try:
                        timestamps[int(key)] = float(timestamp)
                    except (TypeError, ValueError):
                        pass
    if timestamps or scheduler_path is None or not scheduler_path.is_file():
        return timestamps

    with scheduler_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("step") is not None and record.get("timestamp") is not None:
                timestamps[int(record["step"])] = float(record["timestamp"])
    return timestamps


def infer_fps(frames: List[Path], timestamps: Dict[int, float]) -> float:
    frame_times = [timestamps.get(frame_step(path)) for path in frames]
    deltas = [
        current - previous
        for previous, current in zip(frame_times, frame_times[1:])
        if previous is not None and current is not None and current > previous
    ]
    if not deltas:
        raise ValueError(
            "Cannot infer FPS: no matching increasing timestamps in metric_info.json "
            "or remote_scheduler_plan.jsonl. Pass --fps explicitly."
        )
    median_delta = statistics.median(deltas)
    return 1.0 / median_delta


def draw_layer_plan(frame: Any, mask: Optional[List[int]]) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    panel_height = 58
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, dst=frame)

    label = "MODEL LAYER ACTIVATION"
    if mask is None:
        cv2.putText(frame, label + ": N/A", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return

    count = len(mask)
    margin, gap = 12, 3
    cell_width = max(5, (width - 2 * margin - (count - 1) * gap) // count)
    used_width = count * cell_width + (count - 1) * gap
    start_x = max(margin, (width - used_width) // 2)
    active = sum(mask)
    cv2.putText(
        frame,
        f"{label}  active {active}/{count}",
        (start_x, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    for index, selected in enumerate(mask):
        x0 = start_x + index * (cell_width + gap)
        x1 = min(width - 1, x0 + cell_width)
        color = (45, 190, 70) if selected else (90, 90, 90)
        cv2.rectangle(frame, (x0, 20), (x1, 44), color, -1)
        cv2.rectangle(frame, (x0, 20), (x1, 44), (225, 225, 225), 1)
        cv2.putText(frame, str(index + 1), (x0, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (220, 220, 220), 1, cv2.LINE_AA)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    if args.run_dir:
        requested_dir = Path(args.run_dir).expanduser().resolve()
        if not requested_dir.is_dir():
            raise NotADirectoryError(f"Run directory does not exist: {requested_dir}")

        # Accept either the final run directory or any of its ancestors (for
        # example the route-level .../viz/024RouteScenario... directory).
        if (requested_dir / "scene_frames" / "annotated").is_dir():
            run_dir = requested_dir
        else:
            candidates = sorted(
                path.parent.parent
                for path in requested_dir.rglob("scene_frames/annotated")
                if path.is_dir()
            )
            if not candidates:
                raise NotADirectoryError(
                    "No scene_frames/annotated directory found under "
                    f"run-dir: {requested_dir}"
                )
            if len(candidates) > 1:
                choices = "\n  ".join(str(path) for path in candidates)
                raise ValueError(
                    f"Multiple evaluation runs found under {requested_dir}. "
                    "Pass one of these directories as --run-dir:\n  "
                    f"{choices}"
                )
            run_dir = candidates[0]
            print(f"[images-to-video] discovered run-dir={run_dir}")
        imgs_dir = run_dir / "scene_frames" / "annotated"
        output = Path(args.output).expanduser().resolve() if args.output else run_dir / "scene_frames.mp4"
        scheduler = Path(args.scheduler_jsonl).expanduser().resolve() if args.scheduler_jsonl else run_dir / "metric" / "remote_scheduler_plan.jsonl"
        metric = Path(args.metric_json).expanduser().resolve() if args.metric_json else run_dir / "metric" / "metric_info.json"
    else:
        imgs_dir = Path(args.imgs_dir).expanduser().resolve()
        output = Path(args.output).expanduser().resolve() if args.output else imgs_dir.parent / "scene_frames.mp4"
        scheduler = Path(args.scheduler_jsonl).expanduser().resolve() if args.scheduler_jsonl else None
        metric = Path(args.metric_json).expanduser().resolve() if args.metric_json else None
    return imgs_dir, output, scheduler, metric


def main() -> None:
    args = parse_args()
    if args.fps < 0:
        raise ValueError("fps must be >= 0 (0 means auto)")
    imgs_dir, output, scheduler, metric = resolve_paths(args)
    frames = collect_frames(imgs_dir)

    fps = args.fps
    if fps == 0:
        fps = infer_fps(frames, load_frame_timestamps(metric, scheduler))
        print(f"[images-to-video] inferred real-time fps={fps:g}")

    masks: Dict[int, List[int]] = {}
    if scheduler is not None and scheduler.is_file():
        masks.update(load_jsonl_masks(scheduler))
    if metric is not None and metric.is_file():
        # Scheduler data takes precedence because it contains the actual plan in
        # current split-agent evaluations.
        for step, mask in load_metric_masks(metric).items():
            masks.setdefault(step, mask)
    if not masks and not args.allow_missing_plan:
        raise ValueError(
            "No layer activation plans found. Pass --scheduler-jsonl, or use "
            "--allow-missing-plan to render N/A."
        )

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Cannot read frame: {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output}")

    missing: List[int] = []
    try:
        for path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot read frame: {path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame size differs from first frame: {path}")
            step = frame_step(path)
            mask = masks.get(step)
            if mask is None:
                missing.append(step)
                if not args.allow_missing_plan:
                    raise ValueError(f"No layer activation plan for frame step {step}")
            draw_layer_plan(frame, mask)
            writer.write(frame)
    finally:
        writer.release()

    suffix = f" missing_plans={len(missing)}" if missing else ""
    print(f"[images-to-video] frames={len(frames)} fps={fps:g}{suffix} output={output}")


if __name__ == "__main__":
    main()
