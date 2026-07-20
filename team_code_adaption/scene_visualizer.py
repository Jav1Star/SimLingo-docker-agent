"""Lightweight scene-frame visualization for split-agent evaluation."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from team_code_adaption.simlingo_utils import get_camera_intrinsics, project_points


def _draw_points(draw: ImageDraw.ImageDraw, points: Any, intrinsics: np.ndarray, color, radius: int) -> None:
    if points is None:
        return
    try:
        array = np.asarray(points, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2 or array.shape[0] == 0:
            return
        projected = project_points(array, intrinsics)
        for x, y in np.asarray(projected).reshape(-1, 2):
            if np.isfinite(x) and np.isfinite(y):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    except (TypeError, ValueError, IndexError):
        return


def save_scene_visualization(
    *,
    camera_bgr: np.ndarray,
    annotated_path: Path,
    raw_path: Path | None,
    target_points: Any,
    pred_route: Any,
    pred_speed_wps: Any,
    metadata: dict[str, Any],
) -> None:
    """Save a front-camera frame with projected model outputs and an information panel."""
    camera = np.asarray(camera_bgr)
    if camera.ndim != 3 or camera.shape[2] < 3:
        raise ValueError(f"camera_bgr must have shape [H,W,3+], got {camera.shape}")
    rgb = np.ascontiguousarray(camera[:, :, :3][:, :, ::-1])

    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(raw_path)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    intrinsics = np.asarray(get_camera_intrinsics(image.width, image.height, 110))
    _draw_points(draw, target_points, intrinsics, (0, 80, 255), 4)
    _draw_points(draw, pred_route, intrinsics, (255, 40, 40), 3)
    _draw_points(draw, pred_speed_wps, intrinsics, (20, 220, 40), 2)

    font = ImageFont.load_default()
    lines = [
        f"route: {metadata.get('route_key', '-')}",
        f"frame: {metadata.get('frame_id', '-')}  timestamp: {metadata.get('timestamp', '-')}",
        f"speed: {metadata.get('speed_mps', '-')} m/s  budget: {metadata.get('budget_mode', '-')} / {metadata.get('budget_value', '-')}",
        f"token keep ratio: {metadata.get('visual_token_keep_ratio', '-')}",
        f"control: steer={metadata.get('steer', '-')} throttle={metadata.get('throttle', '-')} brake={metadata.get('brake', '-')}",
        f"latency ms: {metadata.get('stage_durations_ms', {})}",
    ]
    prompt = str(metadata.get("prompt", "") or "")
    if prompt:
        lines.extend(textwrap.wrap(f"prompt: {prompt}", width=120))

    line_height = 14
    panel = Image.new("RGB", (image.width, max(90, line_height * (len(lines) + 1))), "black")
    panel_draw = ImageDraw.Draw(panel)
    for index, line in enumerate(lines):
        panel_draw.text((8, 6 + index * line_height), line, font=font, fill="white")
    combined = Image.new("RGB", (image.width, image.height + panel.height))
    combined.paste(image, (0, 0))
    combined.paste(panel, (0, image.height))
    combined.save(annotated_path)
