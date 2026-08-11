#!/usr/bin/env python3
"""Live web viewer for SimLingo split-agent evaluation outputs.

This tool is intentionally read-only. It watches evaluation output folders,
serves annotated scene frames, and joins them with per-frame token and layer
mask records by the shared ``step`` field.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_DIRS = [
    Path("/data/gaoshuo/1-4-J-104/test_options"),
    Path("/data/gaoshuo/1-4-J-104/input"),
]
DEFAULT_LAYERS = 24
# InternVL2-1B uses 256 visual tokens per image patch with the local config
# (448 image size, 14 patch size, 0.5 downsample ratio). The SimLingo
# datamodule splits the front image into 2 patches for this evaluation path.
DEFAULT_ORIGINAL_VISUAL_TOKENS = 512
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    out_root: Path
    route_id: str
    mode: str
    annotated_dir: Path
    debug_dir: Path
    metric_dir: Path
    token_path: Path | None
    result_path: Path | None
    latest_frame: Path | None
    latest_mtime: float


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _extract_out_roots(config_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if not config_dir.exists():
        return roots
    pattern = re.compile(r"^\s*out_root\s*:\s*(.*?)\s*(?:#.*)?$")
    for yaml_path in sorted(config_dir.rglob("*.yaml")):
        try:
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                raw_value = match.group(1).strip().strip("'\"")
                if raw_value:
                    roots.append(Path(os.path.expandvars(os.path.expanduser(raw_value))).resolve())
        except OSError:
            continue
    return sorted(set(roots))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _latest_image(annotated_dir: Path) -> Path | None:
    newest: tuple[float, Path] | None = None
    try:
        entries = list(annotated_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0] or (mtime == newest[0] and entry.name > newest[1].name):
            newest = (mtime, entry)
    return None if newest is None else newest[1]


def _step_from_frame(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stem)
    except ValueError:
        return None


def _route_id_from_annotated_dir(out_root: Path, annotated_dir: Path) -> str:
    try:
        rel = annotated_dir.relative_to(out_root)
    except ValueError:
        return "unknown"
    parts = rel.parts
    if "viz" in parts:
        idx = parts.index("viz")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "debug_viz" in parts:
        idx = parts.index("debug_viz")
        if idx > 0:
            return parts[idx - 1]
    return parts[0] if parts else "unknown"


def _mode_from_annotated_dir(out_root: Path, annotated_dir: Path) -> str:
    try:
        parts = annotated_dir.relative_to(out_root).parts
    except ValueError:
        return "unknown"
    for mode in ("rule_based", "fixed", "random", "smart_assigner"):
        if mode in parts:
            return mode
    return "unknown"


def _find_token_path(out_root: Path, route_id: str) -> Path | None:
    token_route_ids = _candidate_route_ids(route_id)
    candidates = [
        *(out_root / "res" / f"{item}_token.json" for item in token_route_ids),
        *(out_root / route_id / "res" / f"{item}_token.json" for item in token_route_ids),
        out_root / route_id / "route_token.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches: list[Path] = []
    try:
        for token_route_id in token_route_ids:
            matches.extend(out_root.glob(f"**/{token_route_id}_token.json"))
        matches.extend(out_root.glob("**/route_token.json"))
    except OSError:
        return None
    existing = [path for path in matches if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _candidate_route_ids(route_id: str) -> list[str]:
    route_ids = [route_id]
    prefix_match = re.match(r"^(\d+)", route_id)
    if prefix_match and prefix_match.group(1) not in route_ids:
        route_ids.append(prefix_match.group(1).zfill(3))
    return route_ids


def _find_result_path(out_root: Path, route_id: str) -> Path | None:
    route_ids = _candidate_route_ids(route_id)
    candidates = [
        *(out_root / "res" / f"{item}_res.json" for item in route_ids),
        *(out_root / route_id / "res" / f"{item}_res.json" for item in route_ids),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches: list[Path] = []
    try:
        for item in route_ids:
            matches.extend(out_root.glob(f"**/{item}_res.json"))
    except OSError:
        return None
    existing = [path for path in matches if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _read_result_status(result_path: Path | None) -> dict[str, Any]:
    if result_path is None or not result_path.is_file():
        return {
            "path": None,
            "status": None,
            "progress": None,
            "completed": False,
        }
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "path": str(result_path),
            "status": None,
            "progress": None,
            "completed": False,
        }
    checkpoint = data.get("_checkpoint", {}) if isinstance(data, dict) else {}
    global_record = checkpoint.get("global_record", {}) if isinstance(checkpoint, dict) else {}
    progress = checkpoint.get("progress") if isinstance(checkpoint, dict) else None
    status = global_record.get("status") if isinstance(global_record, dict) else None
    completed = status == "Completed" or (
        isinstance(progress, list)
        and len(progress) >= 2
        and isinstance(progress[0], int)
        and isinstance(progress[1], int)
        and progress[1] > 0
        and progress[0] >= progress[1]
    )
    return {
        "path": str(result_path),
        "status": status,
        "progress": progress,
        "completed": bool(completed),
    }


def _run_id(out_root: Path, annotated_dir: Path) -> str:
    try:
        return str(annotated_dir.relative_to(out_root.parent))
    except ValueError:
        return str(annotated_dir)


class LiveEvalState:
    def __init__(
        self,
        config_dirs: list[Path],
        extra_roots: list[Path],
        layer_count: int,
        original_visual_tokens: int,
    ) -> None:
        self.config_dirs = [config_dir.resolve() for config_dir in config_dirs]
        self.extra_roots = [root.resolve() for root in extra_roots]
        self.layer_count = layer_count
        self.original_visual_tokens = original_visual_tokens

    def out_roots(self) -> list[Path]:
        roots: list[Path] = []
        for config_dir in self.config_dirs:
            roots.extend(_extract_out_roots(config_dir))
        roots.extend(self.extra_roots)
        return sorted(set(root.resolve() for root in roots))

    def discover_runs(self) -> list[RunInfo]:
        runs: list[RunInfo] = []
        for out_root in self.out_roots():
            if not out_root.exists():
                continue
            try:
                annotated_dirs = list(out_root.glob("**/scene_frames/annotated"))
            except OSError:
                continue
            for annotated_dir in annotated_dirs:
                if not annotated_dir.is_dir():
                    continue
                latest = _latest_image(annotated_dir)
                debug_dir = annotated_dir.parent.parent
                metric_dir = debug_dir / "metric"
                route_id = _route_id_from_annotated_dir(out_root, annotated_dir)
                mode = _mode_from_annotated_dir(out_root, annotated_dir)
                token_path = _find_token_path(out_root, route_id)
                result_path = _find_result_path(out_root, route_id)
                mtime = 0.0
                if latest is not None:
                    try:
                        mtime = latest.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                runs.append(
                    RunInfo(
                        run_id=_run_id(out_root, annotated_dir),
                        out_root=out_root,
                        route_id=route_id,
                        mode=mode,
                        annotated_dir=annotated_dir,
                        debug_dir=debug_dir,
                        metric_dir=metric_dir,
                        token_path=token_path,
                        result_path=result_path,
                        latest_frame=latest,
                        latest_mtime=mtime,
                    )
                )
        return sorted(runs, key=lambda run: (run.latest_mtime, run.run_id), reverse=True)

    def select_run(self, run_id: str | None) -> RunInfo | None:
        runs = self.discover_runs()
        if not runs:
            return None
        if not run_id or run_id == "latest":
            return runs[0]
        for run in runs:
            if run.run_id == run_id:
                return run
        return None

    def safe_frame_path(self, run_id: str, name: str) -> Path | None:
        run = self.select_run(run_id)
        if run is None:
            return None
        frame_path = (run.annotated_dir / name).resolve()
        if not _is_relative_to(frame_path, run.annotated_dir):
            return None
        if not frame_path.is_file() or frame_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        return frame_path

    def state_payload(self, run_id: str | None) -> dict[str, Any]:
        runs = self.discover_runs()
        run = (
            runs[0]
            if runs and (not run_id or run_id == "latest")
            else next((item for item in runs if item.run_id == run_id), None)
        )
        if run is None:
            return {
                "ok": False,
                "message": "No scene frame run found yet.",
                "config_dirs": [str(config_dir) for config_dir in self.config_dirs],
                "out_roots": [str(root) for root in self.out_roots()],
                "runs": [self._run_summary(item) for item in runs],
            }

        step = _step_from_frame(run.latest_frame)
        layer_record = self._layer_record(run.metric_dir / "remote_scheduler_plan.jsonl", step)
        token_frame = self._token_frame(run.token_path, step, layer_record)
        mask = layer_record.get("layer_active_mask") if isinstance(layer_record, dict) else None
        mask = self._normalize_mask(mask)
        frame_name = run.latest_frame.name if run.latest_frame is not None else None
        frame_url = None
        if frame_name:
            query = urllib.parse.urlencode({"run": run.run_id, "name": frame_name})
            frame_url = f"/frame?{query}&t={int(run.latest_mtime)}"

        return {
            "ok": True,
            "time": time.time(),
            "selected_run": self._run_summary(run),
            "route_status": _read_result_status(run.result_path),
            "runs": [self._run_summary(item) for item in runs[:50]],
            "frame": {
                "step": step,
                "name": frame_name,
                "url": frame_url,
                "mtime": run.latest_mtime,
            },
            "token": token_frame,
            "layer": {
                "mask": mask,
                "active_count": sum(1 for value in mask if value),
                "source_step": layer_record.get("step") if isinstance(layer_record, dict) else None,
            },
        }

    def _run_summary(self, run: RunInfo) -> dict[str, Any]:
        return {
            "id": run.run_id,
            "route_id": run.route_id,
            "mode": run.mode,
            "out_root": str(run.out_root),
            "annotated_dir": str(run.annotated_dir),
            "metric_dir": str(run.metric_dir),
            "token_path": None if run.token_path is None else str(run.token_path),
            "result_path": None if run.result_path is None else str(run.result_path),
            "route_status": _read_result_status(run.result_path),
            "latest_frame": None if run.latest_frame is None else run.latest_frame.name,
            "latest_mtime": run.latest_mtime,
        }

    def _token_frame(
        self,
        token_path: Path | None,
        step: int | None,
        layer_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        if step is None:
            return None
        token_from_file = self._token_frame_from_file(token_path, step)
        if token_from_file is not None:
            return token_from_file
        return self._token_frame_from_layer_record(layer_record)

    def _token_frame_from_file(self, token_path: Path | None, step: int) -> dict[str, Any] | None:
        if token_path is None or not token_path.is_file():
            return None
        try:
            data = json.loads(token_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        frames = data.get("frames")
        if not isinstance(frames, list):
            return None
        selected = None
        for frame in frames:
            if isinstance(frame, dict) and frame.get("step") == step:
                selected = frame
                break
        if selected is None:
            earlier = [
                frame
                for frame in frames
                if isinstance(frame, dict)
                and isinstance(frame.get("step"), int)
                and frame.get("step") <= step
            ]
            selected = max(earlier, key=lambda frame: frame["step"], default=None)
        if selected is None:
            return None
        return {
            "source": "token_json",
            "estimated": False,
            "source_step": selected.get("step"),
            "original_visual_tokens": selected.get("original_visual_tokens"),
            "kept_visual_tokens": selected.get("kept_visual_tokens"),
            "pruned_visual_tokens": selected.get("pruned_visual_tokens"),
            "actual_keep_ratio": selected.get("actual_keep_ratio"),
            "actual_prune_ratio": selected.get("actual_prune_ratio"),
        }

    def _token_frame_from_layer_record(self, layer_record: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(layer_record, dict) or not layer_record:
            return None
        keep_ratio = self._mean_number(layer_record.get("visual_token_keep_ratio"))
        if keep_ratio is None:
            summary = layer_record.get("visual_token_prune_summary")
            if isinstance(summary, dict):
                ratio_info = summary.get("keep_ratio")
                if isinstance(ratio_info, dict):
                    keep_ratio = self._mean_number(ratio_info.get("first_values"))
                    if keep_ratio is None:
                        keep_ratio = ratio_info.get("min")
        if keep_ratio is None:
            return None
        original = int(self.original_visual_tokens)
        kept = int(round(original * float(keep_ratio)))
        pruned = original - kept
        return {
            "source": "remote_scheduler_plan_jsonl",
            "estimated": True,
            "source_step": layer_record.get("step"),
            "original_visual_tokens": original,
            "kept_visual_tokens": kept,
            "pruned_visual_tokens": pruned,
            "actual_keep_ratio": float(keep_ratio),
            "actual_prune_ratio": 1.0 - float(keep_ratio),
        }

    def _mean_number(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list):
            numbers: list[float] = []
            for item in value:
                if isinstance(item, (int, float)):
                    numbers.append(float(item))
            if numbers:
                return sum(numbers) / len(numbers)
        return None

    def _layer_record(self, jsonl_path: Path, step: int | None) -> dict[str, Any]:
        if step is None or not jsonl_path.is_file():
            return {}
        selected: dict[str, Any] = {}
        try:
            with jsonl_path.open("r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record_step = record.get("step")
                    if record_step == step:
                        return record
                    if isinstance(record_step, int) and record_step <= step:
                        if not selected or record_step > selected.get("step", -1):
                            selected = record
        except OSError:
            return {}
        return selected

    def _normalize_mask(self, raw_mask: Any) -> list[int]:
        if not isinstance(raw_mask, list):
            return [0] * self.layer_count
        values = [1 if bool(value) else 0 for value in raw_mask[: self.layer_count]]
        if len(values) < self.layer_count:
            values.extend([0] * (self.layer_count - len(values)))
        return values


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Eval Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #182230;
      --muted: #667085;
      --border: #d9e2ec;
      --blue: #8fd3ff;
      --blue-strong: #2898d8;
      --empty: #f8fafc;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    select, button {
      height: 36px;
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 6px;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    button.active { border-color: var(--blue-strong); color: var(--blue-strong); }
    main {
      display: grid;
      grid-template-columns: minmax(480px, 1fr) 380px;
      gap: 18px;
      padding: 18px 24px 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
    }
    .image-panel { min-height: calc(100vh - 112px); display: flex; flex-direction: column; }
    .image-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }
    .image-wrap {
      flex: 1;
      display: grid;
      place-items: center;
      padding: 14px;
      background: #eef3f8;
      border-radius: 0 0 8px 8px;
      min-height: 420px;
    }
    img {
      max-width: 100%;
      max-height: calc(100vh - 178px);
      border-radius: 6px;
      background: #d9e2ec;
      object-fit: contain;
    }
    .empty { color: var(--muted); text-align: center; line-height: 1.6; }
    .side { display: grid; gap: 14px; align-content: start; }
    .card { padding: 16px; }
    .card h2 { margin: 0 0 12px; font-size: 15px; }
    .kv { display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; font-size: 14px; }
    .kv span:nth-child(odd) { color: var(--muted); }
    .number { font-variant-numeric: tabular-nums; }
    .bar {
      height: 10px;
      background: #edf2f7;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
    }
    .bar-fill { height: 100%; background: linear-gradient(90deg, #35a8e0, #8fd3ff); width: 0%; }
    .layers {
      display: grid;
      grid-template-columns: repeat(24, minmax(0, 1fr));
      gap: 4px;
      margin-top: 10px;
    }
    .layer {
      height: 30px;
      border: 1px solid var(--border);
      background: var(--empty);
      border-radius: 4px;
      display: grid;
      place-items: center;
      color: #667085;
      font-size: 10px;
      font-variant-numeric: tabular-nums;
      min-width: 0;
    }
    .layer.on {
      background: var(--blue);
      border-color: #64bee8;
      color: #0b4f71;
    }
    .status { font-size: 13px; color: var(--muted); }
    .warn { color: var(--warn); }
    .route-banner {
      display: none;
      margin: 14px 14px 0;
      padding: 10px 12px;
      border: 1px solid #9bd9b1;
      background: #ecfdf3;
      color: #067647;
      border-radius: 6px;
      font-size: 14px;
    }
    .route-banner.running {
      display: block;
      border-color: var(--border);
      background: #f8fafc;
      color: var(--muted);
    }
    .route-banner.done { display: block; }
    .path {
      word-break: break-all;
      font-size: 12px;
      line-height: 1.5;
      color: var(--muted);
      margin-top: 10px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .image-panel { min-height: auto; }
      img { max-height: 70vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live Eval Viewer</h1>
    <div class="toolbar">
      <select id="runSelect"><option value="latest">latest</option></select>
      <button id="followBtn" class="active">Follow latest</button>
      <span id="status" class="status">connecting</span>
    </div>
  </header>
  <main>
    <section class="panel image-panel">
      <div class="image-head">
        <div>
          <div id="frameTitle" class="number">Frame: -</div>
          <div id="routeTitle" class="status">Route: -</div>
        </div>
        <div id="updatedAt" class="status">-</div>
      </div>
      <div id="routeBanner" class="route-banner">Route status: -</div>
      <div class="image-wrap">
        <img id="frameImage" alt="latest annotated frame" hidden />
        <div id="emptyState" class="empty">Waiting for scene frame images...</div>
      </div>
    </section>
    <aside class="side">
      <section class="panel card">
        <h2>Visual Tokens</h2>
        <div class="kv">
          <span>Original</span><strong id="tokOriginal" class="number">-</strong>
          <span>Kept after prune</span><strong id="tokKept" class="number">-</strong>
          <span>Pruned</span><strong id="tokPruned" class="number">-</strong>
          <span>Keep ratio</span><strong id="tokRatio" class="number">-</strong>
          <span>Source step</span><strong id="tokStep" class="number">-</strong>
          <span>Source</span><strong id="tokSource">-</strong>
        </div>
        <div class="bar"><div id="tokenBar" class="bar-fill"></div></div>
      </section>
      <section class="panel card">
        <h2>Activated Model Layers</h2>
        <div class="kv">
          <span>Mode</span><strong id="layerMode">-</strong>
          <span>Active layers</span><strong id="layerCount" class="number">- / 24</strong>
          <span>Source step</span><strong id="layerStep" class="number">-</strong>
        </div>
        <div id="layers" class="layers"></div>
      </section>
      <section class="panel card">
        <h2>Run Paths</h2>
        <div id="runPaths" class="path">-</div>
      </section>
    </aside>
  </main>
  <script>
    const runSelect = document.getElementById("runSelect");
    const followBtn = document.getElementById("followBtn");
    const statusEl = document.getElementById("status");
    const frameImage = document.getElementById("frameImage");
    const emptyState = document.getElementById("emptyState");
    const routeBanner = document.getElementById("routeBanner");
    let followLatest = true;
    let selectedRun = "latest";

    function fmt(value) {
      return value === null || value === undefined ? "-" : String(value);
    }

    function fmtRatio(value) {
      return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function renderRuns(runs, currentId) {
      const existing = runSelect.value;
      runSelect.innerHTML = "";
      const latest = document.createElement("option");
      latest.value = "latest";
      latest.textContent = "latest";
      runSelect.appendChild(latest);
      for (const run of runs) {
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = `${run.route_id} | ${run.latest_frame || "no frame"}`;
        option.title = run.id;
        runSelect.appendChild(option);
      }
      runSelect.value = followLatest ? "latest" : (currentId || existing || "latest");
    }

    function renderLayers(mask, sourceStep) {
      const container = document.getElementById("layers");
      container.innerHTML = "";
      const values = Array.isArray(mask) ? mask : Array(24).fill(0);
      values.slice(0, 24).forEach((value, idx) => {
        const div = document.createElement("div");
        div.className = `layer ${value ? "on" : ""}`;
        div.textContent = idx + 1;
        div.title = `layer ${idx + 1}: ${value ? "active" : "skipped"}`;
        container.appendChild(div);
      });
      const active = values.filter(Boolean).length;
      setText("layerCount", `${active} / 24`);
      setText("layerStep", fmt(sourceStep));
    }

    function renderRouteStatus(routeStatus) {
      const status = routeStatus || {};
      const progress = Array.isArray(status.progress) ? `progress ${status.progress.join(" / ")}` : "progress -";
      if (status.completed) {
        routeBanner.className = "route-banner done";
        routeBanner.textContent = `Route completed. Result status: ${fmt(status.status)} (${progress})`;
      } else {
        routeBanner.className = "route-banner running";
        routeBanner.textContent = `Route running. Result status: ${fmt(status.status)} (${progress})`;
      }
    }

    function renderState(data) {
      renderRuns(data.runs || [], data.selected_run && data.selected_run.id);
      if (!data.ok) {
        statusEl.textContent = data.message || "waiting";
        statusEl.className = "status warn";
        frameImage.hidden = true;
        emptyState.hidden = false;
        routeBanner.className = "route-banner";
        return;
      }
      statusEl.textContent = "live";
      statusEl.className = "status";
      const run = data.selected_run;
      renderRouteStatus(data.route_status || (run && run.route_status));
      setText("frameTitle", `Frame: ${fmt(data.frame.step)} (${fmt(data.frame.name)})`);
      setText("routeTitle", `Route: ${fmt(run.route_id)}`);
      setText("updatedAt", data.frame.mtime ? new Date(data.frame.mtime * 1000).toLocaleString() : "-");
      if (data.frame.url) {
        frameImage.src = data.frame.url;
        frameImage.hidden = false;
        emptyState.hidden = true;
      } else {
        frameImage.hidden = true;
        emptyState.hidden = false;
      }

      const token = data.token || {};
      setText("tokOriginal", fmt(token.original_visual_tokens));
      setText("tokKept", fmt(token.kept_visual_tokens));
      setText("tokPruned", fmt(token.pruned_visual_tokens));
      setText("tokRatio", fmtRatio(token.actual_keep_ratio));
      setText("tokStep", fmt(token.source_step));
      setText("tokSource", token.estimated ? "estimated live" : (token.source || "-"));
      const ratio = typeof token.actual_keep_ratio === "number" ? token.actual_keep_ratio : 0;
      document.getElementById("tokenBar").style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;

      renderLayers(data.layer && data.layer.mask, data.layer && data.layer.source_step);
      setText("layerMode", fmt(run.mode));
      document.getElementById("runPaths").innerHTML = [
        `out_root: ${fmt(run.out_root)}`,
        `frames: ${fmt(run.annotated_dir)}`,
        `metric: ${fmt(run.metric_dir)}`,
        `token: ${fmt(run.token_path)}`,
        `result: ${fmt(run.result_path)}`
      ].map(line => `<div>${line.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}</div>`).join("");
    }

    async function tick() {
      const run = followLatest ? "latest" : selectedRun;
      try {
        const response = await fetch(`/api/state?run=${encodeURIComponent(run)}`, { cache: "no-store" });
        renderState(await response.json());
      } catch (error) {
        statusEl.textContent = `disconnected: ${error}`;
        statusEl.className = "status warn";
      }
    }

    runSelect.addEventListener("change", () => {
      selectedRun = runSelect.value;
      followLatest = selectedRun === "latest";
      followBtn.classList.toggle("active", followLatest);
      tick();
    });
    followBtn.addEventListener("click", () => {
      followLatest = !followLatest;
      if (followLatest) {
        selectedRun = "latest";
        runSelect.value = "latest";
      }
      followBtn.classList.toggle("active", followLatest);
      tick();
    });

    tick();
    setInterval(tick, 1000);
  </script>
</body>
</html>
"""


def make_handler(state: LiveEvalState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/":
                _text_response(self, HTML_PAGE)
                return
            if parsed.path == "/api/runs":
                _json_response(self, {"runs": [state._run_summary(run) for run in state.discover_runs()]})
                return
            if parsed.path == "/api/state":
                run_id = query.get("run", ["latest"])[0]
                _json_response(self, state.state_payload(run_id))
                return
            if parsed.path == "/frame":
                run_id = query.get("run", ["latest"])[0]
                name = query.get("name", [""])[0]
                frame_path = state.safe_frame_path(run_id, name)
                if frame_path is None:
                    _json_response(self, {"ok": False, "message": "frame not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                try:
                    data = frame_path.read_bytes()
                except OSError:
                    _json_response(self, {"ok": False, "message": "failed to read frame"}, status=HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(str(frame_path))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            _text_response(self, "not found", status=HTTPStatus.NOT_FOUND, content_type="text/plain; charset=utf-8")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live web viewer for SimLingo evaluation outputs.")
    parser.add_argument(
        "--config-dir",
        dest="config_dirs",
        type=Path,
        action="append",
        default=None,
        help="Directory containing eval YAML files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--root",
        dest="roots",
        type=Path,
        action="append",
        default=[],
        help="Additional output root to scan. Can be passed multiple times.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8899, help="HTTP bind port.")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS, help="Number of model layers to render.")
    parser.add_argument(
        "--original-visual-tokens",
        type=int,
        default=DEFAULT_ORIGINAL_VISUAL_TOKENS,
        help="Fallback original visual token count used before *_token.json is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dirs = args.config_dirs if args.config_dirs is not None else DEFAULT_CONFIG_DIRS
    state = LiveEvalState(config_dirs, args.roots, args.layers, args.original_visual_tokens)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving SimLingo live eval viewer at http://{args.host}:{args.port}", flush=True)
    print("Config dirs:", flush=True)
    for config_dir in state.config_dirs:
        print(f"  - {config_dir}", flush=True)
    print("Discovered out_roots:", flush=True)
    for root in state.out_roots():
        print(f"  - {root}", flush=True)
    print(f"Fallback original visual tokens: {state.original_visual_tokens}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
