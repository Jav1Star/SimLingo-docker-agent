#!/usr/bin/env python3
"""Plot a route-level budget curve and transformer-layer activation panel."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplcfg_")

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_budget_series import add_colored_line, smooth_data


ACTIVE_EDGE = "#6F86C6"
ACTIVE_FILL = "#EAF2FF"
FIXED_FILL = "#1F4E8C"
FIXED_EDGE = "#1F4E8C"
INACTIVE_FILL = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one route metric_info.json and render a two-panel figure: "
            "budget curve on the left, transformer-layer activation over frames on the right."
        )
    )
    parser.add_argument("--route-viz-dir", type=str, required=True, help="Route viz directory, e.g. .../viz/055")
    parser.add_argument("--route-id", type=str, default="", help="Optional route id for sibling-dir fallback.")
    parser.add_argument("--result-json", type=str, default="", help="Optional result JSON to resolve save_name.")
    parser.add_argument("--output-svg", type=str, default="", help="Optional output SVG path.")
    parser.add_argument("--num-fixed-layers", type=int, default=2, help="How many leading layers to render as fixed dark-blue blocks.")
    parser.add_argument("--smooth-sigma", type=float, default=0.0, help="Optional Gaussian smoothing for budget curve.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_result_json(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str).expanduser().resolve()
    return path if path.exists() else None


def resolve_actual_route_dir(route_viz_dir: Path, route_id: str, result_json: Optional[Path]) -> Path:
    if route_viz_dir.joinpath("debug_viz").exists():
        return route_viz_dir

    if result_json is not None:
        payload = load_json(result_json)
        records = payload.get("_checkpoint", {}).get("records", []) if isinstance(payload, dict) else []
        if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict):
            save_name = str(records[0].get("save_name", "")).strip()
            if save_name:
                candidate = route_viz_dir.parent / save_name
                if candidate.exists():
                    return candidate

    route_prefix = route_id.strip() or route_viz_dir.name.strip()
    sibling_dirs = sorted(
        (
            sibling
            for sibling in route_viz_dir.parent.iterdir()
            if sibling.is_dir() and sibling.name.startswith(route_prefix) and sibling.name != route_viz_dir.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for sibling_dir in sibling_dirs:
        if sibling_dir.joinpath("debug_viz").exists():
            return sibling_dir

    return route_viz_dir


def find_latest_metric_file(route_dir: Path) -> Path:
    matches = sorted(
        (path for path in route_dir.glob("**/metric/metric_info.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No metric_info.json found under {route_dir}")
    return matches[-1]


def extract_budget(eval_budget: Dict[str, Any]) -> Optional[float]:
    for key in ("used_budget", "value", "fixed_budget"):
        value = eval_budget.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue

    for key in ("last_update", "last_decision"):
        rows = eval_budget.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            for inner_key in ("used_budget", "value"):
                value = rows[0].get(inner_key)
                if value is None:
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
    return None


def extract_path_mask_hard(eval_budget: Dict[str, Any]) -> Optional[List[int]]:
    value = eval_budget.get("path_mask_hard")
    if isinstance(value, list) and value:
        try:
            return [int(item) for item in value]
        except Exception:
            return None

    execution_plan = eval_budget.get("execution_plan")
    if isinstance(execution_plan, dict):
        rows = execution_plan.get("full_layer_mask_by_batch")
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            try:
                return [int(item) for item in rows[0]]
            except Exception:
                return None
        fallback_mask = extract_path_mask_from_flattened_plan(execution_plan)
        if fallback_mask is not None:
            return fallback_mask
    return None


def extract_path_mask_from_flattened_plan(execution_plan: Dict[str, Any]) -> Optional[List[int]]:
    flattened_by_batch = execution_plan.get("flattened_by_batch")
    controllable_shape = execution_plan.get("controllable_shape")
    num_prefix_layers = execution_plan.get("num_prefix_layers")
    if not isinstance(flattened_by_batch, list) or not flattened_by_batch:
        return None
    if not isinstance(controllable_shape, list) or len(controllable_shape) != 4:
        return None
    if num_prefix_layers is None:
        return None

    flat = flattened_by_batch[0]
    if not isinstance(flat, list):
        return None

    try:
        num_controllable_layers = int(controllable_shape[0])
        num_batch = int(controllable_shape[1])
        num_branches = int(controllable_shape[2])
        num_units = int(controllable_shape[3])
        num_prefix_layers = int(num_prefix_layers)
    except Exception:
        return None

    if num_batch != 1 or num_controllable_layers <= 0 or num_branches <= 0 or num_units <= 0:
        return None

    expected_len = num_controllable_layers * num_branches * num_units
    if len(flat) != expected_len:
        return None

    controllable_mask: List[int] = []
    layer_span = num_branches * num_units
    for layer_idx in range(num_controllable_layers):
        start = layer_idx * layer_span
        end = start + layer_span
        layer_values = flat[start:end]
        try:
            is_active = 1 if any(float(item) > 0.0 for item in layer_values) else 0
        except Exception:
            return None
        controllable_mask.append(is_active)

    return [1] * num_prefix_layers + controllable_mask


def route_label_from_metric_file(metric_file: Path, result_json: Optional[Path], route_dir: Path) -> str:
    if result_json is not None:
        payload = load_json(result_json)
        records = payload.get("_checkpoint", {}).get("records", []) if isinstance(payload, dict) else []
        if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict):
            route_id = str(records[0].get("route_id", "")).strip()
            scenario_name = str(records[0].get("scenario_name", "")).strip()
            if route_id and scenario_name:
                return f"{route_id} | {scenario_name}"
            if route_id:
                return route_id
    return route_dir.name or metric_file.parent.parent.name


def sorted_metric_rows(metric_info: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for key, value in metric_info.items():
        if not isinstance(value, dict):
            continue
        try:
            step = int(key)
        except Exception:
            continue
        rows.append((step, value))
    rows.sort(key=lambda item: item[0])
    return rows


def build_series(metric_info: Dict[str, Any]) -> Tuple[List[int], List[float], List[List[int]]]:
    steps: List[int] = []
    budgets: List[float] = []
    masks: List[List[int]] = []

    for step, frame_data in sorted_metric_rows(metric_info):
        eval_budget = frame_data.get("eval_budget", {})
        if not isinstance(eval_budget, dict):
            continue
        budget = extract_budget(eval_budget)
        path_mask = extract_path_mask_hard(eval_budget)
        if budget is None or path_mask is None:
            continue
        steps.append(step)
        budgets.append(float(budget))
        masks.append(path_mask)

    if not steps:
        raise ValueError("No frames with both budget and path_mask_hard were found.")
    return steps, budgets, masks


def make_mask_matrix(masks: Sequence[Sequence[int]]) -> List[List[int]]:
    num_layers = max(len(mask) for mask in masks)
    matrix: List[List[int]] = []
    for layer_idx in range(num_layers):
        matrix.append([
            int(mask[layer_idx]) if layer_idx < len(mask) else 0
            for mask in masks
        ])
    return matrix


def draw_activation_panel(
    ax: plt.Axes,
    steps: Sequence[int],
    masks: Sequence[Sequence[int]],
    num_fixed_layers: int,
) -> None:
    matrix = make_mask_matrix(masks)
    num_layers = len(matrix)
    num_frames = len(steps)

    ax.set_xlim(0, num_frames)
    ax.set_ylim(num_layers, 0)
    ax.set_facecolor("none")

    cell_w = 0.86
    cell_h = 0.72
    x_pad = (1.0 - cell_w) * 0.5
    y_pad = (1.0 - cell_h) * 0.5

    for frame_idx in range(num_frames):
        for layer_idx in range(num_layers):
            is_active = bool(matrix[layer_idx][frame_idx])
            is_fixed = layer_idx < num_fixed_layers

            facecolor = FIXED_FILL if is_fixed else (ACTIVE_FILL if is_active else INACTIVE_FILL)
            edgecolor = FIXED_EDGE if is_fixed else ACTIVE_EDGE
            linestyle = "solid" if (is_fixed or is_active) else (0, (2.0, 1.6))
            linewidth = 1.3 if is_fixed else 1.0

            patch = FancyBboxPatch(
                (frame_idx + x_pad, layer_idx + y_pad),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.0,rounding_size=0.08",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                mutation_aspect=1.0,
            )
            ax.add_patch(patch)

    if num_frames > 1:
        tick_count = min(6, num_frames)
        tick_positions = [int(round(i * (num_frames - 1) / max(tick_count - 1, 1))) for i in range(tick_count)]
        tick_labels = [str(int(steps[pos])) for pos in tick_positions]
        ax.set_xticks([pos + 0.5 for pos in tick_positions], tick_labels)
    else:
        ax.set_xticks([0.5], [str(int(steps[0]))])

    if num_layers >= 4:
        yticks = [0.5, min(1.5, num_layers - 0.5), num_layers - 0.5]
        ylabels = ["1", "2", str(num_layers)]
        ax.set_yticks(yticks, ylabels)
    else:
        ax.set_yticks([idx + 0.5 for idx in range(num_layers)], [str(idx + 1) for idx in range(num_layers)])

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Layer")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_panel(
    steps: Sequence[int],
    budgets: Sequence[float],
    masks: Sequence[Sequence[int]],
    output_svg: Path,
    route_label: str,
    num_fixed_layers: int,
    smooth_sigma: float,
) -> None:
    smoothed_budgets = smooth_data(list(budgets), smooth_sigma)
    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.35], wspace=0.12)

    ax_budget = fig.add_subplot(gs[0, 0])
    ax_budget.set_facecolor("none")
    add_colored_line(ax_budget, list(steps), list(smoothed_budgets), [True] * len(steps))
    ax_budget.set_ylim(0.0, 1.05)
    ax_budget.set_xlabel("Frame")
    ax_budget.set_ylabel("Budget")
    ax_budget.grid(True, alpha=0.25)
    ax_budget.legend(loc="best")

    ax_layers = fig.add_subplot(gs[0, 1])
    draw_activation_panel(ax_layers, steps, masks, num_fixed_layers=num_fixed_layers)

    fig.suptitle(route_label, y=0.98, fontsize=12)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    route_viz_dir = Path(args.route_viz_dir).expanduser().resolve()
    result_json = resolve_result_json(args.result_json)
    route_dir = resolve_actual_route_dir(route_viz_dir, args.route_id, result_json)
    metric_file = find_latest_metric_file(route_dir)

    metric_info = load_json(metric_file)
    if not isinstance(metric_info, dict):
        raise ValueError(f"{metric_file} does not contain a JSON object.")

    steps, budgets, masks = build_series(metric_info)
    route_label = route_label_from_metric_file(metric_file, result_json, route_dir)

    if args.output_svg:
        output_svg = Path(args.output_svg).expanduser().resolve()
    else:
        output_svg = metric_file.parent.parent / "budget_layer_panel.svg"

    plot_panel(
        steps=steps,
        budgets=budgets,
        masks=masks,
        output_svg=output_svg,
        route_label=route_label,
        num_fixed_layers=max(0, int(args.num_fixed_layers)),
        smooth_sigma=max(0.0, float(args.smooth_sigma)),
    )

    print(
        json.dumps(
            {
                "metric_file": str(metric_file),
                "output_svg": str(output_svg),
                "num_frames": len(steps),
                "num_layers": max(len(mask) for mask in masks),
                "num_fixed_layers": int(args.num_fixed_layers),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
