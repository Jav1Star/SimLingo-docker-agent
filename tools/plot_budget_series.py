#!/usr/bin/env python3
"""Plot budget-trace curves from budget_series.csv."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

def smooth_data(y, sigma=0.0):
    if not HAS_SCIPY or sigma <= 0:
        return y
    return gaussian_filter1d(y, sigma=sigma)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "Bench2Drive_smart_assigner_stage2_budget_trace"
    / "budget_trace_analysis"
    / "budget_series.csv"
)


@dataclass(frozen=True)
class SeriesRecord:
    step: int
    timestamp: float
    budget: float
    phase: str
    vehicle_speed: float
    route_id: str
    scenario_name: str
    town_name: str
    trace_path: str


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._") or "untitled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot budget curves.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--x-axis", type=str, default="step", choices=["step", "timestamp"])
    parser.add_argument("--route-id", type=str, default="")
    parser.add_argument("--start-step", type=int, default=None, help="Inclusive start step to plot.")
    parser.add_argument("--end-step", type=int, default=None, help="Inclusive end step to plot.")
    parser.add_argument("--plot-speed", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=2.0, help="Sigma for Gaussian smoothing.")
    parser.add_argument("--highlight-start-step", type=int, default=None, help="Inclusive start step for blue highlight.")
    parser.add_argument("--highlight-end-step", type=int, default=None, help="Inclusive end step for blue highlight.")
    return parser.parse_args()


def load_records(csv_path: Path, route_id: str = "") -> List[SeriesRecord]:
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("budget"): continue
            if route_id and row.get("route_id") != route_id: continue
            
            records.append(SeriesRecord(
                step=int(row["step"]),
                timestamp=float(row["timestamp"]),
                budget=float(row["budget"]),
                phase=row.get("phase", ""),
                vehicle_speed=float(row.get("vehicle_speed") or 0.0),
                route_id=row.get("route_id", ""),
                scenario_name=row.get("scenario_name", ""),
                town_name=row.get("town_name", ""),
                trace_path=row.get("trace_path", "")
            ))
    return records


def group_by_trace(records: Iterable[SeriesRecord]) -> Dict[str, List[SeriesRecord]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r.trace_path].append(r)
    for k in grouped:
        grouped[k].sort(key=lambda item: (item.step, item.timestamp))
    return dict(grouped)


def phase_color_map(phases: Iterable[str]) -> Dict[str, str]:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return {phase: palette[i % len(palette)] for i, phase in enumerate(sorted(set(phases)))}


def build_step_mask(
    records: Sequence[SeriesRecord],
    highlight_start_step: Optional[int],
    highlight_end_step: Optional[int],
) -> List[bool]:
    if highlight_start_step is None and highlight_end_step is None:
        return [True] * len(records)

    start = highlight_start_step if highlight_start_step is not None else -10**12
    end = highlight_end_step if highlight_end_step is not None else 10**12
    if start > end:
        start, end = end, start
    return [start <= record.step <= end for record in records]


def filter_records_by_step(
    records: Sequence[SeriesRecord],
    start_step: Optional[int],
    end_step: Optional[int],
) -> List[SeriesRecord]:
    if start_step is None and end_step is None:
        return list(records)

    start = start_step if start_step is not None else -10**12
    end = end_step if end_step is not None else 10**12
    if start > end:
        start, end = end, start
    return [record for record in records if start <= record.step <= end]


def compute_highlight_budget_mean(
    records: Sequence[SeriesRecord],
    highlight_mask: Sequence[bool],
) -> Optional[float]:
    highlighted_budgets = [record.budget for record, keep in zip(records, highlight_mask) if keep]
    if not highlighted_budgets:
        return None
    return float(np.mean(highlighted_budgets))


def add_colored_line(
    ax: plt.Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    highlight_mask: Sequence[bool],
) -> None:
    if len(x_values) == 0:
        return
    if len(x_values) == 1:
        color = "#1f77b4" if highlight_mask[0] else "#b8bec9"
        ax.plot(x_values, y_values, color=color, linewidth=2.4, label="budget")
        return

    points = np.column_stack((x_values, y_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_colors = []
    for left_highlight, right_highlight in zip(highlight_mask[:-1], highlight_mask[1:]):
        segment_colors.append("#1f77b4" if (left_highlight or right_highlight) else "#b8bec9")

    collection = LineCollection(
        segments,
        colors=segment_colors,
        linewidths=2.4,
        alpha=0.95,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_collection(collection)
    ax.plot([], [], color="#1f77b4", linewidth=2.4, label="budget")


def plot_single_trace(
    records: List[SeriesRecord],
    output_path: Path,
    x_axis: str,
    plot_speed: bool,
    sigma: float,
    highlight_start_step: Optional[int],
    highlight_end_step: Optional[int],
):
    if not records:
        return None

    x_values = [getattr(r, x_axis) for r in records]
    budgets = smooth_data([r.budget for r in records], sigma)
    speeds = smooth_data([r.vehicle_speed for r in records], sigma) if plot_speed else []
    colors = phase_color_map(r.phase for r in records)
    highlight_mask = build_step_mask(records, highlight_start_step, highlight_end_step)
    highlight_budget_mean = compute_highlight_budget_mean(records, highlight_mask)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    add_colored_line(ax, x_values, budgets, highlight_mask)

    for phase, color in colors.items():
        if phase == "smart_budget_stage2_explore":
            continue
        xs = [getattr(r, x_axis) for r in records if r.phase == phase]
        ys = smooth_data([r.budget for r in records if r.phase == phase], sigma)
        ax.scatter(xs, ys, s=18, color=color, alpha=0.85, label=phase)

    ax.set_xlabel(x_axis)
    ax.set_ylabel("budget")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    
    title = f"{records[0].route_id} | {Path(records[0].trace_path).parent.parent.name}"
    if highlight_budget_mean is not None and (
        highlight_start_step is not None or highlight_end_step is not None
    ):
        title += f" | highlight mean={highlight_budget_mean:.4f}"
    ax.set_title(title)

    if plot_speed:
        ax2 = ax.twinx()
        ax2.plot(x_values, speeds, color="#d62728", linestyle="--", label="speed")
        ax2.set_ylabel("speed (m/s)")
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    else:
        ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", transparent=True)
    plt.close(fig)
    return highlight_budget_mean


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_csv.parent / "plots"
    
    records = load_records(input_csv, args.route_id.strip())
    traces = group_by_trace(records)

    for trace_path, trace_records in traces.items():
        filtered_records = filter_records_by_step(
            trace_records,
            args.start_step,
            args.end_step,
        )
        if not filtered_records:
            print(
                f"Skip {trace_records[0].route_id}: no records in "
                f"steps=[{args.start_step},{args.end_step}]"
            )
            continue

        t_name = sanitize_filename(Path(trace_path).parent.parent.name)
        r_name = sanitize_filename(filtered_records[0].route_id)
        out_path = output_dir / f"trace_plots/{r_name}__{t_name}.svg"
        
        highlight_budget_mean = plot_single_trace(
            filtered_records,
            out_path,
            args.x_axis,
            args.plot_speed,
            args.smooth_sigma,
            args.highlight_start_step,
            args.highlight_end_step,
        )
        if highlight_budget_mean is not None and (
            args.highlight_start_step is not None or args.highlight_end_step is not None
        ):
            print(
                f"{filtered_records[0].route_id} "
                f"highlight_mean_budget={highlight_budget_mean:.6f} "
                f"steps=[{args.highlight_start_step},{args.highlight_end_step}]"
            )
        
    print(f"Processed {len(traces)} traces. Saved to {output_dir}")

if __name__ == "__main__":
    main()
