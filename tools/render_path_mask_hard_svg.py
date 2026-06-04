#!/usr/bin/env python3
"""Render a path_mask_hard sequence into a transparent SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render path_mask_hard as a row of rounded squares in a transparent SVG. "
            "Use either --path-mask-json directly, or provide a decision_trace.jsonl input."
        )
    )
    parser.add_argument("--path-mask-json", type=str, default="", help='Direct mask input, e.g. "[0,1,1,0]"')
    parser.add_argument("--input-jsonl", type=str, default="", help="Path to decision_trace.jsonl")
    parser.add_argument("--step", type=int, default=None, help="Exact step to render from decision_trace.jsonl")
    parser.add_argument("--start-step", type=int, default=None, help="Inclusive start step for budget matching")
    parser.add_argument("--end-step", type=int, default=None, help="Inclusive end step for budget matching")
    parser.add_argument("--target-budget", type=float, default=None, help="Target budget for budget matching")
    parser.add_argument("--output-svg", type=str, required=True, help="Output SVG path")
    parser.add_argument("--cell-size", type=float, default=32.0, help="Square size in px")
    parser.add_argument("--gap", type=float, default=4.0, help="Gap between squares in px")
    parser.add_argument("--padding", type=float, default=6.0, help="Outer padding in px")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected dict row in {path}:{line_no}")
            rows.append(payload)
    return rows


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_budget(record: Dict[str, Any]) -> Optional[float]:
    eval_budget = record.get("eval_budget", {})
    if not isinstance(eval_budget, dict):
        return None
    for key in ("used_budget", "value", "fixed_budget"):
        budget = safe_float(eval_budget.get(key))
        if budget is not None:
            return budget
    return None


def extract_path_mask_hard(record: Dict[str, Any]) -> Optional[List[int]]:
    eval_budget = record.get("eval_budget", {})
    if not isinstance(eval_budget, dict):
        return None
    value = eval_budget.get("path_mask_hard")
    return normalize_path_mask(value)


def normalize_path_mask(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or not value:
        return None
    normalized: List[int] = []
    for item in value:
        try:
            parsed = int(item)
        except Exception:
            return None
        if parsed not in (0, 1):
            return None
        normalized.append(parsed)
    return normalized


def select_record_by_step(records: Sequence[Dict[str, Any]], step: int) -> Dict[str, Any]:
    matches = [record for record in records if int(record.get("step", -1)) == step]
    if not matches:
        raise ValueError(f"No record found for step {step}.")
    return matches[0]


def select_record_by_budget(
    records: Sequence[Dict[str, Any]],
    start_step: int,
    end_step: int,
    target_budget: float,
) -> Dict[str, Any]:
    if start_step > end_step:
        start_step, end_step = end_step, start_step

    midpoint = (start_step + end_step) / 2.0
    candidates: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for record in records:
        step = int(record.get("step", -1))
        if step < start_step or step > end_step:
            continue
        budget = extract_budget(record)
        if budget is None:
            continue
        candidates.append((abs(budget - target_budget), abs(step - midpoint), step, record))

    if not candidates:
        raise ValueError(
            f"No valid record found in step range [{start_step}, {end_step}] "
            f"for target budget {target_budget}."
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def resolve_path_mask(args: argparse.Namespace) -> Tuple[List[int], Dict[str, Any]]:
    if args.path_mask_json:
        parsed = json.loads(args.path_mask_json)
        mask = normalize_path_mask(parsed)
        if mask is None:
            raise ValueError("--path-mask-json must be a non-empty list containing only 0/1.")
        return mask, {"source": "path_mask_json"}

    if not args.input_jsonl:
        raise ValueError("Provide either --path-mask-json or --input-jsonl.")

    input_jsonl = Path(args.input_jsonl).expanduser()
    records = load_jsonl(input_jsonl)

    if args.step is not None:
        record = select_record_by_step(records, args.step)
    else:
        if args.start_step is None or args.end_step is None or args.target_budget is None:
            raise ValueError(
                "When using --input-jsonl without --step, you must provide "
                "--start-step, --end-step, and --target-budget."
            )
        record = select_record_by_budget(
            records=records,
            start_step=args.start_step,
            end_step=args.end_step,
            target_budget=args.target_budget,
        )

    mask = extract_path_mask_hard(record)
    if mask is None:
        raise ValueError("Selected record does not contain a valid path_mask_hard.")

    metadata = {
        "source": str(input_jsonl),
        "step": int(record.get("step", -1)),
        "timestamp": safe_float(record.get("timestamp")),
        "route_id": record.get("route_id", ""),
        "budget": extract_budget(record),
    }
    return mask, metadata


def build_svg(mask: Sequence[int], cell_size: float, gap: float, padding: float) -> str:
    width = padding * 2 + len(mask) * cell_size + max(0, len(mask) - 1) * gap
    height = padding * 2 + cell_size
    radius = cell_size * 0.10
    stroke_width = max(2.0, cell_size * 0.085)
    dashed_stroke = "#6F86C6"
    filled_stroke = "#6F86C6"
    filled_fill = "#EAF2FF"
    dash_on = round(cell_size * 0.24, 2)
    dash_off = round(cell_size * 0.18, 2)

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.2f}" height="{height:.2f}" '
            f'viewBox="0 0 {width:.2f} {height:.2f}" fill="none">'
        ),
        '  <g shape-rendering="geometricPrecision">',
    ]

    for index, value in enumerate(mask):
        x = padding + index * (cell_size + gap)
        y = padding
        if value == 0:
            svg_lines.append(
                "    "
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" '
                f'rx="{radius:.2f}" fill="none" stroke="{dashed_stroke}" '
                f'stroke-width="{stroke_width:.2f}" stroke-dasharray="{dash_on} {dash_off}"/>'
            )
        else:
            svg_lines.append(
                "    "
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" '
                f'rx="{radius:.2f}" fill="{filled_fill}" stroke="{filled_stroke}" '
                f'stroke-width="{stroke_width:.2f}"/>'
            )

    svg_lines.append("  </g>")
    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def main() -> None:
    args = parse_args()
    mask, metadata = resolve_path_mask(args)
    svg_text = build_svg(
        mask=mask,
        cell_size=args.cell_size,
        gap=args.gap,
        padding=args.padding,
    )

    output_svg = Path(args.output_svg).expanduser()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(svg_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_svg": str(output_svg),
                "path_mask_hard": mask,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
