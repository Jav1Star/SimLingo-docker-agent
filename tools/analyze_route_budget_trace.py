#!/usr/bin/env python3
"""Analyze budget trends and change events from SimLingo decision traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "decision_trace_budget_analysis.yaml"


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_relative = PROJECT_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze budget trends and budget-change events from decision_trace.jsonl files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        help="YAML config path.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Optional override for analysis.root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override for analysis.output_dir.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def trim_text(text: Any, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def normalize_language(raw: Any, limit: int) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        raw = " ".join(str(item) for item in raw)
    return trim_text(raw, limit=limit)


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
        value = safe_float(eval_budget.get(key))
        if value is not None:
            return value
    return None


def extract_phase(record: Dict[str, Any]) -> Optional[str]:
    eval_budget = record.get("eval_budget", {})
    if not isinstance(eval_budget, dict):
        return None
    phase = eval_budget.get("phase")
    return None if phase is None else str(phase)


def extract_path_mask_hard(record: Dict[str, Any]) -> Optional[List[int]]:
    eval_budget = record.get("eval_budget", {})
    if not isinstance(eval_budget, dict):
        return None
    path_mask_hard = eval_budget.get("path_mask_hard")
    if not isinstance(path_mask_hard, list):
        return None
    values: List[int] = []
    for item in path_mask_hard:
        try:
            values.append(int(item))
        except Exception:
            return None
    return values


def summarize_route_points(points: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(points, list) or len(points) == 0:
        return None

    valid_points: List[List[float]] = []
    for item in points:
        if not isinstance(item, list) or len(item) < 2:
            continue
        x = safe_float(item[0])
        y = safe_float(item[1])
        if x is None or y is None:
            continue
        valid_points.append([x, y])

    if not valid_points:
        return None

    path_length = 0.0
    for prev, curr in zip(valid_points[:-1], valid_points[1:]):
        path_length += math.dist(prev, curr)

    end_point = valid_points[-1]
    return {
        "num_points": len(valid_points),
        "end_point": [round(end_point[0], 4), round(end_point[1], 4)],
        "path_length": round(path_length, 4),
        "max_abs_lateral": round(max(abs(point[1]) for point in valid_points), 4),
    }


def infer_result_json(trace_path: Path) -> Optional[Path]:
    parts = list(trace_path.parts)
    if "viz" not in parts:
        return None
    viz_idx = parts.index("viz")
    if viz_idx + 1 >= len(parts):
        return None
    route_id = parts[viz_idx + 1]
    prefix = Path(*parts[:viz_idx])
    candidate = prefix / "res" / f"{route_id}_res.json"
    return candidate if candidate.exists() else None


def load_route_meta(trace_path: Path, metadata_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if metadata_path.exists():
        payload = load_json(metadata_path)
        if isinstance(payload, dict):
            meta.update(payload)

    result_path = infer_result_json(trace_path)
    if result_path is not None:
        result_payload = load_json(result_path)
        checkpoint = result_payload.get("_checkpoint", {}) if isinstance(result_payload, dict) else {}
        records = checkpoint.get("records", []) if isinstance(checkpoint, dict) else []
        if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict):
            record = records[0]
            meta.setdefault("route_id", record.get("route_id", ""))
            meta.setdefault("scenario_name", record.get("scenario_name", ""))
            meta.setdefault("status", record.get("status", ""))
            meta.setdefault("score_route", record.get("scores", {}).get("score_route"))
            meta.setdefault("score_composed", record.get("scores", {}).get("score_composed"))

    meta.setdefault("route_id", "")
    meta.setdefault("route_name", "")
    meta.setdefault("route_file", "")
    meta.setdefault("route_key", "")
    meta.setdefault("scenario_name", "")
    meta.setdefault("town_name", "")
    meta.setdefault("status", "")
    return meta


def summarize_budget_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "start": None,
            "end": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "start": values[0],
        "end": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def build_trace_summary(
    trace_path: Path,
    records: List[Dict[str, Any]],
    route_meta: Dict[str, Any],
    change_epsilon: float,
    prompt_char_limit: int,
    language_char_limit: int,
    max_change_events_per_route: int,
) -> Dict[str, Any]:
    budget_series: List[Dict[str, Any]] = []
    change_events: List[Dict[str, Any]] = []
    numeric_budgets: List[float] = []

    prev_budget_record: Optional[Dict[str, Any]] = None
    num_increases = 0
    num_decreases = 0

    for record in records:
        budget_value = extract_budget(record)
        phase = extract_phase(record)
        series_row = {
            "step": int(record.get("step", 0)),
            "timestamp": safe_float(record.get("timestamp")),
            "budget": budget_value,
            "phase": phase,
            "path_mask_hard": extract_path_mask_hard(record),
            "vehicle_speed": safe_float(record.get("vehicle_speed")),
            "route_id": route_meta.get("route_id", ""),
            "scenario_name": route_meta.get("scenario_name", ""),
            "town_name": route_meta.get("town_name", ""),
            "trace_path": str(trace_path),
        }
        budget_series.append(series_row)
        if budget_value is not None:
            numeric_budgets.append(budget_value)

        if prev_budget_record is None or budget_value is None:
            if budget_value is not None:
                prev_budget_record = record
            continue

        prev_budget_value = extract_budget(prev_budget_record)
        if prev_budget_value is None:
            prev_budget_record = record
            continue

        delta = budget_value - prev_budget_value
        if abs(delta) <= change_epsilon:
            prev_budget_record = record
            continue

        if delta > 0:
            num_increases += 1
            direction = "increase"
        else:
            num_decreases += 1
            direction = "decrease"

        if len(change_events) < max_change_events_per_route:
            change_events.append(
                {
                    "route_id": route_meta.get("route_id", ""),
                    "route_name": route_meta.get("route_name", ""),
                    "scenario_name": route_meta.get("scenario_name", ""),
                    "town_name": route_meta.get("town_name", ""),
                    "step_before": int(prev_budget_record.get("step", 0)),
                    "step_after": int(record.get("step", 0)),
                    "timestamp_before": safe_float(prev_budget_record.get("timestamp")),
                    "timestamp_after": safe_float(record.get("timestamp")),
                    "budget_before": prev_budget_value,
                    "budget_after": budget_value,
                    "budget_delta": delta,
                    "direction": direction,
                    "phase_before": extract_phase(prev_budget_record),
                    "phase_after": phase,
                    "path_mask_hard_before": extract_path_mask_hard(prev_budget_record),
                    "path_mask_hard_after": extract_path_mask_hard(record),
                    "prompt_before": trim_text(prev_budget_record.get("prompt", ""), prompt_char_limit),
                    "prompt_after": trim_text(record.get("prompt", ""), prompt_char_limit),
                    "language_before": normalize_language(
                        prev_budget_record.get("output_language"),
                        limit=language_char_limit,
                    ),
                    "language_after": normalize_language(
                        record.get("output_language"),
                        limit=language_char_limit,
                    ),
                    "vehicle_speed_before": safe_float(prev_budget_record.get("vehicle_speed")),
                    "vehicle_speed_after": safe_float(record.get("vehicle_speed")),
                    "planned_control_before": prev_budget_record.get("planned_control"),
                    "planned_control_after": record.get("planned_control"),
                    "applied_control_before": prev_budget_record.get("applied_control"),
                    "applied_control_after": record.get("applied_control"),
                    "pred_route_before": summarize_route_points(prev_budget_record.get("pred_route")),
                    "pred_route_after": summarize_route_points(record.get("pred_route")),
                    "image_before": prev_budget_record.get("image_path"),
                    "image_after": record.get("image_path"),
                    "trace_path": str(trace_path),
                }
            )

        prev_budget_record = record

    budget_summary = summarize_budget_values(numeric_budgets)
    return {
        "trace_path": str(trace_path),
        "metadata": route_meta,
        "num_records": len(records),
        "budget_summary": {
            **budget_summary,
            "num_changes": num_increases + num_decreases,
            "num_increases": num_increases,
            "num_decreases": num_decreases,
        },
        "budget_series": budget_series,
        "budget_change_events": change_events,
    }


def flatten_series(route_summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in route_summaries:
        rows.extend(summary["budget_series"])
    return rows


def flatten_change_events(route_summaries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in route_summaries:
        rows.extend(summary["budget_change_events"])
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(route_summaries: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Budget Trace Analysis", ""]

    if not route_summaries:
        lines.append("No decision trace files found.")
        return "\n".join(lines) + "\n"

    for summary in route_summaries:
        meta = summary["metadata"]
        budget = summary["budget_summary"]
        title_parts = [
            meta.get("route_id") or "unknown_route",
            meta.get("scenario_name") or "unknown_scenario",
        ]
        if meta.get("town_name"):
            title_parts.append(str(meta["town_name"]))
        lines.append(f"## {' | '.join(title_parts)}")
        lines.append("")
        lines.append(f"- trace: `{summary['trace_path']}`")
        lines.append(f"- records: {summary['num_records']}")
        lines.append(
            "- budget trend: "
            f"start={budget['start']} end={budget['end']} min={budget['min']} "
            f"max={budget['max']} mean={budget['mean']}"
        )
        lines.append(
            "- budget changes: "
            f"{budget['num_changes']} total "
            f"(increase={budget['num_increases']}, decrease={budget['num_decreases']})"
        )
        if meta.get("status"):
            lines.append(f"- route status: {meta['status']}")

        events = summary["budget_change_events"]
        if not events:
            lines.append("- change events: none")
            lines.append("")
            continue

        lines.append("- change events:")
        for idx, event in enumerate(events, start=1):
            lines.append(
                f"  {idx}. step {event['step_before']} -> {event['step_after']}, "
                f"{event['budget_before']} -> {event['budget_after']} "
                f"({event['direction']}, delta={event['budget_delta']:.6f})"
            )
            lines.append(
                f"     language: `{event['language_before']}` -> `{event['language_after']}`"
            )
            lines.append(
                "     control: "
                f"{event['planned_control_before']} -> {event['planned_control_after']}, "
                f"speed {event['vehicle_speed_before']} -> {event['vehicle_speed_after']}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    analysis_cfg = cfg.get("analysis", cfg)
    if not isinstance(analysis_cfg, dict):
        raise ValueError("Config must contain an 'analysis' mapping or be a flat mapping")

    root_value = args.root or analysis_cfg.get("root")
    if not root_value:
        raise ValueError("Missing analysis.root")
    output_dir_value = args.output_dir or analysis_cfg.get("output_dir")
    if not output_dir_value:
        raise ValueError("Missing analysis.output_dir")

    root = resolve_path(str(root_value))
    output_dir = resolve_path(str(output_dir_value))
    trace_glob = str(analysis_cfg.get("trace_glob", "**/decision_trace/decision_trace.jsonl"))
    route_ids_filter = {str(x) for x in analysis_cfg.get("route_ids", []) if str(x).strip()}
    change_epsilon = float(analysis_cfg.get("change_epsilon", 1e-6))
    prompt_char_limit = int(analysis_cfg.get("prompt_char_limit", 160))
    language_char_limit = int(analysis_cfg.get("language_char_limit", 160))
    max_change_events_per_route = int(analysis_cfg.get("max_change_events_per_route", 20))

    trace_paths = sorted(root.glob(trace_glob))
    if not trace_paths:
        raise FileNotFoundError(f"No decision trace files found under {root} with glob {trace_glob}")

    route_summaries: List[Dict[str, Any]] = []
    for trace_path in trace_paths:
        metadata_path = trace_path.with_name("metadata.json")
        route_meta = load_route_meta(trace_path=trace_path, metadata_path=metadata_path)
        route_id = str(route_meta.get("route_id", "")).strip()
        if route_ids_filter and route_id not in route_ids_filter:
            continue
        records = load_jsonl(trace_path)
        if not records:
            continue
        route_summaries.append(
            build_trace_summary(
                trace_path=trace_path,
                records=records,
                route_meta=route_meta,
                change_epsilon=change_epsilon,
                prompt_char_limit=prompt_char_limit,
                language_char_limit=language_char_limit,
                max_change_events_per_route=max_change_events_per_route,
            )
        )

    route_summaries.sort(
        key=lambda item: (
            str(item["metadata"].get("route_id", "")),
            str(item["metadata"].get("scenario_name", "")),
            str(item["trace_path"]),
        )
    )
    series_rows = flatten_series(route_summaries)
    change_rows = flatten_change_events(route_summaries)

    summary_payload = {
        "root": str(root),
        "output_dir": str(output_dir),
        "trace_glob": trace_glob,
        "num_traces": len(route_summaries),
        "num_change_events": len(change_rows),
        "route_summaries": route_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "budget_trace_summary.json", summary_payload)
    write_csv(output_dir / "budget_series.csv", series_rows)
    write_csv(output_dir / "budget_change_events.csv", change_rows)
    with (output_dir / "budget_trace_report.md").open("w", encoding="utf-8") as handle:
        handle.write(build_markdown(route_summaries))

    print(f"[budget_trace] root={root}")
    print(f"[budget_trace] traces={len(route_summaries)} change_events={len(change_rows)}")
    print(f"[budget_trace] output_dir={output_dir}")


if __name__ == "__main__":
    main()
