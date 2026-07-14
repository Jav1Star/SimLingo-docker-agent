#!/usr/bin/env python3
"""Summarize TFLOPs / latency comparisons across routes from two result directories."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        description="Summarize route-level TFLOPs and CUDA-event latency for two eval result directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--res-dir-a",
        type=str,
        required=True,
        help="First result directory. Accepts either the run directory or its res/ subdirectory.",
    )
    parser.add_argument(
        "--res-dir-b",
        type=str,
        required=True,
        help="Second result directory. Accepts either the run directory or its res/ subdirectory.",
    )
    parser.add_argument(
        "--route-ids",
        nargs="*",
        default=None,
        help="Optional route ids to summarize. If omitted, use the intersection of completed result files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional summary JSON output path. Default writes under the common parent of the two run directories.",
    )
    return parser.parse_args()


def normalize_result_dir(path: Path) -> Path:
    # 允许用户直接传 run 根目录，脚本内部统一落到真正的 res 目录。
    if path.is_dir() and (path / "res").is_dir():
        return path / "res"
    return path


def resolve_res_dir(value: str) -> Path:
    return normalize_result_dir(resolve_path(value))


def infer_run_meta(res_dir: Path) -> Dict[str, Any]:
    run_dir = res_dir.parent if res_dir.name == "res" else res_dir
    budget_mode = run_dir.parent.name if run_dir.parent else None
    budget_label = run_dir.name

    meta: Dict[str, Any] = {
        "res_dir": str(res_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "budget_mode": budget_mode,
        "budget_label": budget_label,
    }

    match = re.match(
        r"^bud_(?P<fixed_budget>\d+(?:\.\d+)?)(?:_prune_ratio_(?P<prune_ratio>\d+(?:\.\d+)?))?$",
        budget_label,
    )
    if match:
        meta["fixed_budget"] = float(match.group("fixed_budget"))
        prune_ratio = match.group("prune_ratio")
        meta["token_prune_ratio"] = None if prune_ratio is None else float(prune_ratio)

    if budget_mode in {"fixed", "random", "rule_based", "smart_assigner"}:
        seed_dir = run_dir.parent.parent
        benchmark_dir = seed_dir.parent if seed_dir else None
        agent_dir = benchmark_dir.parent if benchmark_dir else None
        meta["seed"] = seed_dir.name if seed_dir else None
        meta["benchmark"] = benchmark_dir.name if benchmark_dir else None
        meta["agent"] = agent_dir.name if agent_dir else None

    return meta


def normalize_route_ids(route_ids: Optional[List[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    out: List[str] = []
    for item in route_ids:
        text = str(item).strip()
        if not text:
            continue
        out.append(text.zfill(3))
    return out or None


def list_route_ids_from_res_dir(res_dir: Path) -> List[str]:
    route_ids = set()
    for result_path in res_dir.glob("*_res.json"):
        route_ids.add(result_path.stem.replace("_res", "").zfill(3))
    return sorted(route_ids)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_global_record(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = result_payload.get("_checkpoint", {})
    if not isinstance(checkpoint, dict):
        return {}
    global_record = checkpoint.get("global_record", {})
    return global_record if isinstance(global_record, dict) else {}


def load_records(result_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoint = result_payload.get("_checkpoint", {})
    if not isinstance(checkpoint, dict):
        return []
    records = checkpoint.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def extract_status(global_record: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[str]:
    status = global_record.get("status")
    if status is not None:
        return str(status)
    if records:
        item = records[0].get("status")
        if item is not None:
            return str(item)
    return None


def extract_score_composed(global_record: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[float]:
    scores_mean = global_record.get("scores_mean")
    if isinstance(scores_mean, dict) and scores_mean.get("score_composed") is not None:
        return float(scores_mean["score_composed"])
    if records:
        scores = records[0].get("scores")
        if isinstance(scores, dict) and scores.get("score_composed") is not None:
            return float(scores["score_composed"])
    return None


def extract_meta(global_record: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = global_record.get("meta")
    if isinstance(meta, dict):
        return meta
    if records:
        first_meta = records[0].get("meta")
        if isinstance(first_meta, dict):
            return first_meta
    return {}


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_run(res_dir: Path, route_ids: List[str]) -> Dict[str, Any]:
    per_route: List[Dict[str, Any]] = []
    total_flops = 0
    total_steps = 0
    latency_values: List[float] = []
    scores: List[float] = []
    duration_systems: List[float] = []
    duration_games: List[float] = []
    completed_routes: List[str] = []
    missing_routes: List[str] = []

    for route_id in route_ids:
        tflops_path = res_dir / f"{route_id}_tflops.json"
        result_path = res_dir / f"{route_id}_res.json"
        if not tflops_path.exists() or not result_path.exists():
            missing_routes.append(route_id)
            continue

        tflops_payload = load_json(tflops_path)
        result_payload = load_json(result_path)
        if not isinstance(tflops_payload, dict) or not isinstance(result_payload, dict):
            missing_routes.append(route_id)
            continue

        global_record = load_global_record(result_payload)
        records = load_records(result_payload)
        status = extract_status(global_record, records)
        if status == "Completed":
            completed_routes.append(route_id)

        route_flops = int(tflops_payload.get("total_inference_flops", 0) or 0)
        route_steps = int(tflops_payload.get("compute_steps", tflops_payload.get("inference_profiled_steps", 0)) or 0)
        route_latency_values = [float(x) for x in (tflops_payload.get("forward_latency_ms_values") or [])]
        route_avg_latency = safe_float(tflops_payload.get("avg_forward_latency_ms"))
        route_p50_latency = safe_float(tflops_payload.get("p50_forward_latency_ms"))
        route_p90_latency = safe_float(tflops_payload.get("p90_forward_latency_ms"))
        route_max_latency = safe_float(tflops_payload.get("max_forward_latency_ms"))
        score_composed = extract_score_composed(global_record, records)
        meta = extract_meta(global_record, records)
        duration_system = safe_float(meta.get("duration_system"))
        duration_game = safe_float(meta.get("duration_game"))

        total_flops += route_flops
        total_steps += route_steps
        latency_values.extend(route_latency_values)
        if score_composed is not None:
            scores.append(score_composed)
        if duration_system is not None:
            duration_systems.append(duration_system)
        if duration_game is not None:
            duration_games.append(duration_game)

        per_route.append(
            {
                "route_id": route_id,
                "status": status,
                "compute_steps": route_steps,
                "total_inference_tflops": route_flops / 1e12,
                "avg_tflops_per_step": None if route_steps <= 0 else (route_flops / route_steps) / 1e12,
                "latency_samples": len(route_latency_values),
                "avg_forward_latency_ms": route_avg_latency,
                "p50_forward_latency_ms": route_p50_latency,
                "p90_forward_latency_ms": route_p90_latency,
                "max_forward_latency_ms": route_max_latency,
                "score_composed": score_composed,
                "duration_system_s": duration_system,
                "duration_game_s": duration_game,
            }
        )

    return {
        "res_dir": str(res_dir),
        "requested_routes": route_ids,
        "completed_routes": completed_routes,
        "missing_routes": missing_routes,
        "route_count": len(route_ids),
        "completed_count": len(completed_routes),
        "total_compute_steps": total_steps,
        "total_inference_tflops": total_flops / 1e12,
        "overall_avg_tflops_per_step": None if total_steps <= 0 else (total_flops / total_steps) / 1e12,
        "latency_samples": len(latency_values),
        "overall_avg_forward_latency_ms": mean(latency_values) if latency_values else None,
        "p50_forward_latency_ms": percentile(latency_values, 50),
        "p90_forward_latency_ms": percentile(latency_values, 90),
        "max_forward_latency_ms": max(latency_values) if latency_values else None,
        "mean_score_composed": mean(scores) if scores else None,
        "mean_duration_system_s": mean(duration_systems) if duration_systems else None,
        "mean_duration_game_s": mean(duration_games) if duration_games else None,
        "per_route": per_route,
    }


def compute_delta(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    metrics = {
        "avg_tflops_per_step": "overall_avg_tflops_per_step",
        "avg_forward_latency_ms": "overall_avg_forward_latency_ms",
        "mean_score_composed": "mean_score_composed",
    }
    delta: Dict[str, Dict[str, Optional[float]]] = {}
    for out_key, run_key in metrics.items():
        a_val = safe_float(run_a.get(run_key))
        b_val = safe_float(run_b.get(run_key))
        delta[out_key] = {
            "a": a_val,
            "b": b_val,
            "delta": None if a_val is None or b_val is None else (b_val - a_val),
        }
    return delta


def build_default_output_path(res_dir_a: Path, res_dir_b: Path, route_ids: List[str]) -> Path:
    run_dir_a = res_dir_a.parent if res_dir_a.name == "res" else res_dir_a
    run_dir_b = res_dir_b.parent if res_dir_b.name == "res" else res_dir_b
    common_parent = Path(os.path.commonpath([str(run_dir_a.parent), str(run_dir_b.parent)]))
    file_name = f"budget_latency_compare_{run_dir_a.name}_vs_{run_dir_b.name}_routes{len(route_ids)}.json"
    return common_parent / file_name


def main() -> None:
    args = parse_args()

    res_dir_a = resolve_res_dir(args.res_dir_a)
    res_dir_b = resolve_res_dir(args.res_dir_b)
    if not res_dir_a.exists():
        raise FileNotFoundError(f"Result directory not found for run-a: {res_dir_a}")
    if not res_dir_b.exists():
        raise FileNotFoundError(f"Result directory not found for run-b: {res_dir_b}")

    route_ids = normalize_route_ids(args.route_ids)
    if route_ids is None:
        route_ids = sorted(set(list_route_ids_from_res_dir(res_dir_a)) & set(list_route_ids_from_res_dir(res_dir_b)))
        if not route_ids:
            raise ValueError("No shared route ids found between the two result directories")

    summary_a = summarize_run(res_dir_a, route_ids)
    summary_b = summarize_run(res_dir_b, route_ids)
    meta_a = infer_run_meta(res_dir_a)
    meta_b = infer_run_meta(res_dir_b)
    output_path = resolve_path(args.output) if args.output else build_default_output_path(res_dir_a, res_dir_b, route_ids)

    summary_payload = {
        "route_ids": route_ids,
        "run_a_meta": meta_a,
        "run_b_meta": meta_b,
        "run_a": summary_a,
        "run_b": summary_b,
        "delta_b_minus_a": compute_delta(summary_a, summary_b),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[budget-latency-compare] route_count={len(route_ids)}")
    print(f"[budget-latency-compare] run_a {meta_a.get('budget_label', res_dir_a.name)} res={res_dir_a}")
    print(f"[budget-latency-compare] run_b {meta_b.get('budget_label', res_dir_b.name)} res={res_dir_b}")
    print(f"[budget-latency-compare] output={output_path}")


if __name__ == "__main__":
    main()
