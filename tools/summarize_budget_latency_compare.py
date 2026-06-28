#!/usr/bin/env python3
"""Summarize fixed-budget TFLOPs / latency comparisons across routes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import yaml


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
        description="Summarize route-level TFLOPs and CUDA-event latency for two fixed-budget eval runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-a",
        type=str,
        required=True,
        help="Eval YAML for the first fixed-budget run.",
    )
    parser.add_argument(
        "--config-b",
        type=str,
        required=True,
        help="Eval YAML for the second fixed-budget run.",
    )
    parser.add_argument(
        "--res-dir-a",
        type=str,
        default="",
        help="Optional explicit result directory override for config-a.",
    )
    parser.add_argument(
        "--res-dir-b",
        type=str,
        default="",
        help="Optional explicit result directory override for config-b.",
    )
    parser.add_argument(
        "--route-ids",
        nargs="*",
        default=None,
        help="Optional route ids to summarize. If omitted, use the intersection of completed result files.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="Optional seed override. Default uses the first seed from config-a/config-b and requires them to match.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional summary JSON output path. Default writes under config-a out_root.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def normalize_eval_cfg(payload: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = payload.get("eval", payload)
    if not isinstance(eval_cfg, dict):
        raise ValueError("Config 'eval' section must be a mapping")
    budget_cfg = payload.get("budget", eval_cfg.get("budget", {})) or {}
    if not isinstance(budget_cfg, dict):
        raise ValueError("Config 'budget' section must be a mapping")

    seeds = eval_cfg.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Config must contain a non-empty eval.seeds list")

    fixed_budget = float(budget_cfg.get("fixed_budget", 1.0))
    return {
        "agent": str(eval_cfg["agent"]),
        "benchmark": str(eval_cfg["benchmark"]),
        "out_root": resolve_path(str(eval_cfg["out_root"])),
        "seed": str(seeds[0]),
        "fixed_budget": fixed_budget,
        "config_path": str(payload.get("__config_path__", "")),
    }


def budget_dir_name(fixed_budget: float) -> str:
    return f"bud_{fixed_budget:.3f}"


def build_res_dir(cfg: Dict[str, Any], seed_override: Optional[str]) -> Path:
    seed = str(seed_override) if seed_override else str(cfg["seed"])
    return (
        Path(cfg["out_root"])
        / cfg["agent"]
        / cfg["benchmark"]
        / seed
        / "fixed"
        / budget_dir_name(float(cfg["fixed_budget"]))
        / "res"
    )


def resolve_res_dir(override: str, cfg: Dict[str, Any], seed_override: Optional[str]) -> Path:
    if override:
        return resolve_path(override)
    return build_res_dir(cfg, seed_override=seed_override)


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


def compute_delta(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics = [
        "total_inference_tflops",
        "overall_avg_tflops_per_step",
        "overall_avg_forward_latency_ms",
        "p50_forward_latency_ms",
        "p90_forward_latency_ms",
        "max_forward_latency_ms",
        "mean_score_composed",
        "mean_duration_system_s",
        "mean_duration_game_s",
    ]
    delta: Dict[str, Optional[float]] = {}
    for key in metrics:
        a_val = safe_float(run_a.get(key))
        b_val = safe_float(run_b.get(key))
        delta[key] = None if a_val is None or b_val is None else (b_val - a_val)
    return delta


def build_default_output_path(cfg_a: Dict[str, Any], seed: str, route_ids: List[str]) -> Path:
    base_out = Path(cfg_a["out_root"])
    return base_out / f"budget_latency_compare_seed{seed}_routes{len(route_ids)}.json"


def main() -> None:
    args = parse_args()

    config_a_path = resolve_path(args.config_a)
    config_b_path = resolve_path(args.config_b)
    payload_a = load_yaml(config_a_path)
    payload_a["__config_path__"] = str(config_a_path)
    payload_b = load_yaml(config_b_path)
    payload_b["__config_path__"] = str(config_b_path)
    cfg_a = normalize_eval_cfg(payload_a)
    cfg_b = normalize_eval_cfg(payload_b)

    if cfg_a["agent"] != cfg_b["agent"] or cfg_a["benchmark"] != cfg_b["benchmark"]:
        raise ValueError("Both configs must target the same agent and benchmark")
    if (
        not args.res_dir_a
        and not args.res_dir_b
        and math.isclose(float(cfg_a["fixed_budget"]), float(cfg_b["fixed_budget"]), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            "The two configs resolve to the same fixed_budget. "
            "Please fix the YAMLs or pass --res-dir-a/--res-dir-b explicitly."
        )

    seed_override = args.seed.strip() if isinstance(args.seed, str) else ""
    if not seed_override and str(cfg_a["seed"]) != str(cfg_b["seed"]):
        raise ValueError("Configs use different default seeds. Pass --seed explicitly.")
    seed = seed_override or str(cfg_a["seed"])

    res_dir_a = resolve_res_dir(args.res_dir_a, cfg_a, seed_override=seed)
    res_dir_b = resolve_res_dir(args.res_dir_b, cfg_b, seed_override=seed)
    if not res_dir_a.exists():
        raise FileNotFoundError(f"Result directory not found for config-a: {res_dir_a}")
    if not res_dir_b.exists():
        raise FileNotFoundError(f"Result directory not found for config-b: {res_dir_b}")

    route_ids = normalize_route_ids(args.route_ids)
    if route_ids is None:
        route_ids = sorted(set(list_route_ids_from_res_dir(res_dir_a)) & set(list_route_ids_from_res_dir(res_dir_b)))
        if not route_ids:
            raise ValueError("No shared route ids found between the two result directories")

    summary_a = summarize_run(res_dir_a, route_ids)
    summary_b = summarize_run(res_dir_b, route_ids)
    output_path = resolve_path(args.output) if args.output else build_default_output_path(cfg_a, seed, route_ids)

    summary_payload = {
        "seed": seed,
        "route_ids": route_ids,
        "config_a": {
            "path": str(config_a_path),
            "fixed_budget": float(cfg_a["fixed_budget"]),
            "res_dir": str(res_dir_a),
        },
        "config_b": {
            "path": str(config_b_path),
            "fixed_budget": float(cfg_b["fixed_budget"]),
            "res_dir": str(res_dir_b),
        },
        "run_a": summary_a,
        "run_b": summary_b,
        "delta_b_minus_a": compute_delta(summary_a, summary_b),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[budget-latency-compare] seed={seed}")
    print(f"[budget-latency-compare] route_count={len(route_ids)}")
    print(f"[budget-latency-compare] config_a budget={cfg_a['fixed_budget']:.3f} res={res_dir_a}")
    print(f"[budget-latency-compare] config_b budget={cfg_b['fixed_budget']:.3f} res={res_dir_b}")
    print(f"[budget-latency-compare] output={output_path}")


if __name__ == "__main__":
    main()
