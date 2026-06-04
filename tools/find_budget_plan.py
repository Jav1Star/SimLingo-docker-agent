#!/usr/bin/env python3
"""Find the decision trace record closest to a target budget within a step range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a decision_trace.jsonl file for the record whose budget is closest "
            "to the target budget within the given step range, then export its plan."
        )
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to decision_trace.jsonl")
    parser.add_argument("--start-step", type=int, required=True, help="Inclusive start step")
    parser.add_argument("--end-step", type=int, required=True, help="Inclusive end step")
    parser.add_argument("--target-budget", type=float, required=True, help="Target budget value")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output path for the selected record summary",
    )
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


def select_best_record(
    records: List[Dict[str, Any]],
    start_step: int,
    end_step: int,
    target_budget: float,
) -> Tuple[Dict[str, Any], float]:
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

        budget_gap = abs(budget - target_budget)
        midpoint_gap = abs(step - midpoint)
        candidates.append((budget_gap, midpoint_gap, step, record))

    if not candidates:
        raise ValueError(
            f"No valid records found in step range [{start_step}, {end_step}] "
            f"for target budget {target_budget}."
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_budget_gap, _, _, best_record = candidates[0]
    return best_record, best_budget_gap


def build_output_payload(
    input_jsonl: Path,
    start_step: int,
    end_step: int,
    target_budget: float,
    record: Dict[str, Any],
    budget_gap: float,
) -> Dict[str, Any]:
    actual_budget = extract_budget(record)
    eval_budget = record.get("eval_budget", {})
    output_language = record.get("output_language", [])
    if not isinstance(output_language, list):
        output_language = [str(output_language)]

    return {
        "input_jsonl": str(input_jsonl),
        "query": {
            "start_step": start_step,
            "end_step": end_step,
            "target_budget": target_budget,
        },
        "match": {
            "step": int(record.get("step", -1)),
            "timestamp": safe_float(record.get("timestamp")),
            "budget": actual_budget,
            "budget_gap": budget_gap,
            "phase": eval_budget.get("phase") if isinstance(eval_budget, dict) else None,
            "route_id": record.get("route_id", ""),
            "scenario_name": record.get("scenario_name", ""),
            "town_name": record.get("town_name", ""),
            "image_path": record.get("image_path"),
        },
        "execution_plan": {
            "path_mask_hard": extract_path_mask_hard(record),
            "prompt": record.get("prompt", ""),
            "output_language": output_language,
            "planned_control": record.get("planned_control"),
            "applied_control": record.get("applied_control"),
            "pred_route": record.get("pred_route"),
            "pred_speed_wps": record.get("pred_speed_wps"),
            "vehicle_speed": safe_float(record.get("vehicle_speed")),
        },
        "budget_decision": eval_budget,
    }


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser()

    records = load_jsonl(input_jsonl)
    best_record, budget_gap = select_best_record(
        records=records,
        start_step=args.start_step,
        end_step=args.end_step,
        target_budget=args.target_budget,
    )
    payload = build_output_payload(
        input_jsonl=input_jsonl,
        start_step=args.start_step,
        end_step=args.end_step,
        target_budget=args.target_budget,
        record=best_record,
        budget_gap=budget_gap,
    )

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
