#!/usr/bin/env python3
"""Select 5 hardest and 5 easiest routes with closely matched route lengths.

The script reads per-route Bench2Drive result JSON files, ranks routes by
`score_composed`, and searches for the tightest route-length window that still
contains separable hard/easy subsets. This helps isolate scene difficulty from
route-length differences.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RES_DIR = PROJECT_ROOT / "eval_results/Bench2Drive/simlingo_compat/seed3/res"
DEFAULT_ROUTE_DIR = PROJECT_ROOT / "leaderboard/data/bench2drive_split"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "eval_results/Bench2Drive/simlingo_compat/seed3/selected_easy_hard_routes.json"


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    scenario_name: str
    town_name: str
    status: str
    num_infractions: int
    route_length: float
    score_composed: float
    score_route: float
    score_penalty: float
    result_file: str
    route_file: Optional[str]


@dataclass(frozen=True)
class SelectionCandidate:
    span: float
    score_gap: float
    score_spread: float
    window_records: Tuple[RouteRecord, ...]
    hardest: Tuple[RouteRecord, ...]
    easiest: Tuple[RouteRecord, ...]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    repo_relative = PROJECT_ROOT / path
    if repo_relative.exists():
        return repo_relative.resolve()

    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select matched-length hard/easy Bench2Drive routes from evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default=str(DEFAULT_RES_DIR.relative_to(PROJECT_ROOT)),
        help="Directory containing per-route `*_res.json` files.",
    )
    parser.add_argument(
        "--route-dir",
        type=str,
        default=str(DEFAULT_ROUTE_DIR.relative_to(PROJECT_ROOT)),
        help="Directory containing split route XML files such as `bench2drive_000.xml`.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many hard routes and easy routes to select.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON.relative_to(PROJECT_ROOT)),
        help="Path to save the selection result as JSON. Use an empty string to disable saving.",
    )
    parser.add_argument(
        "--allow-nonpositive-gap",
        action="store_true",
        help="Allow windows whose hard/easy score boundary has zero or negative margin.",
    )
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_route_file(route_dir: Path, result_path: Path) -> Optional[str]:
    stem = result_path.stem
    if not stem.endswith("_res"):
        return None

    try:
        route_index = int(stem[:-4])
    except ValueError:
        return None

    candidates = (
        route_dir / f"bench2drive_{route_index:03d}.xml",
        route_dir / f"bench2drive_{route_index:02d}.xml",
        route_dir / f"bench2drive_{route_index}.xml",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def load_route_records(res_dir: Path, route_dir: Path) -> List[RouteRecord]:
    if not res_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {res_dir}")

    records_by_id: Dict[str, RouteRecord] = {}
    duplicate_ids: List[str] = []

    for result_path in sorted(res_dir.glob("*.json")):
        if result_path.name == "merged.json":
            continue

        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        route_file = infer_route_file(route_dir, result_path)
        raw_records = payload.get("_checkpoint", {}).get("records", [])
        for raw_record in raw_records:
            route_id = str(raw_record.get("route_id", ""))
            if not route_id:
                continue

            meta = raw_record.get("meta", {})
            scores = raw_record.get("scores", {})

            record = RouteRecord(
                route_id=route_id,
                scenario_name=str(raw_record.get("scenario_name", "")),
                town_name=str(raw_record.get("town_name", "")),
                status=str(raw_record.get("status", "")),
                num_infractions=_to_int(raw_record.get("num_infractions"), default=0),
                route_length=_to_float(meta.get("route_length"), default=0.0),
                score_composed=_to_float(scores.get("score_composed"), default=0.0),
                score_route=_to_float(scores.get("score_route"), default=0.0),
                score_penalty=_to_float(scores.get("score_penalty"), default=0.0),
                result_file=str(result_path.resolve()),
                route_file=route_file,
            )

            if route_id in records_by_id:
                duplicate_ids.append(route_id)
            records_by_id[route_id] = record

    if duplicate_ids:
        duplicate_preview = ", ".join(sorted(set(duplicate_ids))[:10])
        print(
            "Warning: duplicate route ids found in results. Keeping the last record for each route. "
            f"Examples: {duplicate_preview}"
        )

    records = sorted(records_by_id.values(), key=lambda record: (record.route_length, record.route_id))
    if len(records) < 2:
        raise ValueError(f"Need at least 2 routes, but only found {len(records)} in {res_dir}")
    return records


def hard_sort_key(record: RouteRecord) -> Tuple[float, float, int, int, str]:
    success_flag = 1 if record.status in {"Completed", "Perfect"} else 0
    return (
        record.score_composed,
        record.score_route,
        success_flag,
        -record.num_infractions,
        record.route_id,
    )


def easy_sort_key(record: RouteRecord) -> Tuple[float, float, int, int, str]:
    success_flag = 1 if record.status in {"Completed", "Perfect"} else 0
    return (
        -record.score_composed,
        -record.score_route,
        -success_flag,
        record.num_infractions,
        record.route_id,
    )


def split_window(records: Sequence[RouteRecord], top_k: int) -> Tuple[List[RouteRecord], List[RouteRecord]]:
    hardest = sorted(records, key=hard_sort_key)[:top_k]
    easiest = sorted(records, key=easy_sort_key)[:top_k]
    return hardest, easiest


def build_candidate(window_records: Sequence[RouteRecord], top_k: int) -> Optional[SelectionCandidate]:
    hardest, easiest = split_window(window_records, top_k)
    hard_ids = {record.route_id for record in hardest}
    easy_ids = {record.route_id for record in easiest}
    if hard_ids & easy_ids:
        return None

    max_hard_score = max(record.score_composed for record in hardest)
    min_easy_score = min(record.score_composed for record in easiest)
    span = window_records[-1].route_length - window_records[0].route_length
    spread = max(record.score_composed for record in easiest) - min(record.score_composed for record in hardest)

    return SelectionCandidate(
        span=span,
        score_gap=min_easy_score - max_hard_score,
        score_spread=spread,
        window_records=tuple(window_records),
        hardest=tuple(hardest),
        easiest=tuple(easiest),
    )


def choose_best_candidate(
    records: Sequence[RouteRecord],
    top_k: int,
    allow_nonpositive_gap: bool,
) -> SelectionCandidate:
    window_size = top_k * 2
    if len(records) < window_size:
        raise ValueError(f"Need at least {window_size} routes, but only found {len(records)}")

    positive_gap_candidates: List[SelectionCandidate] = []
    fallback_candidates: List[SelectionCandidate] = []

    for start_index in range(len(records) - window_size + 1):
        window_records = records[start_index:start_index + window_size]
        candidate = build_candidate(window_records, top_k)
        if candidate is None:
            continue
        fallback_candidates.append(candidate)
        if candidate.score_gap > 0:
            positive_gap_candidates.append(candidate)

    if positive_gap_candidates:
        pool = positive_gap_candidates
    elif allow_nonpositive_gap and fallback_candidates:
        pool = fallback_candidates
    else:
        raise ValueError(
            "Could not find a matched-length window with disjoint hard/easy selections and a positive score gap. "
            "Try `--allow-nonpositive-gap` if you still want the tightest window."
        )

    return min(
        pool,
        key=lambda candidate: (
            candidate.span,
            -candidate.score_gap,
            -candidate.score_spread,
            min(record.route_id for record in candidate.window_records),
        ),
    )


def summarize_selection(candidate: SelectionCandidate) -> Dict[str, Any]:
    lengths = [record.route_length for record in candidate.window_records]
    scores = [record.score_composed for record in candidate.window_records]
    return {
        "window_route_count": len(candidate.window_records),
        "route_length_min": round(min(lengths), 6),
        "route_length_max": round(max(lengths), 6),
        "route_length_span": round(candidate.span, 6),
        "score_min": round(min(scores), 6),
        "score_max": round(max(scores), 6),
        "score_gap": round(candidate.score_gap, 6),
        "score_spread": round(candidate.score_spread, 6),
    }


def serialize_records(records: Iterable[RouteRecord]) -> List[Dict[str, Any]]:
    return [asdict(record) for record in records]


def print_group(title: str, records: Sequence[RouteRecord]) -> None:
    print(f"\n{title}")
    print(
        f"{'route_id':<24} {'score':>9} {'length':>9} {'status':<36} "
        f"{'infractions':>11} {'scenario'}"
    )
    for record in records:
        print(
            f"{record.route_id:<24} "
            f"{record.score_composed:>9.3f} "
            f"{record.route_length:>9.3f} "
            f"{record.status:<36} "
            f"{record.num_infractions:>11} "
            f"{record.scenario_name}"
        )


def main() -> None:
    args = parse_args()
    res_dir = resolve_path(args.res_dir)
    route_dir = resolve_path(args.route_dir)

    records = load_route_records(res_dir, route_dir)
    candidate = choose_best_candidate(
        records=records,
        top_k=args.top_k,
        allow_nonpositive_gap=args.allow_nonpositive_gap,
    )
    summary = summarize_selection(candidate)

    print(f"Loaded {len(records)} unique route records from: {res_dir}")
    print(f"Selected a matched-length window with {summary['window_route_count']} routes")
    print(
        "Route length range: "
        f"{summary['route_length_min']:.3f} -> {summary['route_length_max']:.3f} "
        f"(span {summary['route_length_span']:.3f})"
    )
    print(
        "Score range inside window: "
        f"{summary['score_min']:.3f} -> {summary['score_max']:.3f} "
        f"(hard/easy boundary gap {summary['score_gap']:.3f})"
    )

    print_group(f"Hardest {args.top_k} Routes", candidate.hardest)
    print_group(f"Easiest {args.top_k} Routes", candidate.easiest)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "res_dir": str(res_dir),
            "route_dir": str(route_dir),
            "top_k": args.top_k,
            "selection_strategy": {
                "window_size": args.top_k * 2,
                "goal": "Minimize route length span first, then maximize hard/easy score separation.",
                "allow_nonpositive_gap": args.allow_nonpositive_gap,
            },
            "summary": summary,
            "hardest": serialize_records(candidate.hardest),
            "easiest": serialize_records(candidate.easiest),
            "window_records": serialize_records(candidate.window_records),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved selection JSON to: {output_path}")


if __name__ == "__main__":
    main()
