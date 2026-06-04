#!/usr/bin/env python3
"""Select key frame pairs for fixed-budget SimLingo comparisons.

This script compares two evaluation budgets on the same routes, aligns the
decision traces by step, scores each shared frame using:

- low-budget language degeneration relative to high budget
- control divergence between the two budgets
- low-budget aggressive / unstable control patterns
- low-budget progress lag relative to high budget

It then exports:

- a JSON summary with ranked candidates
- a Markdown report for easy manual inspection
- copied frame images for the best candidate on each route
- a side-by-side paired image for each selected route
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    Image = None
    ImageDraw = None
    ImageFont = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = (
    PROJECT_ROOT
    / "eval_results/intro_sample/simlingo/bench2drive/3/smart_assigner/no_checkpoint"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "eval_results/intro_sample/sample_to_intro/no_checkpoint_fixed_budget_compare"
)


@dataclass(frozen=True)
class TraceRecord:
    step: int
    timestamp: float
    vehicle_speed: float
    output_language: str
    planned_control: Dict[str, float]
    applied_control: Dict[str, float]
    image_path: Path
    source_image_path: Path


@dataclass(frozen=True)
class Candidate:
    route_id: str
    scenario_name: str
    status_low: str
    status_high: str
    score: float
    step: int
    timestamp: float
    progress_ratio: float
    low_text_badness: float
    high_text_badness: float
    text_gap: float
    action_gap: float
    speed_gap: float
    low_action_aggressiveness: float
    low_text: str
    high_text: str
    low_control: Dict[str, float]
    high_control: Dict[str, float]
    low_speed: float
    high_speed: float
    low_image_path: str
    high_image_path: str
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select key budget-comparison frames from SimLingo decision traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory that contains fixed-budget evaluation outputs.",
    )
    parser.add_argument(
        "--budget-low",
        type=str,
        default="stage1_fixed_0p4",
        help="Directory name for the lower budget run.",
    )
    parser.add_argument(
        "--budget-high",
        type=str,
        default="stage1_fixed_0p7",
        help="Directory name for the higher budget run.",
    )
    parser.add_argument(
        "--layer-selection-dir",
        type=str,
        default="layer_select_random",
        help="Subdirectory below each budget root that contains viz/res outputs.",
    )
    parser.add_argument(
        "--route-ids",
        nargs="+",
        default=["037", "070"],
        help="Route ids to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to write selected frame outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many ranked candidates to keep per route in the JSON report.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=100,
        help="Ignore very early decision frames before this step.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_trace_jsonl(
    base_dir: Path,
    budget_dir: str,
    layer_selection_dir: str,
    route_id: str,
) -> Path:
    viz_root = base_dir / budget_dir / layer_selection_dir / "viz" / route_id
    matches = sorted(viz_root.glob("**/decision_trace/decision_trace.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No decision trace found under {viz_root}")
    if len(matches) > 1:
        # Prefer the longest path last because it usually corresponds to the
        # actual trace inside the nested debug_viz run directory.
        return matches[-1]
    return matches[0]


def find_result_json(
    base_dir: Path,
    budget_dir: str,
    layer_selection_dir: str,
    route_id: str,
) -> Path:
    result_path = base_dir / budget_dir / layer_selection_dir / "res" / f"{route_id}_res.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result json: {result_path}")
    return result_path


def normalize_language(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "".join(str(item) for item in raw)
    return str(raw)


def load_trace_records(path: Path) -> Dict[int, TraceRecord]:
    records: Dict[int, TraceRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            step = int(payload["step"])
            image_path = Path(payload["image_path"])
            records[step] = TraceRecord(
                step=step,
                timestamp=float(payload["timestamp"]),
                vehicle_speed=float(payload.get("vehicle_speed", 0.0)),
                output_language=normalize_language(payload.get("output_language")),
                planned_control={
                    "steer": float(payload["planned_control"]["steer"]),
                    "throttle": float(payload["planned_control"]["throttle"]),
                    "brake": float(payload["planned_control"]["brake"]),
                },
                applied_control={
                    "steer": float(payload["applied_control"]["steer"]),
                    "throttle": float(payload["applied_control"]["throttle"]),
                    "brake": float(payload["applied_control"]["brake"]),
                },
                image_path=image_path,
                source_image_path=image_path,
            )
    if not records:
        raise ValueError(f"Decision trace is empty: {path}")
    return records


def extract_route_meta(result_json: Path) -> Dict[str, Any]:
    payload = load_json(result_json)
    records = payload["_checkpoint"]["records"]
    if len(records) != 1:
        raise ValueError(f"Expected exactly one route record in {result_json}, found {len(records)}")
    record = records[0]
    return {
        "route_id": record["route_id"],
        "scenario_name": record.get("scenario_name", "unknown"),
        "status": record.get("status", "unknown"),
        "score_route": record["scores"].get("score_route"),
        "score_composed": record["scores"].get("score_composed"),
        "budget": record.get("budget", {}),
    }


def trigram_repeat_ratio(text: str) -> float:
    if len(text) < 3:
        return 0.0
    grams = [text[idx : idx + 3] for idx in range(len(text) - 2)]
    return 1.0 - (len(set(grams)) / len(grams))


def unique_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(set(text)) / len(text)


def punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return punct / len(text)


def language_badness(text: str) -> float:
    text = text.strip()
    if not text:
        return 3.0

    badness = 0.0
    badness += max(0.0, trigram_repeat_ratio(text) - 0.55) * 5.0
    badness += max(0.0, 0.18 - unique_char_ratio(text)) * 12.0
    badness += max(0.0, punctuation_ratio(text) - 0.35) * 6.0
    badness += text.count("�") * 0.5
    badness += text.count("?") * 0.05
    badness += text.count("—") * 0.05
    if len(text) <= 2:
        badness += 1.5
    return badness


def action_aggressiveness(control: Dict[str, float]) -> float:
    steer_abs = abs(control["steer"])
    throttle = control["throttle"]
    brake = control["brake"]
    aggressiveness = 0.0
    aggressiveness += max(0.0, steer_abs - 0.6) * 2.0
    aggressiveness += max(0.0, steer_abs - 0.8) * 1.5
    if throttle > 0.5 and steer_abs > 0.6:
        aggressiveness += 1.5
    if brake > 0.5 and steer_abs > 0.5:
        aggressiveness += 0.5
    return aggressiveness


def action_gap(low: Dict[str, float], high: Dict[str, float]) -> float:
    return (
        abs(low["steer"] - high["steer"])
        + 1.2 * abs(low["throttle"] - high["throttle"])
        + 1.2 * abs(low["brake"] - high["brake"])
    )


def trim_text(text: str, limit: int = 200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_reason(
    text_gap: float,
    action_gap_value: float,
    speed_gap: float,
    low_control: Dict[str, float],
    high_control: Dict[str, float],
) -> str:
    parts: List[str] = []
    if text_gap >= 2.0:
        parts.append("0.4 budget language is much more degenerated than 0.7")
    elif text_gap >= 0.8:
        parts.append("0.4 budget language is noticeably less stable")

    if action_gap_value >= 2.0:
        parts.append(
            "0.4 budget takes a substantially different control action "
            f"(steer {low_control['steer']:.3f}/{high_control['steer']:.3f}, "
            f"throttle {low_control['throttle']:.3f}/{high_control['throttle']:.3f}, "
            f"brake {low_control['brake']:.3f}/{high_control['brake']:.3f})"
        )
    elif action_gap_value >= 1.0:
        parts.append("0.4 budget already deviates in control from 0.7")

    if speed_gap >= 2.0:
        parts.append(f"0.4 budget is lagging behind by {speed_gap:.1f} m/s")

    if not parts:
        parts.append("combined language and control gap is locally maximal")
    return "; ".join(parts)


def compare_route(
    route_id: str,
    low_meta: Dict[str, Any],
    high_meta: Dict[str, Any],
    low_records: Dict[int, TraceRecord],
    high_records: Dict[int, TraceRecord],
    top_k: int,
    min_step: int,
) -> Dict[str, Any]:
    common_steps = sorted(set(low_records) & set(high_records))
    if not common_steps:
        raise ValueError(f"No shared trace steps found for route {route_id}")

    max_common_step = max(common_steps)
    candidates: List[Candidate] = []

    for step in common_steps:
        if step < min_step:
            continue

        low = low_records[step]
        high = high_records[step]
        low_bad = language_badness(low.output_language)
        high_bad = language_badness(high.output_language)
        text_gap = max(0.0, low_bad - high_bad)
        act_gap = action_gap(low.planned_control, high.planned_control)
        speed_gap = max(0.0, high.vehicle_speed - low.vehicle_speed)
        low_aggr = action_aggressiveness(low.planned_control)
        progress_ratio = step / max_common_step

        score = (
            2.2 * text_gap
            + 1.9 * act_gap
            + 0.35 * speed_gap
            + 1.2 * low_aggr
            + 0.8 * progress_ratio
        )

        reason = build_reason(
            text_gap=text_gap,
            action_gap_value=act_gap,
            speed_gap=speed_gap,
            low_control=low.planned_control,
            high_control=high.planned_control,
        )
        candidates.append(
            Candidate(
                route_id=route_id,
                scenario_name=str(low_meta["scenario_name"]),
                status_low=str(low_meta["status"]),
                status_high=str(high_meta["status"]),
                score=score,
                step=step,
                timestamp=low.timestamp,
                progress_ratio=progress_ratio,
                low_text_badness=low_bad,
                high_text_badness=high_bad,
                text_gap=text_gap,
                action_gap=act_gap,
                speed_gap=speed_gap,
                low_action_aggressiveness=low_aggr,
                low_text=trim_text(low.output_language),
                high_text=trim_text(high.output_language),
                low_control=low.planned_control,
                high_control=high.planned_control,
                low_speed=low.vehicle_speed,
                high_speed=high.vehicle_speed,
                low_image_path=str(low.image_path),
                high_image_path=str(high.image_path),
                selection_reason=reason,
            )
        )

    if not candidates:
        raise ValueError(
            f"All shared steps for route {route_id} were filtered out. "
            f"Try lowering --min-step."
        )

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    return {
        "route_id": route_id,
        "scenario_name": low_meta["scenario_name"],
        "status_low": low_meta["status"],
        "status_high": high_meta["status"],
        "shared_step_count": len(common_steps),
        "best": best,
        "top_candidates": ranked[:top_k],
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def save_pair_image(
    low_src: Path,
    high_src: Path,
    out_path: Path,
    route_id: str,
    step: int,
    budget_low: str,
    budget_high: str,
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        return

    low_img = Image.open(low_src).convert("RGB")
    high_img = Image.open(high_src).convert("RGB")

    width = low_img.width + high_img.width
    header_height = 42
    height = max(low_img.height, high_img.height) + header_height

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((10, 8), f"{route_id} step {step} | left={budget_low} | right={budget_high}", fill=(0, 0, 0), font=font)
    draw.text((10, 24), "paired decision-trace frame", fill=(80, 80, 80), font=font)

    canvas.paste(low_img, (0, header_height))
    canvas.paste(high_img, (low_img.width, header_height))

    ensure_parent(out_path)
    canvas.save(out_path)


def serialize_candidate(candidate: Candidate) -> Dict[str, Any]:
    return asdict(candidate)


def render_report(
    report_path: Path,
    comparisons: Sequence[Dict[str, Any]],
    budget_low: str,
    budget_high: str,
) -> None:
    lines: List[str] = []
    lines.append("# Fixed Budget Key Frame Selection")
    lines.append("")
    lines.append(
        f"Compare `{budget_low}` vs `{budget_high}` under the same `layer_select_random` policy. "
        "The selector aligns decision traces by step and scores each shared frame using:"
    )
    lines.append("")
    lines.append("1. Low-budget language degeneration relative to the higher budget.")
    lines.append("2. Planned control divergence between the two budgets.")
    lines.append("3. Low-budget aggressive / unstable action patterns.")
    lines.append("4. Low-budget speed lag relative to the higher budget.")
    lines.append("")

    for comparison in comparisons:
        best: Candidate = comparison["best"]
        lines.append(f"## Route {comparison['route_id']} - {comparison['scenario_name']}")
        lines.append("")
        lines.append(
            f"- Outcome: `{budget_low}` = **{comparison['status_low']}**, "
            f"`{budget_high}` = **{comparison['status_high']}**"
        )
        lines.append(
            f"- Selected step: `{best.step}` (timestamp `{best.timestamp:.2f}` s, "
            f"shared-progress `{best.progress_ratio:.2%}`)"
        )
        lines.append(f"- Why this frame: {best.selection_reason}.")
        lines.append(
            f"- Planned action `{budget_low}`: steer `{best.low_control['steer']:.3f}`, "
            f"throttle `{best.low_control['throttle']:.3f}`, brake `{best.low_control['brake']:.3f}`"
        )
        lines.append(
            f"- Planned action `{budget_high}`: steer `{best.high_control['steer']:.3f}`, "
            f"throttle `{best.high_control['throttle']:.3f}`, brake `{best.high_control['brake']:.3f}`"
        )
        lines.append(
            f"- Speed `{budget_low}` / `{budget_high}`: "
            f"`{best.low_speed:.2f}` / `{best.high_speed:.2f}` m/s"
        )
        lines.append(
            f"- Language `{budget_low}`: `{best.low_text}`"
        )
        lines.append(
            f"- Language `{budget_high}`: `{best.high_text}`"
        )
        lines.append("")
        lines.append("Top ranked candidates:")
        lines.append("")
        lines.append("| rank | step | score | action_gap | text_gap | speed_gap |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for rank, candidate in enumerate(comparison["top_candidates"], start=1):
            lines.append(
                f"| {rank} | {candidate.step} | {candidate.score:.3f} | "
                f"{candidate.action_gap:.3f} | {candidate.text_gap:.3f} | {candidate.speed_gap:.3f} |"
            )
        lines.append("")

    ensure_parent(report_path)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: List[Dict[str, Any]] = []
    json_summary: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "budget_low": args.budget_low,
        "budget_high": args.budget_high,
        "layer_selection_dir": args.layer_selection_dir,
        "route_ids": args.route_ids,
        "min_step": args.min_step,
        "routes": [],
    }

    for route_id in args.route_ids:
        low_trace = find_trace_jsonl(base_dir, args.budget_low, args.layer_selection_dir, route_id)
        high_trace = find_trace_jsonl(base_dir, args.budget_high, args.layer_selection_dir, route_id)
        low_result = find_result_json(base_dir, args.budget_low, args.layer_selection_dir, route_id)
        high_result = find_result_json(base_dir, args.budget_high, args.layer_selection_dir, route_id)

        low_meta = extract_route_meta(low_result)
        high_meta = extract_route_meta(high_result)
        low_records = load_trace_records(low_trace)
        high_records = load_trace_records(high_trace)

        comparison = compare_route(
            route_id=route_id,
            low_meta=low_meta,
            high_meta=high_meta,
            low_records=low_records,
            high_records=high_records,
            top_k=args.top_k,
            min_step=args.min_step,
        )
        best: Candidate = comparison["best"]

        route_output_dir = output_dir / route_id
        route_output_dir.mkdir(parents=True, exist_ok=True)

        low_dst = route_output_dir / f"{route_id}_step{best.step:04d}_{args.budget_low}.png"
        high_dst = route_output_dir / f"{route_id}_step{best.step:04d}_{args.budget_high}.png"
        pair_dst = route_output_dir / f"{route_id}_step{best.step:04d}_pair.png"

        copy_image(Path(best.low_image_path), low_dst)
        copy_image(Path(best.high_image_path), high_dst)
        save_pair_image(
            low_src=Path(best.low_image_path),
            high_src=Path(best.high_image_path),
            out_path=pair_dst,
            route_id=route_id,
            step=best.step,
            budget_low=args.budget_low,
            budget_high=args.budget_high,
        )

        comparison_payload = {
            "route_id": route_id,
            "scenario_name": comparison["scenario_name"],
            "status_low": comparison["status_low"],
            "status_high": comparison["status_high"],
            "shared_step_count": comparison["shared_step_count"],
            "best": serialize_candidate(best),
            "exported_files": {
                "low_image": str(low_dst),
                "high_image": str(high_dst),
                "pair_image": str(pair_dst) if pair_dst.exists() else None,
            },
            "top_candidates": [serialize_candidate(item) for item in comparison["top_candidates"]],
        }

        route_json_path = route_output_dir / f"{route_id}_candidates.json"
        route_json_path.write_text(
            json.dumps(comparison_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        comparisons.append(comparison)
        json_summary["routes"].append(comparison_payload)

    summary_json_path = output_dir / "selection_summary.json"
    summary_json_path.write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / "selection_report.md"
    render_report(
        report_path=report_path,
        comparisons=comparisons,
        budget_low=args.budget_low,
        budget_high=args.budget_high,
    )

    print(f"Saved selection summary to: {summary_json_path}")
    print(f"Saved report to: {report_path}")
    for comparison in comparisons:
        best: Candidate = comparison["best"]
        print(
            f"Route {comparison['route_id']}: step {best.step} "
            f"(score={best.score:.3f})"
        )


if __name__ == "__main__":
    main()
