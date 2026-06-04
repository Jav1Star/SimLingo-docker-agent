#!/usr/bin/env python3
"""Summarize SimLingo `language_samples.json` statistics.

The script recursively scans a Bench2Drive SimLingo result tree, loads every
`language_samples.json`, and computes mean/min/max for:

- raw character length of the `language` field
- raw character length of the `prompt` field
- SimLingo tokenizer length of the `language` field
- SimLingo tokenizer length of the `prompt` field

Token counts are computed with `pretrained/InternVL2-1B` by default, using
`add_special_tokens=False` so the counts match the raw tokenizer span lengths.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval_results/Bench2Drive/simlingo/bench2drive/3/simlingo_compat/viz"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "pretrained/InternVL2-1B"


@dataclass(frozen=True)
class SampleRecord:
    file_path: str
    sample_index: int
    step: Any
    language: str
    prompt: str
    language_char_len: int
    prompt_char_len: int
    language_token_len: int
    prompt_token_len: int


def resolve_resource_path(value: str) -> str:
    """Resolve a local path relative to the repo when possible.

    If the path does not exist locally, return the original string so remote
    Hugging Face model ids still work.
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    repo_relative = PROJECT_ROOT / path
    if repo_relative.exists():
        return str(repo_relative)

    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return str(cwd_relative)

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SimLingo language_samples.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT.relative_to(PROJECT_ROOT)),
        help="Root directory that contains the Bench2Drive SimLingo evaluation results.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH.relative_to(PROJECT_ROOT)),
        help="Tokenizer path or Hugging Face model id used to count tokens.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the aggregated summary as JSON.",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_path: str):
    resolved = resolve_resource_path(tokenizer_path)
    return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True, use_fast=False)


def token_length(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def summarize(values: List[int]) -> Dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def collect_records(root: Path, tokenizer) -> Dict[str, Any]:
    files = sorted(root.rglob("language_samples.json"))
    if not files:
        raise FileNotFoundError(f"No language_samples.json files found under: {root}")

    records: List[SampleRecord] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"Unexpected JSON structure in {file_path}: expected a samples list")

        if len(samples) != 3:
            print(
                f"Warning: {file_path} contains {len(samples)} samples (expected 3).",
                file=sys.stderr,
            )

        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"Unexpected sample type in {file_path} at index {sample_index}: {type(sample)!r}")

            if "language" not in sample:
                raise KeyError(f"Missing 'language' in {file_path} sample #{sample_index}")
            if "prompt" not in sample:
                raise KeyError(f"Missing 'prompt' in {file_path} sample #{sample_index}")

            language = str(sample["language"])
            prompt = str(sample["prompt"])
            step = sample.get("step")
            try:
                step_value = int(step) if step is not None else None
            except (TypeError, ValueError):
                step_value = step

            records.append(
                SampleRecord(
                    file_path=str(file_path),
                    sample_index=sample_index,
                    step=step_value,
                    language=language,
                    prompt=prompt,
                    language_char_len=len(language),
                    prompt_char_len=len(prompt),
                    language_token_len=token_length(tokenizer, language),
                    prompt_token_len=token_length(tokenizer, prompt),
                )
            )

    return {
        "files": [str(path) for path in files],
        "records": records,
    }


def build_summary(root: Path, tokenizer_path: str, records: List[SampleRecord], files: List[str]) -> Dict[str, Any]:
    language_char_lengths = [record.language_char_len for record in records]
    prompt_char_lengths = [record.prompt_char_len for record in records]
    language_token_lengths = [record.language_token_len for record in records]
    prompt_token_lengths = [record.prompt_token_len for record in records]

    return {
        "root": str(root),
        "tokenizer_path": tokenizer_path,
        "file_count": len(files),
        "sample_count": len(records),
        "character_length": {
            "language": summarize(language_char_lengths),
            "prompt": summarize(prompt_char_lengths),
        },
        "token_length": {
            "language": summarize(language_token_lengths),
            "prompt": summarize(prompt_token_lengths),
        },
        "records": [asdict(record) for record in records],
    }


def print_table(title: str, section: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"{'metric':<14} {'count':>8} {'mean':>12} {'min':>8} {'max':>8}")
    for metric_name in ("language", "prompt"):
        summary = section[metric_name]
        print(
            f"{metric_name:<14} {summary['count']:>8} {summary['mean']:>12.2f} {summary['min']:>8} {summary['max']:>8}"
        )


def main() -> None:
    args = parse_args()
    root = Path(resolve_resource_path(args.root))
    if not root.exists():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    collected = collect_records(root, tokenizer)
    records = collected["records"]
    files = collected["files"]

    summary = build_summary(root, args.tokenizer_path, records, files)

    print(f"Scanned {summary['file_count']} files and {summary['sample_count']} samples")
    print(f"Tokenizer: {summary['tokenizer_path']} ({type(tokenizer).__name__})")
    print_table("Character Lengths", summary["character_length"])
    print_table("Token Lengths", summary["token_length"])

    if args.output_json:
        output_path = Path(resolve_resource_path(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()