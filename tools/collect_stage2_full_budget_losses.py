#!/usr/bin/env python3
"""
Collect per-frame full-budget stage2 driving losses (L_full) offline.

Highlights:
- Dual-GPU (or multi-GPU) collection via torchrun.
- Route-level partitioning across ranks to preserve route history continuity.
- Deterministic per-sample randomness using seed derived from
  (base_seed, measurement_path), so each frame maps to stable stochastic choices.
- Records all loss components, plus explicit driving decomposition:
  loss_wp_total, loss_lang, and loss_driving.

Example:
  torchrun --nproc_per_node=2 tools/collect_stage2_full_budget_losses.py \
      --experiment smart_assigner_stage2 \
      --seed 9876
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import AutoProcessor

# Importing config module registers structured config groups in Hydra ConfigStore.
import simlingo_adaption_training.config  # noqa: F401


@dataclass(frozen=True)
class SampleRecord:
    index: int
    measurement_path: str
    route_key: str
    frame_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-frame stage2 full-budget losses with multi-GPU route partitioning."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="smart_assigner_stage2",
        help="Hydra experiment name under simlingo_adaption_training/config/experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9876,
        help="Base deterministic seed. Default: 9876.",
    )
    parser.add_argument(
        "--full-budget",
        type=float,
        default=1.0,
        help="External fixed budget used for L_full collection.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=0,
        help="Per-rank fixed-slot route batch size. 0 means use cfg.data_module.batch_size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory for collected JSONL. "
            "Default: <repo>/<data_path>/train/stage2_full_budget_losses"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint override. If empty, uses cfg.checkpoint.",
    )
    parser.add_argument(
        "--hydra-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides, e.g. data_module.batch_size=4",
    )
    return parser.parse_args()


def _decode_path(value) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _route_key_from_measurement_path(measurement_path: str) -> str:
    parts = measurement_path.rsplit("/measurements/", 1)
    if len(parts) == 2:
        return parts[0]
    return os.path.dirname(measurement_path)


def _records_from_base_dataset(dataset, global_offset: int = 0) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for local_idx in range(len(dataset)):
        measurement_root = _decode_path(dataset.measurements[local_idx][0])
        frame_id = int(dataset.sample_start[local_idx]) + int(dataset.hist_len) - 1
        measurement_path = f"{measurement_root}/{frame_id:04}.json.gz"
        route_key = _route_key_from_measurement_path(measurement_path)
        records.append(
            SampleRecord(
                index=global_offset + local_idx,
                measurement_path=measurement_path,
                route_key=route_key,
                frame_id=frame_id,
            )
        )
    return records


def build_sample_records(dataset) -> List[SampleRecord]:
    if isinstance(dataset, ConcatDataset):
        records: List[SampleRecord] = []
        offset = 0
        for sub_dataset in dataset.datasets:
            records.extend(_records_from_base_dataset(sub_dataset, global_offset=offset))
            offset += len(sub_dataset)
        return records
    return _records_from_base_dataset(dataset, global_offset=0)


def split_route_keys_for_rank(route_keys: Sequence[str], rank: int, world_size: int) -> List[str]:
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    return [route_key for idx, route_key in enumerate(route_keys) if (idx % world_size) == rank]


def build_route_batches(records: Sequence[SampleRecord], route_batch_size: int) -> Iterable[List[SampleRecord]]:
    if route_batch_size <= 0:
        raise ValueError(f"route_batch_size must be >= 1, got {route_batch_size}")

    route_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
    for record in records:
        route_to_records[record.route_key].append(record)

    route_order = sorted(route_to_records.keys())
    for route_key in route_order:
        frames = sorted(route_to_records[route_key], key=lambda x: x.frame_id)
        frame_ids = [r.frame_id for r in frames]
        for i in range(1, len(frame_ids)):
            if frame_ids[i] <= frame_ids[i - 1]:
                raise RuntimeError(
                    f"Route frame order violation in {route_key}: {frame_ids[i - 1]} -> {frame_ids[i]}"
                )
        route_to_records[route_key] = deque(frames)

    pending_routes = deque(route_order)
    active_routes: List[str] = []
    for _ in range(min(route_batch_size, len(pending_routes))):
        active_routes.append(pending_routes.popleft())

    while active_routes:
        batch_records: List[SampleRecord] = []
        next_active_routes: List[str] = []
        for route_key in active_routes:
            batch_records.append(route_to_records[route_key].popleft())
            if route_to_records[route_key]:
                next_active_routes.append(route_key)
            elif pending_routes:
                next_active_routes.append(pending_routes.popleft())

        if len({r.route_key for r in batch_records}) != len(batch_records):
            raise RuntimeError("Batch contains duplicate routes, violating route-local continuity.")

        yield batch_records
        active_routes = next_active_routes


def to_repo_relative(path: str, repo_root: Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return str(path_obj.relative_to(repo_root.resolve()))
    except ValueError:
        return str(path_obj)


def resolve_autocast_settings(cfg_precision: str, device: torch.device) -> Tuple[bool, Optional[torch.dtype], str]:
    if device.type != "cuda":
        return False, None, "cpu-no-autocast"

    precision = str(cfg_precision).strip().lower()
    fp16_aliases = {"16", "16-mixed", "16-true", "fp16", "fp16-mixed", "fp16-true", "half"}
    bf16_aliases = {"bf16", "bf16-mixed", "bf16-true"}
    fp32_aliases = {"32", "32-true", "fp32", "full"}

    if precision in bf16_aliases:
        return True, torch.bfloat16, f"{precision}->bf16"
    if precision in fp16_aliases:
        return True, torch.float16, f"{precision}->fp16"
    if precision in fp32_aliases:
        return True, torch.float16, f"{precision}->forced-fp16-for-flash-attn"
    return True, torch.float16, f"unknown({precision})->fp16"


def per_sample_loss_scalars(loss_dict: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
    per_sample: Dict[str, torch.Tensor] = {}
    for key, (values, counts) in loss_dict.items():
        values = values.detach().float()
        counts = counts.detach().float()
        reduce_dims = tuple(range(1, values.dim()))
        if len(reduce_dims) > 0:
            values_sum = values.sum(dim=reduce_dims)
            counts_sum = counts.sum(dim=reduce_dims).clamp_min(1.0)
            per_sample[key] = values_sum / counts_sum
        else:
            per_sample[key] = values
    return per_sample


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(move_to_device(v, device) for v in x))
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def sample_seed_from_measurement_path(base_seed: int, measurement_path: str) -> int:
    payload = f"{int(base_seed)}::{measurement_path}".encode("utf-8")
    # Keep seed in uint32 range for numpy/random compatibility.
    return int(hashlib.sha1(payload).hexdigest()[:8], 16)


@contextlib.contextmanager
def fixed_sample_random_state(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def fetch_batch_samples(dataset, batch_records: Sequence[SampleRecord], base_seed: int):
    samples = []
    sample_seeds = []
    for record in batch_records:
        sample_seed = sample_seed_from_measurement_path(base_seed=base_seed, measurement_path=record.measurement_path)
        with fixed_sample_random_state(sample_seed):
            samples.append(dataset[record.index])
        sample_seeds.append(sample_seed)
    return samples, sample_seeds


def init_distributed() -> Tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed collection requires CUDA when WORLD_SIZE > 1")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
    return rank, world_size, local_rank, device


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def barrier_if_needed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def compose_cfg(experiment: str, extra_overrides: Sequence[str]):
    config_dir = PROJECT_ROOT / "simlingo_adaption_training" / "config"
    overrides = [f"experiment={experiment}"] + list(extra_overrides)
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


def resolve_checkpoint_path(repo_root: Path, checkpoint: Optional[str]) -> Optional[Path]:
    if checkpoint is None:
        return None
    checkpoint_str = str(checkpoint).strip()
    if checkpoint_str == "":
        return None
    checkpoint_path = Path(checkpoint_str)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    return checkpoint_path.resolve()


def default_output_dir(repo_root: Path, cfg) -> Path:
    data_path = str(cfg.data_module.base_dataset.data_path).strip().rstrip("/")
    if data_path == "":
        return (repo_root / "data" / "simlingo" / "train" / "stage2_full_budget_losses").resolve()
    data_path_obj = Path(data_path)
    if data_path_obj.is_absolute():
        return (data_path_obj / "train" / "stage2_full_budget_losses").resolve()
    return (repo_root / data_path_obj / "train" / "stage2_full_budget_losses").resolve()


def set_seed(seed: int, rank: int) -> int:
    effective_seed = int(seed) + int(rank)
    pl.seed_everything(effective_seed, workers=True)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2 ** 32))
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    try:
        repo_root = Path(get_original_cwd()).resolve() if os.getcwd() else PROJECT_ROOT.resolve()
    except Exception:
        repo_root = PROJECT_ROOT.resolve()

    try:
        cfg = compose_cfg(experiment=args.experiment, extra_overrides=args.hydra_overrides)

        cfg.seed = int(args.seed)
        if str(args.checkpoint).strip() != "":
            cfg.checkpoint = str(args.checkpoint).strip()

        effective_seed = set_seed(seed=int(args.seed), rank=rank)
        torch.set_float32_matmul_precision("high")

        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        data_module = hydra.utils.instantiate(
            cfg.data_module,
            processor=processor,
            encoder_variant=cfg.model.vision_model.variant,
            llm_variant=cfg.model.language_model.variant,
            _recursive_=False,
        )
        # Use the canonical "all" training bucket directly; no sampler needed for offline collection.
        train_dataset = data_module._instantiate_train_bucket_dataset(bucket_name="all")

        all_records = build_sample_records(train_dataset)
        if len(all_records) == 0:
            raise RuntimeError("No training records found in all bucket.")

        route_to_records: Dict[str, List[SampleRecord]] = defaultdict(list)
        for record in all_records:
            route_to_records[record.route_key].append(record)

        all_route_keys = sorted(route_to_records.keys())
        local_route_keys = split_route_keys_for_rank(all_route_keys, rank=rank, world_size=world_size)
        local_records: List[SampleRecord] = []
        for route_key in local_route_keys:
            local_records.extend(route_to_records[route_key])

        local_batch_size = int(args.local_batch_size) if int(args.local_batch_size) > 0 else int(cfg.data_module.batch_size)
        if local_batch_size <= 0:
            raise ValueError(f"local batch size must be > 0, got {local_batch_size}")

        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=None,
            _recursive_=False,
        )

        checkpoint_path = resolve_checkpoint_path(repo_root=repo_root, checkpoint=cfg.checkpoint)
        if checkpoint_path is not None:
            if checkpoint_path.is_dir():
                state_dict = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_path))
            else:
                state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()
        if hasattr(model, "budget_assigner") and model.budget_assigner is not None:
            model.budget_assigner.reset()

        use_autocast, autocast_dtype, autocast_reason = resolve_autocast_settings(cfg_precision=cfg.precision, device=device)

        if str(args.output_dir).strip() == "":
            out_dir = default_output_dir(repo_root=repo_root, cfg=cfg)
        else:
            candidate = Path(str(args.output_dir).strip())
            out_dir = candidate if candidate.is_absolute() else (repo_root / candidate)
            out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        budget_value = float(args.full_budget)
        tag_budget = f"{budget_value:.4f}".replace(".", "p")
        part_file = out_dir / f"train_lfull_stage2_seed{int(args.seed)}_budget{tag_budget}_rank{rank}.jsonl"

        stage2_cfg = cfg.model.smart_assigner_train.stage2
        lambda_wp = float(stage2_cfg.lambda_wp)
        lambda_lang = float(stage2_cfg.lambda_lang)
        driving_loss_weight = float(stage2_cfg.driving_loss_weight)
        driving_loss_warmup_steps = int(stage2_cfg.driving_loss_warmup_steps)

        local_num_frames = len(local_records)
        local_route_counts = {route_key: len(route_to_records[route_key]) for route_key in local_route_keys}

        local_loss_keys = set()
        local_rows_written = 0

        route_batches = build_route_batches(local_records, route_batch_size=local_batch_size)
        route_frame_index = defaultdict(int)
        route_last_frame_id = {}

        with torch.no_grad(), open(part_file, "w", encoding="utf-8") as f:
            pbar = tqdm(
                total=local_num_frames,
                desc=f"Rank {rank} collecting",
                unit="frame",
                disable=not is_main_process(rank),
            )
            for batch_records in route_batches:
                samples, sample_seeds = fetch_batch_samples(
                    dataset=train_dataset,
                    batch_records=batch_records,
                    base_seed=int(args.seed),
                )
                batch = data_module.dl_collate_fn(samples)
                batch = move_to_device(batch, device)
                route_keys = [record.route_key for record in batch_records]

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
                    if use_autocast
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    loss_dict, _ = model.forward_loss(
                        batch,
                        per_sample=True,
                        budget=budget_value,
                        route_keys=route_keys,
                    )

                per_sample_losses = per_sample_loss_scalars(loss_dict)
                loss_keys_sorted = sorted(per_sample_losses.keys())
                for key in loss_keys_sorted:
                    local_loss_keys.add(key)

                batch_size = len(batch_records)
                wp_keys = sorted([key for key in loss_keys_sorted if key.endswith("_loss") and key != "language_loss"])
                wp_total = torch.zeros(batch_size, device=device)
                for key in wp_keys:
                    wp_total = wp_total + per_sample_losses[key]
                if "language_loss" in per_sample_losses:
                    lang_loss = per_sample_losses["language_loss"]
                else:
                    lang_loss = torch.zeros(batch_size, device=device)

                loss_driving_unweighted = wp_total + lang_loss
                loss_driving_weighted = lambda_wp * wp_total + lambda_lang * lang_loss

                loss_total_all = torch.zeros(batch_size, device=device)
                for key in loss_keys_sorted:
                    loss_total_all = loss_total_all + per_sample_losses[key]

                for i, record in enumerate(batch_records):
                    last_frame = route_last_frame_id.get(record.route_key, None)
                    if last_frame is not None and record.frame_id <= last_frame:
                        raise RuntimeError(
                            f"Frame order violation in route {record.route_key}: prev={last_frame}, curr={record.frame_id}"
                        )
                    route_last_frame_id[record.route_key] = record.frame_id

                    frame_idx_in_route = route_frame_index[record.route_key]
                    route_frame_index[record.route_key] += 1

                    row = {
                        "measurement_path": to_repo_relative(record.measurement_path, repo_root),
                        "route_key": to_repo_relative(record.route_key, repo_root),
                        "route_id": Path(record.route_key).name,
                        "frame_id": int(record.frame_id),
                        "route_frame_index": int(frame_idx_in_route),
                        "route_frame_count": int(local_route_counts[record.route_key]),
                        "rank": int(rank),
                        "sample_seed": int(sample_seeds[i]),
                        "seed_base": int(args.seed),
                        "seed_effective_rank": int(effective_seed),
                        "full_budget": float(budget_value),
                        "lambda_wp_stage2": float(lambda_wp),
                        "lambda_lang_stage2": float(lambda_lang),
                        "driving_loss_weight_stage2": float(driving_loss_weight),
                        "driving_loss_warmup_steps_stage2": int(driving_loss_warmup_steps),
                        "loss_wp_total_full": float(wp_total[i].detach().cpu()),
                        "loss_lang_full": float(lang_loss[i].detach().cpu()),
                        "loss_driving_full_unweighted": float(loss_driving_unweighted[i].detach().cpu()),
                        "loss_driving_full": float(loss_driving_weighted[i].detach().cpu()),
                        "loss_total_all_components_full": float(loss_total_all[i].detach().cpu()),
                    }

                    for key in loss_keys_sorted:
                        row[f"{key}_full"] = float(per_sample_losses[key][i].detach().cpu())

                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    local_rows_written += 1

                pbar.update(batch_size)
            pbar.close()

        local_report = {
            "rank": int(rank),
            "local_rank": int(local_rank),
            "num_routes": int(len(local_route_keys)),
            "num_frames": int(local_num_frames),
            "rows_written": int(local_rows_written),
            "part_file": str(part_file),
            "loss_keys": sorted(local_loss_keys),
        }

        barrier_if_needed(world_size)

        if world_size > 1:
            gathered_reports: List[Optional[dict]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_reports, local_report)
        else:
            gathered_reports = [local_report]

        barrier_if_needed(world_size)

        if is_main_process(rank):
            merged_file = out_dir / f"train_lfull_stage2_seed{int(args.seed)}_budget{tag_budget}_all.jsonl"
            with open(merged_file, "w", encoding="utf-8") as out_f:
                for report in sorted(gathered_reports, key=lambda x: int(x["rank"])):
                    src = Path(report["part_file"])
                    with open(src, "r", encoding="utf-8") as in_f:
                        shutil.copyfileobj(in_f, out_f)

            summary = {
                "created_at": datetime.datetime.now().isoformat(),
                "experiment": str(args.experiment),
                "world_size": int(world_size),
                "seed_base": int(args.seed),
                "full_budget": float(budget_value),
                "precision": str(cfg.precision),
                "autocast": {
                    "enabled": bool(use_autocast),
                    "dtype": None if autocast_dtype is None else str(autocast_dtype),
                    "reason": str(autocast_reason),
                },
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                "output_files": {
                    "merged": str(merged_file),
                    "parts": [str(report["part_file"]) for report in sorted(gathered_reports, key=lambda x: int(x["rank"]))],
                },
                "stage2_driving_loss": {
                    "lambda_wp": float(lambda_wp),
                    "lambda_lang": float(lambda_lang),
                    "driving_loss_weight": float(driving_loss_weight),
                    "driving_loss_warmup_steps": int(driving_loss_warmup_steps),
                },
                "reports": sorted(gathered_reports, key=lambda x: int(x["rank"])),
            }
            summary_file = out_dir / f"train_lfull_stage2_seed{int(args.seed)}_budget{tag_budget}_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print("[collect_lfull] done")
            print(f"[collect_lfull] merged_jsonl={merged_file}")
            print(f"[collect_lfull] summary_json={summary_file}")

        barrier_if_needed(world_size)
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
