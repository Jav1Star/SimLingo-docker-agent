import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_metric_cache_ready(metric_cache_path: Path) -> None:
    metadata_dir = metric_cache_path / "metadata"
    metadata_files = list(metadata_dir.glob("*.csv"))
    if metadata_files:
        return

    # 评分入口依赖 cache metadata，没生成完时直接给清晰报错。
    raise RuntimeError(
        f"metric cache未就绪：{metadata_dir} 下还没有metadata csv。"
        "请先运行 run_metric_caching.py，并等待缓存任务完成后再启动评分。"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--train-test-split")
    parser.add_argument("--experiment-name")
    parser.add_argument("--worker")
    parser.add_argument("--device")
    parser.add_argument("--navsim-log-path")
    parser.add_argument("--original-sensor-path")
    parser.add_argument("--metric-cache-path")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    simlingo_root = Path(__file__).resolve().parents[1]
    navsim_root = simlingo_root.parent / "navsim"
    script_path = navsim_root / "navsim" / "planning" / "script" / "run_pdm_score_one_stage.py"

    cfg = {}
    if args.config is not None:
        cfg = load_config(args.config)

    checkpoint_path = args.checkpoint_path or cfg.get("checkpoint_path")
    train_test_split = args.train_test_split or cfg.get("train_test_split", "navtest")
    experiment_name = args.experiment_name or cfg.get("experiment_name", "navsim_smart_assigner_eval")
    worker = args.worker or cfg.get("worker", "sequential")
    device = args.device or cfg.get("device", "cuda")
    navsim_log_path = args.navsim_log_path or cfg.get("navsim_log_path")
    original_sensor_path = args.original_sensor_path or cfg.get("original_sensor_path")
    metric_cache_path = args.metric_cache_path or cfg.get("metric_cache_path")
    config_overrides = list(cfg.get("hydra_overrides", []))
    config_overrides.extend(args.override)

    if metric_cache_path is not None:
        ensure_metric_cache_ready(Path(metric_cache_path).expanduser().resolve())

    cmd = [
        sys.executable,
        str(script_path),
        f"experiment_name={experiment_name}",
        f"train_test_split={train_test_split}",
        f"worker={worker}",
        "agent._target_=navism_eval.agent.NavsimSmartAssignerAgent",
        "agent._convert_=all",
        f"+agent.checkpoint_path={Path(checkpoint_path).expanduser().resolve()}",
        f"+agent.device={device}",
        "agent.trajectory_sampling._target_=nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
        "agent.trajectory_sampling._convert_=all",
        "agent.trajectory_sampling.time_horizon=4.0",
        "agent.trajectory_sampling.interval_length=0.1",
    ]
    if "verbose" in cfg:
        cmd.append(f"verbose={str(cfg['verbose']).lower()}")
    if navsim_log_path is not None:
        cmd.append(f"navsim_log_path={Path(navsim_log_path).expanduser().resolve()}")
    if original_sensor_path is not None:
        cmd.append(f"original_sensor_path={Path(original_sensor_path).expanduser().resolve()}")
    if metric_cache_path is not None:
        cmd.append(f"metric_cache_path={Path(metric_cache_path).expanduser().resolve()}")
    cmd.extend(config_overrides)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{simlingo_root}:{navsim_root}:{env.get('PYTHONPATH', '')}"
    if "open_scene_data_root" in cfg:
        env["OPENSCENE_DATA_ROOT"] = str(Path(cfg["open_scene_data_root"]).expanduser().resolve())
    if "navsim_exp_root" in cfg:
        env["NAVSIM_EXP_ROOT"] = str(Path(cfg["navsim_exp_root"]).expanduser().resolve())
    if "nuplan_maps_root" in cfg:
        env["NUPLAN_MAPS_ROOT"] = str(Path(cfg["nuplan_maps_root"]).expanduser().resolve())

    subprocess.run(cmd, cwd=navsim_root, env=env, check=True)


if __name__ == "__main__":
    main()
