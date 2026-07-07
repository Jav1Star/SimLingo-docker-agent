#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_RUN_DIR = REPO_ROOT / "models" / "2026_01_29_21_59_03_adaption_train_seed_9876"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_RUN_DIR / "checkpoints" / "epoch=002.ckpt" / "pytorch_model.bin"
DEFAULT_ROUTE_PATH = REPO_ROOT / "leaderboard" / "data" / "bench2drive_split"
DEFAULT_OUT_ROOT = REPO_ROOT / "eval_results" / "split_agents_local"
DEFAULT_CARLA_ROOT = Path("/data/gaoshuo/data/carla0915")


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def _http_health_url(execute_url: str) -> str:
    parsed = urllib.parse.urlparse(execute_url)
    path = parsed.path
    if path.endswith("/a2a/execute"):
        path = path[: -len("/a2a/execute")] + "/health"
    else:
        path = "/health"
    return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))


def _probe_http_health(execute_url: str, timeout_sec: float) -> tuple[bool, str]:
    health_url = _http_health_url(execute_url)
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
        return True, f"{health_url} ok {body[:200]}"
    except urllib.error.URLError as exc:
        return False, f"{health_url} failed: {exc}"
    except Exception as exc:
        return False, f"{health_url} failed: {exc}"


def _probe_socket(target_url: str, timeout_sec: float) -> tuple[bool, str]:
    parsed = urllib.parse.urlparse(target_url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False, f"unable to parse host/port from {target_url}"
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            pass
        return True, f"{host}:{port} ok"
    except OSError as exc:
        return False, f"{host}:{port} failed: {exc}"


def _default_urls(profile: str, node_host: str) -> dict[str, str]:
    if profile == "k8s":
        return {
            "encoder": "http://simlingo-encoder-service:9011/a2a/execute",
            "scheduler": "http://simlingo-scheduler-service:9013/a2a/execute",
            "llm": "http://simlingo-llm-service:9012/a2a/execute",
            "nats": "nats://nats:4222",
        }
    if profile == "nodeport":
        return {
            "encoder": f"http://{node_host}:30111/a2a/execute",
            "scheduler": f"http://{node_host}:30113/a2a/execute",
            "llm": f"http://{node_host}:30112/a2a/execute",
            "nats": "nats://127.0.0.1:4222",
        }
    return {
        "encoder": "http://127.0.0.1:9011/a2a/execute",
        "scheduler": "http://127.0.0.1:9013/a2a/execute",
        "llm": "http://127.0.0.1:9012/a2a/execute",
        "nats": "nats://127.0.0.1:4222",
    }


def _resolve_urls(args: argparse.Namespace) -> dict[str, str]:
    defaults = _default_urls(args.agent_profile, args.node_host)
    return {
        "encoder": args.encoder_url or os.getenv("SIMLINGO_ENCODER_AGENT_URL") or defaults["encoder"],
        "scheduler": args.scheduler_url or os.getenv("SIMLINGO_SCHEDULER_AGENT_URL") or defaults["scheduler"],
        "llm": args.llm_url or os.getenv("SIMLINGO_LLM_AGENT_URL") or defaults["llm"],
        "nats": args.nats_url or os.getenv("NATS_SERVER_URL") or defaults["nats"],
    }


def _load_eval_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _build_generated_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    source_path = _expand(args.eval_config)
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"eval config does not exist: {source_path}")

    data = copy.deepcopy(_load_eval_config(source_path))
    if "eval" not in data:
        data = {"eval": data}
    eval_cfg = data["eval"]
    if not isinstance(eval_cfg, dict):
        raise TypeError("'eval' section must be a mapping")

    eval_cfg["agent"] = "simlingo_remote_split"
    eval_cfg["agent_file"] = str(REPO_ROOT / "team_code_adaption" / "agent_simlingo_remote.py")
    eval_cfg["repo_root"] = str(REPO_ROOT)
    eval_cfg["carla_host"] = args.carla_host

    if args.checkpoint:
        eval_cfg["checkpoint"] = _expand(args.checkpoint)
    if args.route_path:
        eval_cfg["route_path"] = _expand(args.route_path)
    if args.out_root:
        eval_cfg["out_root"] = _expand(args.out_root)
    if args.carla_root:
        eval_cfg["carla_root"] = _expand(args.carla_root)

    return data


def _validate_eval_config_paths(config: dict[str, Any]) -> None:
    eval_cfg = config.get("eval", config)
    required_files = {
        "checkpoint": eval_cfg.get("checkpoint"),
        "agent_file": eval_cfg.get("agent_file"),
    }
    required_dirs = {
        "route_path": eval_cfg.get("route_path"),
        "carla_root": eval_cfg.get("carla_root"),
    }

    missing = []
    for label, path in required_files.items():
        expanded = _expand(path)
        if not expanded or not os.path.isfile(expanded):
            missing.append(f"{label}={expanded}")
    for label, path in required_dirs.items():
        expanded = _expand(path)
        if not expanded or not os.path.isdir(expanded):
            missing.append(f"{label}={expanded}")

    checkpoint = _expand(eval_cfg.get("checkpoint"))
    if checkpoint:
        hydra_config = Path(checkpoint).parent.parent.parent / ".hydra" / "config.yaml"
        if not hydra_config.is_file():
            missing.append(f"hydra_config={hydra_config}")

    if missing:
        raise FileNotFoundError("Invalid generated eval config paths: " + "; ".join(missing))


def _write_generated_config(config: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_path = run_dir / "generated_split_eval.yaml"
    with open(generated_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return generated_path


def _build_env(args: argparse.Namespace, urls: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["SIMLINGO_SPLIT_STACK_PROFILE"] = args.agent_profile
    env["SIMLINGO_ENCODER_AGENT_URL"] = urls["encoder"]
    env["SIMLINGO_SCHEDULER_AGENT_URL"] = urls["scheduler"]
    env["SIMLINGO_LLM_AGENT_URL"] = urls["llm"]
    env["NATS_SERVER_URL"] = urls["nats"]
    env.setdefault("NATS_STREAM_SUBJECTS", "workflow.*.*.*")
    env["SIMLINGO_MCP_SUBJECT_PREFIX"] = args.subject_prefix
    env["SIMLINGO_MCP_SENDER_ID"] = args.sender_id

    if args.nats_jetstream_domain is not None:
        env["NATS_JETSTREAM_DOMAIN"] = args.nats_jetstream_domain
    elif args.agent_profile in {"k8s", "nodeport"} and "NATS_JETSTREAM_DOMAIN" not in env:
        env["NATS_JETSTREAM_DOMAIN"] = "hub"

    return env


def _validate_runtime(args: argparse.Namespace, urls: dict[str, str]) -> bool:
    timeout_sec = args.check_timeout
    checks = [
        ("carla", _probe_socket(f"tcp://{args.carla_host}:{args.carla_port}", timeout_sec)),
        ("nats", _probe_socket(urls["nats"], timeout_sec)),
        ("encoder", _probe_http_health(urls["encoder"], timeout_sec)),
        ("scheduler", _probe_http_health(urls["scheduler"], timeout_sec)),
        ("llm", _probe_http_health(urls["llm"], timeout_sec)),
    ]
    print(
        "[check] traffic_manager: configured - "
        f"{args.carla_host}:{args.traffic_manager_port} "
        "(created/connected by leaderboard_evaluator via CARLA client)"
    )
    ok = True
    for name, (passed, detail) in checks:
        status = "ok" if passed else "error"
        print(f"[check] {name}: {status} - {detail}")
        ok = ok and passed
    return ok


def _build_command(args: argparse.Namespace, generated_eval_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "start_eval_simlingo_adaption.py"),
        "--eval-config",
        str(generated_eval_path),
        "--remote-carla-port",
        str(args.carla_port),
        "--remote-tm-port",
        str(args.traffic_manager_port),
        "--remote-carla-host",
        args.carla_host,
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.route_id:
        command.append("--route-id")
        command.extend(args.route_id)
    if args.gpu:
        command.append("--gpu")
        command.extend(str(item) for item in args.gpu)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a local Bench2Drive/CARLA evaluation that delegates per-frame "
            "inference to the split encoder/scheduler/llm agents."
        )
    )
    parser.add_argument(
        "--eval-config",
        default=str(REPO_ROOT / "configs" / "simlingo_adaption_eval.yaml"),
        help="Source eval YAML. checkpoint/route_path/budget are inherited from this file.",
    )
    parser.add_argument("--route-id", nargs="+", default=None, help="Only evaluate these route ids, for example: 053")
    parser.add_argument("--seed", type=int, default=None, help="Override seeds from eval config.")
    parser.add_argument("--gpu", type=int, nargs="+", default=None, help="GPU ids passed to start_eval_simlingo_adaption.py.")

    parser.add_argument("--carla-host", default=os.getenv("SIMLINGO_CARLA_HOST", "127.0.0.1"))
    parser.add_argument("--carla-port", type=int, default=int(os.getenv("SIMLINGO_CARLA_PORT", "2000")))
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=int(os.getenv("SIMLINGO_CARLA_TM_PORT", "8000")),
    )

    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Override eval.checkpoint.")
    parser.add_argument("--route-path", default=str(DEFAULT_ROUTE_PATH), help="Override eval.route_path.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Override eval.out_root.")
    parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT), help="Override eval.carla_root.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory for generated config. Defaults to simlingo_agents/local_split_eval_runs/<timestamp>.",
    )

    parser.add_argument(
        "--agent-profile",
        choices=["local", "nodeport", "k8s"],
        default=os.getenv("SIMLINGO_SPLIT_STACK_PROFILE", "local"),
        help="Default endpoint profile for split agents.",
    )
    parser.add_argument("--node-host", default=os.getenv("SIMLINGO_K8S_NODE_HOST", "127.0.0.1"))
    parser.add_argument("--encoder-url", default=None)
    parser.add_argument("--scheduler-url", default=None)
    parser.add_argument("--llm-url", default=None)
    parser.add_argument("--nats-url", default=None)
    parser.add_argument(
        "--nats-jetstream-domain",
        default=None,
        help="Set NATS_JETSTREAM_DOMAIN. For k8s/nodeport, defaults to hub when unset.",
    )
    parser.add_argument("--subject-prefix", default=os.getenv("SIMLINGO_MCP_SUBJECT_PREFIX", "workflow.local_eval"))
    parser.add_argument("--sender-id", default=os.getenv("SIMLINGO_MCP_SENDER_ID", "LocalBench2DriveEval"))

    parser.add_argument("--skip-runtime-checks", action="store_true", help="Do not probe CARLA/NATS/agent health before launch.")
    parser.add_argument("--check-timeout", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="Print generated config path and command without running evaluation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = _resolve_urls(args)
    run_dir = Path(
        _expand(args.run_dir)
        or REPO_ROOT / "simlingo_agents" / "local_split_eval_runs" / time.strftime("%Y%m%d-%H%M%S")
    )
    generated_eval_config = _build_generated_eval_config(args)
    _validate_eval_config_paths(generated_eval_config)
    generated_config = _write_generated_config(generated_eval_config, run_dir)
    env = _build_env(args, urls)
    command = _build_command(args, generated_config)

    print(f"[config] generated eval config: {generated_config}")
    print(f"[agent] encoder:   {urls['encoder']}")
    print(f"[agent] scheduler: {urls['scheduler']}")
    print(f"[agent] llm:       {urls['llm']}")
    print(f"[nats] {urls['nats']}")

    if not args.skip_runtime_checks and not _validate_runtime(args, urls):
        print("[error] runtime checks failed. Use --skip-runtime-checks only if the probes are expected to fail.")
        return 2

    print("[command] " + " ".join(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
