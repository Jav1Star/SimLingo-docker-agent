from __future__ import annotations

import glob
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import yaml

from simlingo_agents.bench2drive_mcp_server.remote_inference import (
    configured_jetstream_domain,
    configured_nats_server_url,
    configured_pipeline_endpoints,
    split_stack_profile,
)


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))


def _sanitize_label(value: str | None) -> str:
    if not value:
        return "session"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    safe = safe.strip("-_")
    return safe or "session"


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _tail_lines(path: str, limit: int) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = deque(f, maxlen=max(1, int(limit)))
    return "".join(lines)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@dataclass
class SessionPaths:
    session_dir: str
    session_json: str
    generated_eval_yaml: str
    stdout_log: str
    stderr_log: str


class EvaluationSessionManager:
    """Persists and supervises background Bench2Drive evaluation jobs."""

    def __init__(self, repo_root: str, runtime_root: str | None = None) -> None:
        self.repo_root = _expand(repo_root) or os.getcwd()
        self.config_root = os.path.join(self.repo_root, "configs")
        self.bench_root = os.path.join(self.repo_root, "Bench2Drive")
        self.team_code_root = os.path.join(self.repo_root, "team_code_adaption")
        self.runtime_root = _expand(runtime_root) or os.path.join(
            self.repo_root,
            "simlingo_agents",
            "bench2drive_mcp_server",
            "runtime",
        )
        os.makedirs(self.runtime_root, exist_ok=True)
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def start_session(
        self,
        *,
        eval_config_path: str,
        route_ids: list[str] | None = None,
        seed: int | None = None,
        carla_host: str | None = None,
        carla_port: int | None = None,
        traffic_manager_port: int | None = None,
        use_existing_carla: bool = True,
        remote_carla_port: int | None = None,
        remote_tm_port: int | None = None,
        remote_carla_host: str | None = None,
        gpu_ids: list[int] | None = None,
        session_label: str | None = None,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        eval_config_path = _expand(eval_config_path)
        if not eval_config_path or not os.path.exists(eval_config_path):
            raise FileNotFoundError(f"eval_config_path does not exist: {eval_config_path}")

        eval_doc, eval_cfg = self._load_eval_document(eval_config_path)
        carla_conn = self._resolve_carla_connection(
            eval_cfg=eval_cfg,
            carla_host=carla_host,
            carla_port=carla_port,
            traffic_manager_port=traffic_manager_port,
            remote_carla_host=remote_carla_host,
            remote_carla_port=remote_carla_port,
            remote_tm_port=remote_tm_port,
            use_existing_carla=use_existing_carla,
        )

        session_id = self._new_session_id(session_label)
        paths = self._build_paths(session_id)
        os.makedirs(paths.session_dir, exist_ok=True)

        generated_cfg = self._build_generated_eval_config(
            source_eval_document=eval_doc,
            session_id=session_id,
            output_root=output_root,
            carla_host=carla_conn["carla_host"],
        )
        with open(paths.generated_eval_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(generated_cfg, f, allow_unicode=True, sort_keys=False)

        expected_routes = self._resolve_expected_routes(
            generated_cfg,
            route_ids=route_ids,
        )
        command = [
            sys.executable,
            os.path.join(self.repo_root, "start_eval_simlingo_adaption.py"),
            "--eval-config",
            paths.generated_eval_yaml,
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if route_ids:
            command.append("--route-id")
            command.extend([str(item) for item in route_ids])
        if carla_conn["use_existing_carla"]:
            command.extend(["--remote-carla-port", str(carla_conn["carla_port"])])
            command.extend(["--remote-tm-port", str(carla_conn["traffic_manager_port"])])
            command.extend(["--remote-carla-host", carla_conn["carla_host"]])
        if gpu_ids:
            command.append("--gpu")
            command.extend([str(item) for item in gpu_ids])

        stdout_handle = open(paths.stdout_log, "w", encoding="utf-8")
        stderr_handle = open(paths.stderr_log, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            preexec_fn=os.setsid,
        )
        stdout_handle.close()
        stderr_handle.close()
        self._processes[session_id] = process

        session = {
            "session_id": session_id,
            "label": session_label,
            "status": "running",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "started_ts": _now_ts(),
            "eval_config_path": eval_config_path,
            "generated_eval_config_path": paths.generated_eval_yaml,
            "stdout_log": paths.stdout_log,
            "stderr_log": paths.stderr_log,
            "pid": process.pid,
            "pgid": process.pid,
            "command": command,
            "route_ids": route_ids,
            "seed": seed,
            "carla_mode": "external" if carla_conn["use_existing_carla"] else "managed",
            "carla_host": carla_conn["carla_host"],
            "carla_port": carla_conn["carla_port"],
            "traffic_manager_port": carla_conn["traffic_manager_port"],
            "gpu_ids": gpu_ids,
            "result_root": generated_cfg["eval"]["out_root"],
            "agent_name": generated_cfg["eval"]["agent"],
            "benchmark": generated_cfg["eval"]["benchmark"],
            "expected_routes": expected_routes,
        }
        _write_json(paths.session_json, session)
        return self.get_session_status(session_id)

    def describe_runtime(self) -> dict[str, Any]:
        stack = self._split_stack_config()
        config_candidates = self.list_eval_configs()
        active_sessions = self.list_sessions()
        return {
            "repo_root": self.repo_root,
            "bench_root": self.bench_root,
            "runtime_root": self.runtime_root,
            "team_code_root": self.team_code_root,
            "default_carla": {
                "host": os.getenv("SIMLINGO_CARLA_HOST", "127.0.0.1"),
                "port": int(os.getenv("SIMLINGO_CARLA_PORT", "2000")),
                "traffic_manager_port": int(os.getenv("SIMLINGO_CARLA_TM_PORT", "8000")),
                "mode": "external",
            },
            "split_stack": stack,
            "paths": {
                "start_eval_script": os.path.join(self.repo_root, "start_eval_simlingo_adaption.py"),
                "remote_agent_file": os.path.join(self.team_code_root, "agent_simlingo_remote.py"),
                "leaderboard_evaluator": os.path.join(
                    self.bench_root, "leaderboard", "leaderboard", "leaderboard_evaluator.py"
                ),
            },
            "available_eval_configs": config_candidates,
            "session_count": len(active_sessions),
        }

    def validate_runtime(
        self,
        *,
        eval_config_path: str | None = None,
        carla_host: str | None = None,
        carla_port: int | None = None,
        traffic_manager_port: int | None = None,
        require_carla: bool = True,
        require_agents: bool = True,
        require_nats: bool = True,
    ) -> dict[str, Any]:
        checks: dict[str, Any] = {
            "files": {},
            "services": {},
            "overall_status": "ok",
        }

        start_eval_script = os.path.join(self.repo_root, "start_eval_simlingo_adaption.py")
        remote_agent_file = os.path.join(self.team_code_root, "agent_simlingo_remote.py")
        leaderboard_evaluator = os.path.join(
            self.bench_root, "leaderboard", "leaderboard", "leaderboard_evaluator.py"
        )
        for label, path in {
            "start_eval_script": start_eval_script,
            "remote_agent_file": remote_agent_file,
            "leaderboard_evaluator": leaderboard_evaluator,
        }.items():
            checks["files"][label] = {
                "path": path,
                "exists": os.path.exists(path),
            }

        if eval_config_path:
            expanded = _expand(eval_config_path)
            exists = bool(expanded and os.path.exists(expanded))
            checks["files"]["eval_config"] = {
                "path": expanded,
                "exists": exists,
            }
            if exists:
                _doc, eval_cfg = self._load_eval_document(expanded)
                route_path = _expand(eval_cfg.get("route_path"))
                checkpoint = _expand(eval_cfg.get("checkpoint"))
                checks["files"]["route_path"] = {
                    "path": route_path,
                    "exists": bool(route_path and os.path.isdir(route_path)),
                }
                checks["files"]["checkpoint"] = {
                    "path": checkpoint,
                    "exists": bool(checkpoint and os.path.exists(checkpoint)),
                }

        stack = self._split_stack_config()
        if require_agents:
            for agent_name, execute_url in stack["agent_execute_urls"].items():
                checks["services"][agent_name] = self._probe_http_health(execute_url)

        if require_nats:
            checks["services"]["nats"] = self._probe_socket(stack["nats_server_url"])

        if require_carla:
            checks["services"]["carla"] = self._probe_carla(
                carla_host=carla_host,
                carla_port=carla_port,
                traffic_manager_port=traffic_manager_port,
            )

        failed = []
        for group in ("files", "services"):
            for name, item in checks[group].items():
                if not item.get("exists", item.get("ok", False)):
                    failed.append(f"{group}.{name}")
        if failed:
            checks["overall_status"] = "error"
            checks["failed_checks"] = failed
        return checks

    def list_eval_configs(self, search_root: str | None = None) -> list[dict[str, Any]]:
        root = _expand(search_root) or self.config_root
        patterns = [
            os.path.join(root, "*.yaml"),
            os.path.join(root, "*.yml"),
        ]
        configs: list[dict[str, Any]] = []
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                try:
                    _doc, eval_cfg = self._load_eval_document(path)
                except Exception as exc:
                    configs.append(
                        {
                            "path": path,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    continue
                configs.append(
                    {
                        "path": path,
                        "benchmark": eval_cfg.get("benchmark"),
                        "agent": eval_cfg.get("agent"),
                        "route_path": eval_cfg.get("route_path"),
                        "out_root": eval_cfg.get("out_root"),
                        "carla_root": eval_cfg.get("carla_root"),
                        "status": "ok",
                    }
                )
        return configs

    def list_routes(self, eval_config_path: str, route_ids: list[str] | None = None) -> dict[str, Any]:
        eval_config_path = _expand(eval_config_path)
        if not eval_config_path or not os.path.exists(eval_config_path):
            raise FileNotFoundError(f"eval_config_path does not exist: {eval_config_path}")
        _doc, eval_cfg = self._load_eval_document(eval_config_path)
        route_path = _expand(eval_cfg["route_path"])
        if not route_path or not os.path.isdir(route_path):
            raise FileNotFoundError(f"route_path does not exist or is not a directory: {route_path}")

        routes = []
        wanted = {str(item).zfill(3) for item in route_ids} if route_ids else None
        for filename in sorted(os.listdir(route_path)):
            if not filename.endswith(".xml"):
                continue
            route_id = filename.split("_")[-1][:-4].zfill(3)
            if wanted and route_id not in wanted:
                continue
            routes.append(
                {
                    "route_id": route_id,
                    "path": os.path.join(route_path, filename),
                }
            )
        return {
            "eval_config_path": eval_config_path,
            "route_path": route_path,
            "route_count": len(routes),
            "routes": routes,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for session_json in sorted(glob.glob(os.path.join(self.runtime_root, "*", "session.json"))):
            session_id = pathlib.Path(session_json).parent.name
            try:
                sessions.append(self.get_session_status(session_id))
            except Exception as exc:  # pragma: no cover - defensive
                sessions.append(
                    {
                        "session_id": session_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        return sessions

    def get_session_status(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        session = self._refresh_session_status(session)
        return session

    def cancel_session(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        pgid = int(session["pgid"])
        if session.get("status") not in {"running", "cancelling"}:
            return session
        try:
            os.killpg(pgid, signal.SIGTERM)
            session["status"] = "cancelling"
            session["updated_at"] = _now_iso()
            _write_json(self._build_paths(session_id).session_json, session)
        except ProcessLookupError:
            pass
        return self.get_session_status(session_id)

    def tail_log(self, session_id: str, *, stream: str = "stdout", lines: int = 80) -> dict[str, Any]:
        session = self.get_session_status(session_id)
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        key = "stdout_log" if stream == "stdout" else "stderr_log"
        return {
            "session_id": session_id,
            "stream": stream,
            "path": session[key],
            "content": _tail_lines(session[key], lines),
        }

    def _refresh_session_status(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = session["session_id"]
        process = self._processes.get(session_id)
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                session["returncode"] = returncode
                session["finished_ts"] = _now_ts()
                session["status"] = "completed" if returncode == 0 else "failed"
                self._processes.pop(session_id, None)
        else:
            pid = int(session.get("pid", -1))
            alive = pid > 0 and _is_pid_alive(pid)
            if alive:
                if session.get("status") not in {"running", "cancelling"}:
                    session["status"] = "running"
            else:
                session.setdefault("finished_ts", _now_ts())
                if session.get("status") in {"running", "cancelling"}:
                    session["status"] = self._infer_terminal_status(session)

        session["result_summary"] = self._summarize_results(session)
        session["status"] = self._normalize_status(session)
        session["updated_at"] = _now_iso()
        _write_json(self._build_paths(session_id).session_json, session)
        return session

    def _infer_terminal_status(self, session: dict[str, Any]) -> str:
        summary = self._summarize_results(session)
        if summary["finished_routes"] >= summary["expected_route_count"] and summary["failed_routes"] == 0:
            return "completed"
        if session.get("status") == "cancelling":
            return "cancelled"
        return "failed"

    def _normalize_status(self, session: dict[str, Any]) -> str:
        status = session.get("status", "unknown")
        summary = session.get("result_summary", {})
        if status in {"running", "cancelling", "cancelled"}:
            return status
        if status == "completed":
            if summary.get("failed_routes", 0) > 0:
                return "completed_with_failures"
            return "completed"
        if status == "failed":
            return "failed"
        if summary.get("finished_routes", 0) >= summary.get("expected_route_count", 0):
            return "completed_with_failures" if summary.get("failed_routes", 0) > 0 else "completed"
        return status

    def _summarize_results(self, session: dict[str, Any]) -> dict[str, Any]:
        result_root = session["result_root"]
        expected_routes = session.get("expected_routes", [])
        result_files = sorted(glob.glob(os.path.join(result_root, "**", "res", "*_res.json"), recursive=True))
        route_statuses: dict[str, str] = {}
        for result_file in result_files:
            route_id = os.path.basename(result_file).split("_")[0]
            try:
                data = _read_json(result_file)
                status = (
                    data.get("_checkpoint", {})
                    .get("global_record", {})
                    .get("status", "Unknown")
                )
            except Exception:
                status = "Unreadable"
            route_statuses[route_id] = status

        finished_routes = 0
        completed_routes = 0
        failed_routes = 0
        for route_id in expected_routes:
            status = route_statuses.get(route_id)
            if status is None:
                continue
            finished_routes += 1
            if status == "Completed":
                completed_routes += 1
            else:
                failed_routes += 1

        return {
            "expected_route_count": len(expected_routes),
            "finished_routes": finished_routes,
            "completed_routes": completed_routes,
            "failed_routes": failed_routes,
            "route_statuses": route_statuses,
        }

    def _resolve_expected_routes(
        self,
        generated_cfg: dict[str, Any],
        *,
        route_ids: list[str] | None,
    ) -> list[str]:
        eval_cfg = generated_cfg.get("eval", generated_cfg)
        route_path = _expand(eval_cfg["route_path"])
        if not route_path or not os.path.isdir(route_path):
            raise FileNotFoundError(f"route_path does not exist or is not a directory: {route_path}")
        routes = sorted(
            route_id.split("_")[-1][:-4].zfill(3)
            for route_id in os.listdir(route_path)
            if route_id.endswith(".xml")
        )
        if not route_ids:
            return routes
        wanted = {str(item).zfill(3) for item in route_ids}
        return [route_id for route_id in routes if route_id in wanted]

    def _build_generated_eval_config(
        self,
        *,
        source_eval_document: dict[str, Any],
        session_id: str,
        output_root: str | None,
        carla_host: str | None,
    ) -> dict[str, Any]:
        data = dict(source_eval_document)
        if "eval" not in data:
            data = {"eval": data}
        eval_cfg = data["eval"]
        agent_file = os.path.join(self.repo_root, "team_code_adaption", "agent_simlingo_remote.py")
        base_out_root = _expand(output_root) or _expand(eval_cfg["out_root"]) or os.path.join(self.repo_root, "eval_results")
        session_out_root = os.path.join(base_out_root, "mcp_sessions", session_id)
        eval_cfg["agent"] = "simlingo_remote_split"
        eval_cfg["agent_file"] = agent_file
        eval_cfg["out_root"] = session_out_root
        eval_cfg["repo_root"] = self.repo_root
        if carla_host:
            eval_cfg["carla_host"] = carla_host
        return data

    def _load_eval_document(self, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError(f"YAML root must be a mapping: {path}")
        eval_cfg = data.get("eval", data)
        if not isinstance(eval_cfg, dict):
            raise TypeError(f"'eval' section must be a mapping: {path}")
        return data, eval_cfg

    def _resolve_carla_connection(
        self,
        *,
        eval_cfg: dict[str, Any],
        carla_host: str | None,
        carla_port: int | None,
        traffic_manager_port: int | None,
        remote_carla_host: str | None,
        remote_carla_port: int | None,
        remote_tm_port: int | None,
        use_existing_carla: bool,
    ) -> dict[str, Any]:
        resolved_host = (
            carla_host
            or remote_carla_host
            or eval_cfg.get("carla_host")
            or os.getenv("SIMLINGO_CARLA_HOST", "127.0.0.1")
        )
        resolved_port = int(
            carla_port
            or remote_carla_port
            or os.getenv("SIMLINGO_CARLA_PORT", "2000")
        )
        resolved_tm_port = int(
            traffic_manager_port
            or remote_tm_port
            or os.getenv("SIMLINGO_CARLA_TM_PORT", "8000")
        )
        if use_existing_carla is None:
            use_existing_carla = True
        return {
            "use_existing_carla": bool(use_existing_carla),
            "carla_host": str(resolved_host),
            "carla_port": resolved_port,
            "traffic_manager_port": resolved_tm_port,
        }

    def _split_stack_config(self) -> dict[str, Any]:
        endpoints = configured_pipeline_endpoints()
        return {
            "profile": split_stack_profile(),
            "agent_execute_urls": {
                "encoder": endpoints.encoder_url,
                "scheduler": endpoints.scheduler_url,
                "llm": endpoints.llm_url,
            },
            "nats_server_url": configured_nats_server_url(),
            "nats_stream": os.getenv("NATS_STREAM", "WORKFLOW"),
            "nats_stream_subjects": [
                item.strip()
                for item in os.getenv("NATS_STREAM_SUBJECTS", "workflow.>").split(",")
                if item.strip()
            ],
            "nats_jetstream_domain": configured_jetstream_domain(),
        }

    def _probe_http_health(self, execute_url: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(execute_url)
        health_path = parsed.path
        if health_path.endswith("/a2a/execute"):
            health_path = health_path[: -len("/a2a/execute")] + "/health"
        else:
            health_path = "/health"
        health_url = urllib.parse.urlunparse(parsed._replace(path=health_path, query="", fragment=""))
        result = {
            "ok": False,
            "execute_url": execute_url,
            "health_url": health_url,
        }
        try:
            with urllib.request.urlopen(health_url, timeout=3.0) as response:
                body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body else {}
            result["ok"] = True
            result["response"] = payload
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def _probe_socket(self, target_url: str) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(target_url)
        host = parsed.hostname
        port = parsed.port
        result = {
            "ok": False,
            "url": target_url,
            "host": host,
            "port": port,
        }
        if not host or not port:
            result["error"] = "unable to parse host/port"
            return result
        try:
            with socket.create_connection((host, port), timeout=3.0):
                pass
            result["ok"] = True
        except OSError as exc:
            result["error"] = str(exc)
        return result

    def _probe_carla(
        self,
        *,
        carla_host: str | None,
        carla_port: int | None,
        traffic_manager_port: int | None,
    ) -> dict[str, Any]:
        host = carla_host or os.getenv("SIMLINGO_CARLA_HOST", "127.0.0.1")
        port = int(carla_port or os.getenv("SIMLINGO_CARLA_PORT", "2000"))
        tm_port = int(traffic_manager_port or os.getenv("SIMLINGO_CARLA_TM_PORT", "8000"))
        rpc_check = self._probe_socket(f"tcp://{host}:{port}")
        tm_check = self._probe_socket(f"tcp://{host}:{tm_port}")
        return {
            "ok": bool(rpc_check.get("ok")),
            "host": host,
            "carla_port": port,
            "traffic_manager_port": tm_port,
            "rpc": rpc_check,
            "traffic_manager": tm_check,
        }

    def _load_session(self, session_id: str) -> dict[str, Any]:
        session_json = self._build_paths(session_id).session_json
        if not os.path.exists(session_json):
            raise FileNotFoundError(f"Unknown session_id: {session_id}")
        return _read_json(session_json)

    def _new_session_id(self, label: str | None) -> str:
        return f"{_sanitize_label(label)}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def _build_paths(self, session_id: str) -> SessionPaths:
        session_dir = os.path.join(self.runtime_root, session_id)
        return SessionPaths(
            session_dir=session_dir,
            session_json=os.path.join(session_dir, "session.json"),
            generated_eval_yaml=os.path.join(session_dir, "generated_eval.yaml"),
            stdout_log=os.path.join(session_dir, "launcher.stdout.log"),
            stderr_log=os.path.join(session_dir, "launcher.stderr.log"),
        )
