from __future__ import annotations

import glob
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import yaml


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

        session_id = self._new_session_id(session_label)
        paths = self._build_paths(session_id)
        os.makedirs(paths.session_dir, exist_ok=True)

        generated_cfg = self._build_generated_eval_config(
            source_eval_config=eval_config_path,
            session_id=session_id,
            output_root=output_root,
            remote_carla_host=remote_carla_host,
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
        if remote_carla_port is not None:
            command.extend(["--remote-carla-port", str(remote_carla_port)])
        if remote_tm_port is not None:
            command.extend(["--remote-tm-port", str(remote_tm_port)])
        if remote_carla_host:
            command.extend(["--remote-carla-host", remote_carla_host])
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
            "remote_carla_port": remote_carla_port,
            "remote_tm_port": remote_tm_port,
            "remote_carla_host": remote_carla_host,
            "gpu_ids": gpu_ids,
            "result_root": generated_cfg["eval"]["out_root"],
            "agent_name": generated_cfg["eval"]["agent"],
            "benchmark": generated_cfg["eval"]["benchmark"],
            "expected_routes": expected_routes,
        }
        _write_json(paths.session_json, session)
        return self.get_session_status(session_id)

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
        source_eval_config: str,
        session_id: str,
        output_root: str | None,
        remote_carla_host: str | None,
    ) -> dict[str, Any]:
        with open(source_eval_config, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
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
        if remote_carla_host:
            eval_cfg["carla_host"] = remote_carla_host
        return data

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
