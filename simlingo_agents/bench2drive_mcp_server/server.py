from __future__ import annotations

import os
from typing import Any

from simlingo_agents.bench2drive_mcp_server.session_manager import EvaluationSessionManager

try:  # pragma: no cover - depends on optional runtime package
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - compile-time fallback
    FastMCP = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager = EvaluationSessionManager(repo_root=REPO_ROOT)


def build_server():
    if FastMCP is None:
        raise RuntimeError(
            "The 'mcp' package is not installed. Install dependencies from "
            "simlingo_agents/bench2drive_mcp_server/requirements.txt first."
        )

    mcp = FastMCP("bench2drive-mcp")

    @mcp.tool()
    def start_evaluation(
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
        """Launch a Bench2Drive evaluation in the background and return a session id immediately."""

        return manager.start_session(
            eval_config_path=eval_config_path,
            route_ids=route_ids,
            seed=seed,
            remote_carla_port=remote_carla_port,
            remote_tm_port=remote_tm_port,
            remote_carla_host=remote_carla_host,
            gpu_ids=gpu_ids,
            session_label=session_label,
            output_root=output_root,
        )

    @mcp.tool()
    def get_evaluation_status(session_id: str) -> dict[str, Any]:
        """Poll one evaluation session without blocking on route execution."""

        return manager.get_session_status(session_id)

    @mcp.tool()
    def list_evaluations() -> list[dict[str, Any]]:
        """List known evaluation sessions and their latest summaries."""

        return manager.list_sessions()

    @mcp.tool()
    def cancel_evaluation(session_id: str) -> dict[str, Any]:
        """Request termination of a background evaluation session."""

        return manager.cancel_session(session_id)

    @mcp.tool()
    def read_evaluation_log(session_id: str, stream: str = "stdout", lines: int = 80) -> dict[str, Any]:
        """Read the tail of the launcher stdout/stderr log for a session."""

        return manager.tail_log(session_id, stream=stream, lines=lines)

    return mcp


if __name__ == "__main__":  # pragma: no cover - manual runtime entrypoint
    build_server().run()
