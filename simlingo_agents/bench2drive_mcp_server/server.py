from __future__ import annotations

import os
from typing import Any

from simlingo_agents.bench2drive_mcp_server.session_manager import EvaluationSessionManager

try:  # pragma: no cover - depends on optional runtime package
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - compile-time fallback
    FastMCP = None


APP_NAME = os.getenv("MCP_APP_NAME", "bench2drive-mcp")
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "8124"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
manager = EvaluationSessionManager(repo_root=REPO_ROOT)


def build_server():
    if FastMCP is None:
        raise RuntimeError(
            "The 'mcp' package is not installed. Install dependencies from "
            "simlingo_agents/bench2drive_mcp_server/requirements.txt first."
        )

    mcp = FastMCP(
        APP_NAME,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    )

    @mcp.resource("bench2drive://runtime")
    def runtime_resource() -> dict[str, Any]:
        """Describe the local Bench2Drive + split-agent runtime wiring."""

        return manager.describe_runtime()

    @mcp.resource("bench2drive://sessions")
    def sessions_resource() -> list[dict[str, Any]]:
        """Return the latest view of all evaluation sessions."""

        return manager.list_sessions()

    @mcp.resource("bench2drive://sessions/{session_id}")
    def session_resource(session_id: str) -> dict[str, Any]:
        """Return one evaluation session snapshot."""

        return manager.get_session_status(session_id)

    @mcp.resource("bench2drive://sessions/{session_id}/logs/{stream}")
    def session_log_resource(session_id: str, stream: str) -> dict[str, Any]:
        """Return the tail of one session log."""

        return manager.tail_log(session_id, stream=stream, lines=120)

    @mcp.tool()
    def describe_runtime() -> dict[str, Any]:
        """Describe repo paths, default local CARLA wiring, and split-agent endpoints."""

        return manager.describe_runtime()

    @mcp.tool()
    def validate_runtime(
        eval_config_path: str | None = None,
        carla_host: str | None = None,
        carla_port: int | None = None,
        traffic_manager_port: int | None = None,
        require_carla: bool = True,
        require_agents: bool = True,
        require_nats: bool = True,
    ) -> dict[str, Any]:
        """Validate the local Bench2Drive runtime before starting a session."""

        return manager.validate_runtime(
            eval_config_path=eval_config_path,
            carla_host=carla_host,
            carla_port=carla_port,
            traffic_manager_port=traffic_manager_port,
            require_carla=require_carla,
            require_agents=require_agents,
            require_nats=require_nats,
        )

    @mcp.tool()
    def list_eval_configs(search_root: str | None = None) -> list[dict[str, Any]]:
        """List candidate eval YAML files under the repo config directory."""

        return manager.list_eval_configs(search_root=search_root)

    @mcp.tool()
    def list_routes(eval_config_path: str, route_ids: list[str] | None = None) -> dict[str, Any]:
        """List routes resolved from one Bench2Drive eval config."""

        return manager.list_routes(eval_config_path=eval_config_path, route_ids=route_ids)

    @mcp.tool()
    def start_evaluation(
        eval_config_path: str,
        route_ids: list[str] | None = None,
        seed: int | None = None,
        carla_host: str | None = None,
        carla_port: int | None = None,
        traffic_manager_port: int | None = None,
        use_existing_carla: bool = True,
        remote_carla_host: str | None = None,
        remote_carla_port: int | None = None,
        remote_tm_port: int | None = None,
        gpu_ids: list[int] | None = None,
        session_label: str | None = None,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        """Launch a Bench2Drive evaluation against an already running local/remote CARLA server."""

        return manager.start_session(
            eval_config_path=eval_config_path,
            route_ids=route_ids,
            seed=seed,
            carla_host=carla_host,
            carla_port=carla_port,
            traffic_manager_port=traffic_manager_port,
            use_existing_carla=use_existing_carla,
            remote_carla_host=remote_carla_host,
            remote_carla_port=remote_carla_port,
            remote_tm_port=remote_tm_port,
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


mcp = build_server() if FastMCP is not None else None


if __name__ == "__main__":  # pragma: no cover - manual runtime entrypoint
    if mcp is None:
        build_server().run(transport=DEFAULT_TRANSPORT)
    else:
        mcp.run(transport=DEFAULT_TRANSPORT)
