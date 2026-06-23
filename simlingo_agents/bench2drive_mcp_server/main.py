from simlingo_agents.bench2drive_mcp_server.server import (
    APP_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TRANSPORT,
    mcp,
)


def main() -> None:
    if mcp is None:
        raise RuntimeError("MCP server is unavailable because the 'mcp' package is not installed.")
    print(
        f"Starting {APP_NAME} on {DEFAULT_HOST}:{DEFAULT_PORT} with transport={DEFAULT_TRANSPORT}",
        flush=True,
    )
    mcp.run(transport=DEFAULT_TRANSPORT)


if __name__ == "__main__":
    main()
