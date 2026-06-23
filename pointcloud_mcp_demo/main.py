from utils.logger_utils import get_logger
from mcp_server.server import (
    APP_NAME,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TRANSPORT,
    mcp,
)

logger = get_logger(__name__)

def main():
    logger.info("Starting MCP server with app name: %s", APP_NAME)
    logger.info("Listening on %s:%d with transport: %s", DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TRANSPORT)

    try:
        mcp.run(transport=DEFAULT_TRANSPORT)
    except Exception:
        logger.exception("MCP server exited unexpectedly")
        raise

if __name__ == "__main__":
    main()
