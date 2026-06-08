from __future__ import annotations

import json

from typing import Any
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from utils.logger_utils import get_logger
from utils.numpy_utils import decode_array_from_dict

logger = get_logger(__name__)


class PointCloudMCPClient:
    """MCP client for reading resources from a remote point-cloud MCP server."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        protocol: str = "http",
        endpoint: str = "/mcp",
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    @property
    def server_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.endpoint}"

    async def fetch_perception_info(self, resource_uri: str) -> Any:
        """Fetch perception info from the MCP server for a given resource URI."""
        """
        On success, returns a dict with fields like:
            {
                "status": "success",
                "lidar_pose": list[float] | None,
                "ts_lidar_pose": int,
                "pcd": {
                    "data": str,   # base64-encoded point cloud bytes
                    "shape": tuple[int, ...] | None,
                    "dtype": str | None
                },
                "ts_pcd": int,
                "speed": list[float] | None,
                "ts_speed": int,
                "camera_infos": list[dict[str, Any]]  # optional
            }
        """

        logger.info(f"Fetching perception info for resource: {resource_uri}")
        async with streamable_http_client(self.server_url) as transport:
            if not isinstance(transport, tuple) or len(transport) < 2:
                raise RuntimeError("Unexpected streamable_http_client transport shape")

            read_stream, write_stream = transport[0], transport[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                resource_content = await session.read_resource(resource_uri)
                if not resource_content.contents:
                    message = "MCP server returned empty resource contents"
                    logger.error(message)
                    return {"status": "error", "message": message}

                content_block = resource_content.contents[0]
                if isinstance(content_block, types.TextResourceContents):
                    text_resource = content_block.text
                    perception_info = json.loads(text_resource)

                    return perception_info
                else:
                    message = f"Unsupported content block type: {type(content_block)}"
                    logger.error(message)
                    return {"status": "error", "message": message}