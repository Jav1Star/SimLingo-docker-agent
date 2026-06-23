import os
import base64
import numpy as np

from typing import Any
from mcp.server.fastmcp import FastMCP

from utils.logger_utils import get_logger
from utils.perception_utils import fetch_info_from_pcd_dataset

logger = get_logger(__name__)

APP_NAME = os.getenv("MCP_APP_NAME", "pointcloud-mcp-demo")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "8123"))
DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

DATA_ROOT = os.getenv("POINTCLOUD_DATA_ROOT", "/home/t/Projects/gs/datasets/OPV2V/test_culver_city/")


mcp = FastMCP(
    APP_NAME,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
)


@mcp.resource("perception://{folder}/{subfolder}/{index}")
def fetch_perception_info(folder: str, subfolder: str, index: int) -> dict[str, Any]:
    """Read perception metadata from a `.pcd` file under the configured dataset root.

    This function is exposed as an MCP Resource at:
        perception://{folder}/{subfolder}/{index}

    URI template parameters:
        folder: Dataset subdirectory name under `POINTCLOUD_DATA_ROOT`.
        subfolder: Subdirectory name under the folder.
        index: Zero-based file index in sorted `.pcd` files of the subfolder.
               If index is larger than file count, modulo indexing is applied.

    Returns:
        On success, returns a JSON-serializable dict with fields like:
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
        On failure, returns:
            {"status": "error", "message": "..."}
    """
    logger.info("Received resource request for fetch_perception_info with folder=%s, subfolder=%s, index=%d", folder, subfolder, index)
    folder_path = os.path.join(DATA_ROOT, folder)
    subfolder_path = os.path.join(folder_path, subfolder)
    if not os.path.isdir(subfolder_path):
        message = f"Error in fetch_perception_info: Subfolder not found: {subfolder}"
        logger.error(message)
        return {"status": "error", "message": message}

    files = sorted([file_name for file_name in os.listdir(subfolder_path)
                                    if file_name.endswith(('.pcd'))])
    if index < 0:
        message = f"Error in fetch_perception_info: Index must be non-negative: {index}"
        logger.error(message)
        return {"status": "error", "message": message}
    
    index = index % len(files)  # 循环访问文件

    file_path = os.path.join(subfolder_path, files[index])
    if not os.path.isfile(file_path):
        message = f"Error in fetch_perception_info: File not found: index {index} in subfolder {subfolder}"
        logger.error(message)
        return {"status": "error", "message": message}

    perception_info = fetch_info_from_pcd_dataset(file_path)
    if isinstance(perception_info, dict) and perception_info.get("status") == "success":
        logger.info("Successfully fetched perception info for file: %s", file_path)

    return perception_info