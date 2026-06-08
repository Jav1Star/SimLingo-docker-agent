from collections.abc import Generator
from contextlib import asynccontextmanager
import os
import threading
from typing import Any

import json
import numpy as np
from fastapi import FastAPI, HTTPException

from utils.logger_utils import get_logger
from utils.numpy_utils import encode_array_to_dict, decode_array_from_dict
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from fast_api.model_runtime import model_runtime
from mcp_clients.pointcloud_mcp_clients import PointCloudMCPClient

logger = get_logger(__name__)

# Configurable model checkpoint path for container/local deployment.
FIXED_MODEL_CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",
    "checkpoints/point_pillar_where2comm/",
)

FRONTEND_CALLBACK_URL = os.getenv(
    "FRONTEND_CALLBACK_URL",
    "http://localhost:9002/temp/post_data",
)

MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8123"))
RESOURCE_URI_PREFIX = "perception://2021_09_03_09_32_17/302"

NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://nats:4222")
NATS_SUBJECT = os.getenv("NATS_SUBJECT", "workflow.demo.perception2feature.result")

pointcloud_mcp_client = PointCloudMCPClient(host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)

logger.info("NATS communication initialized with server: %s", NATS_SERVER_URL)
_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


_resource_index = 0
_resource_index_lock = threading.Lock()


def _next_resource_uri() -> str:
    global _resource_index
    with _resource_index_lock:
        current_index = _resource_index
        _resource_index += 1
    return f"{RESOURCE_URI_PREFIX}/{current_index}"


def _encode_numpy_payload(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return encode_array_to_dict(payload)

    if isinstance(payload, dict):
        return {key: _encode_numpy_payload(value) for key, value in payload.items()}

    if isinstance(payload, list):
        return [_encode_numpy_payload(item) for item in payload]

    return payload

# def _temp_post_data(data: dict) -> None:
#     """临时函数: 将数据发送到前端指定接口, 供调试使用."""
#     import requests

#     try:
#         response = requests.post(FRONTEND_CALLBACK_URL, json=data, timeout=5)
#         response.raise_for_status()
#         logger.info("Data posted to frontend successfully")
#     except Exception as exc:
#         logger.error(f"Failed to post data to frontend: {exc}")

async def _send_data(data: dict, nats_subject: str = NATS_SUBJECT) -> None:
    ack = await _nats_comm.send(
        subject=nats_subject,
        payload=data,
    )
    logger.info(f"Data sent to NATS subject '{nats_subject}' with ack: {ack}")

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model once when app starts."""
    try:
        model_runtime.load_model(FIXED_MODEL_CHECKPOINT_PATH)
        logger.info("Model loaded from fixed path: %s", FIXED_MODEL_CHECKPOINT_PATH)
        await _nats_comm.connect()
    except FileNotFoundError as exc:
        # Keep service alive in demo mode even when fixed path is not present.
        logger.error("Fixed checkpoint missing (%s), loading built-in demo weights", exc)
    except Exception as exc:
        logger.exception("Failed to load model during startup")
        raise RuntimeError(f"Startup model loading failed: {exc}") from exc

    try:
        yield
    finally:
        await _nats_comm.close()


app = FastAPI(title="Perception to Feature Agent Demo API", lifespan=lifespan)


# @app.get("/model/forward")
@app.get("/model/forward")
async def model_forward(nats_subject: str = NATS_SUBJECT) -> dict:
    try:
        resource_uri = _next_resource_uri()
        perception_info = await pointcloud_mcp_client.fetch_perception_info(resource_uri)

        if not isinstance(perception_info, dict):
            raise RuntimeError("perception_info is not a dict")

        if perception_info.get("status") == "error":
            message = perception_info.get("message", "Unknown MCP error")
            raise RuntimeError(f"MCP resource read failed: {message}")

        pointcloud = perception_info.get("pcd")

        if isinstance(pointcloud, dict) and pointcloud.get("data") is not None:
            try:
                pointcloud = decode_array_from_dict(pointcloud)
            except Exception as exc:
                logger.error(f"Failed to decode point cloud data: {exc}")
                pointcloud = None
        else:
            pointcloud = None

        if not isinstance(pointcloud, np.ndarray):
            raise RuntimeError("pcd is missing or not a decoded numpy array")

        intermediate_feature = model_runtime.pointcloud_inference(pointcloud=pointcloud)
    except Exception as exc:
        logger.exception("Forward inference failed")
        raise HTTPException(status_code=500, detail=f"Forward failed: {exc}") from exc
    
    data = {
        "status": "success",
        "resource_uri": resource_uri,
        "intermediate_feature": _encode_numpy_payload(intermediate_feature),
        "pcd": perception_info.get("pcd"),
    }

    # _temp_post_data(data)
    await _send_data(data, nats_subject=nats_subject)

    return {
        "status": "success",
        "resource_uri": resource_uri,
    }

@app.post("/a2a/execute")
async def model_forward_with_input(message: dict) -> dict:
    logger.info("Received message: %s", message)

    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)
    
    # 优先使用 metadata 中的 nats_subject/nats_durable
    metadata = getattr(task_request, 'metadata', {}) or {}
    if 'nats_subject' in metadata and metadata.get('nats_subject'):
        nats_subject = metadata['nats_subject']
    else:
        # metadata 中没有指定 nats_subject，则回退到默认的 SUBJECT 和 DURABLE
        nats_subject = NATS_SUBJECT

    result = await model_forward(nats_subject=nats_subject)

    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "unknown"),
        result=json.dumps(result)
    )

    response_message = A2AMessage(
        sender_id="Perception2IntermediateFeatureAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict()
    )

    # return json.dumps(response_message.dict())
    return response_message.dict()
