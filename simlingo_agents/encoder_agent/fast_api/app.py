from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from fast_api.model_runtime import encoder_runtime, save_artifacts
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from utils.logger_utils import get_logger
from utils.numpy_utils import decode_numpy_payload, encode_numpy_payload


logger = get_logger(__name__)

MODEL_VARIANT = os.getenv("ENCODER_MODEL_VARIANT", "/app/models/InternVL2-1B")
CHECKPOINT_PATH = os.getenv("ENCODER_CHECKPOINT_PATH", "").strip() or None
ARTIFACT_DIR = os.getenv("ENCODER_ARTIFACT_DIR", "/app/artifacts")
NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://host.docker.internal:4222")
NATS_SUBJECT = os.getenv("NATS_SUBJECT", "workflow.simlingo.encoded_tokens")
PUBLISH_BY_DEFAULT = os.getenv("ENCODER_PUBLISH_BY_DEFAULT", "true").strip().lower() in {"1", "true", "yes", "on"}

_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        encoder_runtime.load_model(MODEL_VARIANT, checkpoint_path=CHECKPOINT_PATH)
        try:
            await _nats_comm.connect()
        except Exception as exc:
            logger.warning("NATS is unavailable at startup: %s", exc)
        yield
    finally:
        await _nats_comm.close()


app = FastAPI(title="SimLingo Encoder Agent API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": "simlingo-encoder-agent",
        "model_loaded": encoder_runtime.is_loaded,
        "device": encoder_runtime.device,
    }


@app.post("/encoder/encode")
async def encode(payload: dict[str, Any]) -> dict[str, Any]:
    """Encode prompt/images into token ids and dense encoder outputs.

    By default large arrays are written to ENCODER_ARTIFACT_DIR and returned as
    file URIs. Set inline_payload=true for small debugging requests.
    """
    try:
        decoded_payload = decode_numpy_payload(payload)
        encoded = encoder_runtime.encode(decoded_payload)
        frame_id = str(encoded.get("frame_id") or uuid.uuid4().hex)
        inline_payload = _as_bool(payload.get("inline_payload"), default=False)
        if inline_payload:
            response_payload = encode_numpy_payload(encoded)
        else:
            response_payload = save_artifacts(encoded, ARTIFACT_DIR, frame_id)

        nats_subject = payload.get("nats_subject", NATS_SUBJECT)
        publish = _as_bool(payload.get("publish"), default=PUBLISH_BY_DEFAULT)
        ack = None
        if publish:
            ack = await _nats_comm.send(subject=nats_subject, payload=response_payload)

        return {
            "status": "success",
            "agent": "simlingo-encoder-agent",
            "frame_id": frame_id,
            "published": publish,
            "nats_ack": ack,
            "payload": response_payload,
        }
    except Exception as exc:
        logger.exception("Encoder forward failed")
        raise HTTPException(status_code=500, detail=f"Encoder failed: {exc}") from exc


@app.post("/a2a/execute")
async def execute_a2a(message: dict[str, Any]) -> dict[str, Any]:
    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)
    metadata = task_request.metadata or {}
    context = dict(task_request.context or {})
    if metadata.get("nats_subject"):
        context["nats_subject"] = metadata["nats_subject"]
    if "publish" not in context:
        context["publish"] = True

    result = await encode(context)
    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "error"),
        result=result,
    )
    response_message = A2AMessage(
        sender_id="SimLingoEncoderAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.model_dump(),
        correlation_id=request_message.correlation_id,
    )
    return response_message.model_dump()
