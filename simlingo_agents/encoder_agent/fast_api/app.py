from collections.abc import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from fast_api.model_runtime import encoder_runtime
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from utils.logger_utils import get_logger


logger = get_logger(__name__)

MODEL_VARIANT = os.getenv("ENCODER_MODEL_VARIANT", "/app/models/InternVL2-1B")
CHECKPOINT_PATH = os.getenv("ENCODER_CHECKPOINT_PATH", "").strip() or None
TOKEN_PRUNE_MODE = os.getenv("ENCODER_TOKEN_PRUNE_MODE", "origin")
TOKEN_PRUNE_RATIO = float(os.getenv("ENCODER_TOKEN_PRUNE_RATIO", "0.0"))
TOKEN_PRUNE_MIN_KEEP = int(os.getenv("ENCODER_TOKEN_PRUNE_MIN_KEEP", "1"))
NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://host.docker.internal:4222")
NATS_IN_SUBJECT = os.getenv("NATS_IN_SUBJECT", "workflow.previousagent.result")
NATS_IN_DURABLE = os.getenv("NATS_IN_DURABLE", "workflow-previousagent-result")
NATS_OUT_SUBJECT = os.getenv("NATS_OUT_SUBJECT", os.getenv("NATS_SUBJECT", "workflow.simlingo.scheduler_budget_input"))

_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await asyncio.to_thread(
            encoder_runtime.load_model,
            MODEL_VARIANT,
            checkpoint_path=CHECKPOINT_PATH,
            token_prune_mode=TOKEN_PRUNE_MODE,
            token_prune_ratio=TOKEN_PRUNE_RATIO,
            token_prune_min_keep=TOKEN_PRUNE_MIN_KEEP,
        )
        logger.info("Encoder model loaded successfully during startup")
    except Exception as exc:
        logger.exception("Failed to load encoder model during startup")
        raise RuntimeError(f"Startup model loading failed: {exc}") from exc
    try:
        await _nats_comm.start()
        logger.info("Encoder instance workflow stream is ready")
        yield
    finally:
        await _nats_comm.close()


app = FastAPI(title="SimLingo Encoder Agent API", lifespan=lifespan)


async def _receive_data_from_nats(
    nats_in_subject: str = NATS_IN_SUBJECT,
    nats_in_durable: str = NATS_IN_DURABLE,
) -> dict[str, Any]:
    try:
        _nats_comm.validate_own_subject(nats_in_subject)
        messages = await _nats_comm.receive(
            subject=nats_in_subject,
            durable=nats_in_durable,
            batch=1,
            timeout_sec=float(os.getenv("NATS_RECEIVE_TIMEOUT_SEC", "5")),
            ack=False,
        )
        if messages:
            message = messages[0]
            await message.ack()
            logger.info(
                "Received message on subject '%s' with durable '%s'",
                message.subject,
                nats_in_durable,
            )
            return message.payload
        raise HTTPException(
            status_code=504,
            detail=f"No messages found on subject '{nats_in_subject}'",
        )
    except Exception as exc:
        logger.exception("Error receiving message from NATS subject '%s'", nats_in_subject)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to receive message: {exc}") from exc


async def _send_data_to_nats(data: dict[str, Any], nats_out_subject: str = NATS_OUT_SUBJECT) -> None:
    ack = await _nats_comm.send(subject=nats_out_subject, payload=data)
    logger.info("Data sent to NATS subject '%s' with ack: %s", nats_out_subject, ack)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": "simlingo-encoder-agent",
        "model_loaded": encoder_runtime.is_loaded,
        "device": encoder_runtime.device,
        "nats_server_url": NATS_SERVER_URL,
        "nats_in_subject": NATS_IN_SUBJECT,
        "nats_out_subject": NATS_OUT_SUBJECT,
        "token_prune": {
            "mode": TOKEN_PRUNE_MODE,
            "prune_ratio": TOKEN_PRUNE_RATIO,
            "min_keep": TOKEN_PRUNE_MIN_KEEP,
        },
    }


async def agent_function(
    nats_in_subject: str = NATS_IN_SUBJECT,
    nats_in_durable: str = NATS_IN_DURABLE,
    nats_out_subject: str = NATS_OUT_SUBJECT,
) -> dict[str, Any]:
    data = await _receive_data_from_nats(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
    )
    try:
        encoded_result = await asyncio.to_thread(encoder_runtime.encode, data)
    except ValueError as exc:
        logger.warning("Encoder request validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Encoder runtime failed")
        raise HTTPException(status_code=500, detail=f"Encoder failed: {exc}") from exc

    result = {
        "status": "success",
        "encoded_payload": encoded_result,
    }
    await _send_data_to_nats(result, nats_out_subject=nats_out_subject)
    return {
        "status": "success",
        "frame_id": encoded_result.get("frame_id"),
    }


@app.post("/a2a/execute")
async def agent_execute(message: dict[str, Any]) -> dict[str, Any]:
    logger.info("Received message: %s", message)
    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)
    metadata = getattr(task_request, "metadata", {}) or {}

    if "nats_in_subject" in metadata and metadata.get("nats_in_subject"):
        nats_in_subject = metadata["nats_in_subject"]
        nats_in_durable = metadata.get("nats_in_durable") or nats_in_subject.replace(".", "-")
    else:
        nats_in_subject = NATS_IN_SUBJECT
        nats_in_durable = NATS_IN_DURABLE

    nats_out_subject = metadata.get("nats_out_subject") or NATS_OUT_SUBJECT

    result = await agent_function(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
        nats_out_subject=nats_out_subject,
    )
    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "unknown"),
        result=json.dumps(result),
    )
    response_message = A2AMessage(
        sender_id="SimLingoEncoderAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict(),
    )
    return response_message.dict()
