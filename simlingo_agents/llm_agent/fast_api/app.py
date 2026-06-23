from collections.abc import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from fast_api.model_runtime import llm_runtime
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from utils.logger_utils import get_logger
from utils.numpy_utils import decode_structured_numpy, encode_structured_numpy


logger = get_logger(__name__)

MODEL_VARIANT = os.getenv("LLM_MODEL_VARIANT", "/app/models/InternVL2-1B")
CHECKPOINT_PATH = os.getenv("LLM_CHECKPOINT_PATH", "").strip() or None
SPEED_WPS_MODE = os.getenv("LLM_SPEED_WPS_MODE", "2d")
PREDICT_ROUTE_AS_WPS = os.getenv("LLM_PREDICT_ROUTE_AS_WPS", "true").strip().lower() in {"1", "true", "yes", "on"}
USE_LORA = os.getenv("LLM_LORA", "true").strip().lower() in {"1", "true", "yes", "on"}
ADAPTION_TRAIN = os.getenv("LLM_ADAPTION_TRAIN", "false").strip().lower() in {"1", "true", "yes", "on"}
LORA_ALPHA = int(os.getenv("LLM_LORA_ALPHA", "64"))
LORA_R = int(os.getenv("LLM_LORA_R", "32"))
LORA_DROPOUT = float(os.getenv("LLM_LORA_DROPOUT", "0.1"))
NUM_PREFIX_LAYERS = int(os.getenv("LLM_NUM_PREFIX_LAYERS", "2"))
LLM_PHASE = os.getenv("LLM_PHASE", "prefix").strip().lower()
LLM_SCHEDULER_TARGET = os.getenv(
    "LLM_SCHEDULER_TARGET",
    "simlingo_adaption_training.models.scheduler.simple_scheduler.SimpleScheduler_L",
)
LLM_SCHEDULER_TAU = float(os.getenv("LLM_SCHEDULER_TAU", "5"))
LLM_SCHEDULER_IS_HARD = os.getenv("LLM_SCHEDULER_IS_HARD", "true").strip().lower() in {"1", "true", "yes", "on"}
LLM_SCHEDULER_THRESHOLD = float(os.getenv("LLM_SCHEDULER_THRESHOLD", "0.5"))
LLM_SCHEDULER_BIAS = os.getenv("LLM_SCHEDULER_BIAS", "true").strip().lower() in {"1", "true", "yes", "on"}

NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://host.docker.internal:4222")
LLM_PREFIX_IN_SUBJECT = os.getenv("LLM_PREFIX_IN_SUBJECT", os.getenv("NATS_IN_SUBJECT", "workflow.simlingo.llm_prefix_input"))
LLM_PREFIX_IN_DURABLE = os.getenv("LLM_PREFIX_IN_DURABLE", os.getenv("NATS_IN_DURABLE", "workflow-simlingo-llm-prefix-input"))
LLM_PREFIX_OUT_SUBJECT = os.getenv("LLM_PREFIX_OUT_SUBJECT", os.getenv("NATS_OUT_SUBJECT", "workflow.simlingo.llm_prefix_output"))
LLM_FINAL_IN_SUBJECT = os.getenv("LLM_FINAL_IN_SUBJECT", "workflow.simlingo.llm_final_input")
LLM_FINAL_IN_DURABLE = os.getenv("LLM_FINAL_IN_DURABLE", "workflow-simlingo-llm-final-input")
LLM_FINAL_OUT_SUBJECT = os.getenv("LLM_FINAL_OUT_SUBJECT", "workflow.simlingo.llm_output")

_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await asyncio.to_thread(
            llm_runtime.load_model,
            MODEL_VARIANT,
            checkpoint_path=CHECKPOINT_PATH,
            speed_wps_mode=SPEED_WPS_MODE,
            predict_route_as_wps=PREDICT_ROUTE_AS_WPS,
            use_lora=USE_LORA,
            adaption_train=ADAPTION_TRAIN,
            lora_alpha=LORA_ALPHA,
            lora_r=LORA_R,
            lora_dropout=LORA_DROPOUT,
            num_prefix_layers=NUM_PREFIX_LAYERS,
            scheduler_target=LLM_SCHEDULER_TARGET,
            scheduler_tau=LLM_SCHEDULER_TAU,
            scheduler_is_hard=LLM_SCHEDULER_IS_HARD,
            scheduler_threshold=LLM_SCHEDULER_THRESHOLD,
            scheduler_bias=LLM_SCHEDULER_BIAS,
        )
        logger.info("LLM model loaded successfully during startup")
    except Exception as exc:
        logger.exception("Failed to load LLM model during startup")
        raise RuntimeError(f"Startup model loading failed: {exc}") from exc
    try:
        yield
    finally:
        await _nats_comm.close()


app = FastAPI(title="SimLingo LLM Agent API", lifespan=lifespan)


def _phase_default_routes(llm_phase: str) -> tuple[str, str, str]:
    if llm_phase == "final":
        return LLM_FINAL_IN_SUBJECT, LLM_FINAL_IN_DURABLE, LLM_FINAL_OUT_SUBJECT
    return LLM_PREFIX_IN_SUBJECT, LLM_PREFIX_IN_DURABLE, LLM_PREFIX_OUT_SUBJECT


async def _receive_data_from_nats(
    nats_in_subject: str,
    nats_in_durable: str,
) -> dict[str, Any]:
    try:
        messages = await _nats_comm.receive(
            subject=nats_in_subject,
            durable=nats_in_durable,
            batch=1,
            timeout_sec=5,
        )
        for message in messages:
            logger.info("Received message on subject '%s'", nats_in_subject)
            await message.ack()
            return message.payload
        raise HTTPException(
            status_code=504,
            detail=f"No messages received on subject '{nats_in_subject}' within timeout",
        )
    except Exception as exc:
        logger.exception("Error receiving message from NATS subject '%s'", nats_in_subject)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Failed to receive message: {exc}") from exc


async def _send_data_to_nats(data: dict[str, Any], nats_out_subject: str) -> None:
    ack = await _nats_comm.send(subject=nats_out_subject, payload=data)
    logger.info("Data sent to NATS subject '%s' with ack: %s", nats_out_subject, ack)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "agent": "simlingo-llm-agent",
        "model_loaded": llm_runtime.is_loaded,
        "device": llm_runtime.device,
        "nats_server_url": NATS_SERVER_URL,
        "llm_phase": LLM_PHASE,
        "prefix_in_subject": LLM_PREFIX_IN_SUBJECT,
        "prefix_out_subject": LLM_PREFIX_OUT_SUBJECT,
        "final_in_subject": LLM_FINAL_IN_SUBJECT,
        "final_out_subject": LLM_FINAL_OUT_SUBJECT,
    }


async def agent_function(
    nats_in_subject: str,
    nats_in_durable: str,
    nats_out_subject: str,
    llm_phase: str = LLM_PHASE,
) -> dict[str, Any]:
    data = await _receive_data_from_nats(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
    )
    decoded_data = decode_structured_numpy(data)

    try:
        llm_result = await asyncio.to_thread(llm_runtime.run, decoded_data, llm_phase)
    except ValueError as exc:
        logger.warning("LLM request validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("LLM runtime failed")
        raise HTTPException(status_code=500, detail=f"LLM forward failed: {exc}") from exc

    outbound_payload = encode_structured_numpy(llm_result)
    await _send_data_to_nats(outbound_payload, nats_out_subject=nats_out_subject)
    return {
        "status": "success",
        "frame_id": llm_result.get("frame_id"),
        "route_key": llm_result.get("route_key"),
        "phase": llm_result.get("phase"),
    }


@app.post("/a2a/execute")
async def agent_execute(message: dict[str, Any]) -> dict[str, Any]:
    logger.info("Received message: %s", message)
    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)
    metadata = getattr(task_request, "metadata", {}) or {}

    llm_phase = str(metadata.get("llm_phase") or LLM_PHASE).strip().lower()
    default_in_subject, default_in_durable, default_out_subject = _phase_default_routes(llm_phase)

    if "nats_in_subject" in metadata and metadata.get("nats_in_subject"):
        nats_in_subject = metadata["nats_in_subject"]
        nats_in_durable = metadata.get("nats_in_durable") or nats_in_subject.replace(".", "-")
    else:
        nats_in_subject = default_in_subject
        nats_in_durable = default_in_durable

    nats_out_subject = metadata.get("nats_out_subject") or default_out_subject

    result = await agent_function(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
        nats_out_subject=nats_out_subject,
        llm_phase=llm_phase,
    )
    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "unknown"),
        result=json.dumps(result),
    )
    response_message = A2AMessage(
        sender_id="SimLingoLLMAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict(),
    )
    return response_message.dict()
