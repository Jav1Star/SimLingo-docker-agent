from collections.abc import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from fast_api.model_runtime import scheduler_runtime
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from utils.logger_utils import get_logger
from utils.numpy_utils import decode_structured_numpy, encode_structured_numpy


logger = get_logger(__name__)

MODEL_VARIANT = os.getenv("SCHEDULER_MODEL_VARIANT", "/app/models/InternVL2-1B")
CHECKPOINT_PATH = os.getenv("SCHEDULER_CHECKPOINT_PATH", "").strip() or None
MODE = os.getenv("SCHEDULER_MODE", "rule_based")
FIXED_BUDGET = float(os.getenv("SCHEDULER_FIXED_BUDGET", "0.8"))
HISTORY_ALPHA = float(os.getenv("SCHEDULER_HISTORY_ALPHA", "0.6"))
DECISION_SHIFT_T_LAP = float(os.getenv("SCHEDULER_DECISION_SHIFT_T_LAP", "0.2"))
SCHEDULER_TARGET = os.getenv(
    "SCHEDULER_TARGET",
    "simlingo_adaption_training.models.scheduler.simple_scheduler.SimpleScheduler_L",
)
SCHEDULER_TAU = float(os.getenv("SCHEDULER_TAU", "5"))
SCHEDULER_IS_HARD = os.getenv("SCHEDULER_IS_HARD", "true").strip().lower() in {"1", "true", "yes", "on"}
SCHEDULER_THRESHOLD = float(os.getenv("SCHEDULER_THRESHOLD", "0.5"))
SCHEDULER_BIAS = os.getenv("SCHEDULER_BIAS", "true").strip().lower() in {"1", "true", "yes", "on"}
SCHEDULER_NUM_PREFIX_LAYERS = int(os.getenv("SCHEDULER_NUM_PREFIX_LAYERS", "2"))
RULE_BASED_CFG_JSON = os.getenv("SCHEDULER_RULE_BASED_CFG_JSON", "").strip() or None
SCHEDULER_PHASE = os.getenv("SCHEDULER_PHASE", "budget").strip().lower()

NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://host.docker.internal:4222")
SCHEDULER_BUDGET_IN_SUBJECT = os.getenv(
    "SCHEDULER_BUDGET_IN_SUBJECT",
    os.getenv("NATS_IN_SUBJECT", "workflow.simlingo.encoded_tokens"),
)
SCHEDULER_BUDGET_IN_DURABLE = os.getenv(
    "SCHEDULER_BUDGET_IN_DURABLE",
    os.getenv("NATS_IN_DURABLE", "workflow-simlingo-encoded-tokens"),
)
SCHEDULER_BUDGET_OUT_SUBJECT = os.getenv(
    "SCHEDULER_BUDGET_OUT_SUBJECT",
    os.getenv("NATS_OUT_SUBJECT", "workflow.simlingo.llm_prefix_input"),
)
SCHEDULER_PLAN_IN_SUBJECT = os.getenv("SCHEDULER_PLAN_IN_SUBJECT", "workflow.simlingo.llm_prefix_output")
SCHEDULER_PLAN_IN_DURABLE = os.getenv("SCHEDULER_PLAN_IN_DURABLE", "workflow-simlingo-llm-prefix-output")
SCHEDULER_PLAN_OUT_SUBJECT = os.getenv("SCHEDULER_PLAN_OUT_SUBJECT", "workflow.simlingo.llm_final_input")

_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await asyncio.to_thread(
            scheduler_runtime.load_model,
            MODEL_VARIANT,
            checkpoint_path=CHECKPOINT_PATH,
            mode=MODE,
            fixed_budget=FIXED_BUDGET,
            history_alpha=HISTORY_ALPHA,
            decision_shift_t_lap=DECISION_SHIFT_T_LAP,
            scheduler_target=SCHEDULER_TARGET,
            tau=SCHEDULER_TAU,
            is_hard=SCHEDULER_IS_HARD,
            threshold=SCHEDULER_THRESHOLD,
            bias=SCHEDULER_BIAS,
            num_prefix_layers=SCHEDULER_NUM_PREFIX_LAYERS,
            rule_based_cfg_json=RULE_BASED_CFG_JSON,
        )
        logger.info("Scheduler model loaded successfully during startup")
    except Exception as exc:
        logger.exception("Failed to load scheduler model during startup")
        raise RuntimeError(f"Startup scheduler loading failed: {exc}") from exc
    try:
        yield
    finally:
        await _nats_comm.close()


app = FastAPI(title="SimLingo Scheduler Agent API", lifespan=lifespan)


def _phase_default_routes(scheduler_phase: str) -> tuple[str, str, str]:
    if scheduler_phase == "plan":
        return SCHEDULER_PLAN_IN_SUBJECT, SCHEDULER_PLAN_IN_DURABLE, SCHEDULER_PLAN_OUT_SUBJECT
    return SCHEDULER_BUDGET_IN_SUBJECT, SCHEDULER_BUDGET_IN_DURABLE, SCHEDULER_BUDGET_OUT_SUBJECT


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
        "agent": "simlingo-scheduler-agent",
        "model_loaded": scheduler_runtime.is_loaded,
        "device": scheduler_runtime.device,
        "nats_server_url": NATS_SERVER_URL,
        "scheduler_phase": SCHEDULER_PHASE,
        "budget_in_subject": SCHEDULER_BUDGET_IN_SUBJECT,
        "budget_out_subject": SCHEDULER_BUDGET_OUT_SUBJECT,
        "plan_in_subject": SCHEDULER_PLAN_IN_SUBJECT,
        "plan_out_subject": SCHEDULER_PLAN_OUT_SUBJECT,
    }


async def agent_function(
    nats_in_subject: str,
    nats_in_durable: str,
    nats_out_subject: str,
    scheduler_phase: str = SCHEDULER_PHASE,
) -> dict[str, Any]:
    data = await _receive_data_from_nats(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
    )
    decoded_data = decode_structured_numpy(data)

    try:
        scheduler_result = await asyncio.to_thread(scheduler_runtime.run, decoded_data, scheduler_phase)
    except ValueError as exc:
        logger.warning("Scheduler request validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scheduler runtime failed")
        raise HTTPException(status_code=500, detail=f"Scheduler failed: {exc}") from exc

    outbound_payload = encode_structured_numpy(scheduler_result)
    await _send_data_to_nats(outbound_payload, nats_out_subject=nats_out_subject)
    return {
        "status": "success",
        "frame_id": scheduler_result.get("encoded_payload", {}).get("frame_id"),
        "route_key": scheduler_result.get("encoded_payload", {}).get("route_key"),
        "phase": scheduler_result.get("phase"),
    }


@app.post("/a2a/execute")
async def agent_execute(message: dict[str, Any]) -> dict[str, Any]:
    logger.info("Received message: %s", message)
    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)
    metadata = getattr(task_request, "metadata", {}) or {}

    scheduler_phase = str(metadata.get("scheduler_phase") or SCHEDULER_PHASE).strip().lower()
    default_in_subject, default_in_durable, default_out_subject = _phase_default_routes(scheduler_phase)

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
        scheduler_phase=scheduler_phase,
    )
    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "unknown"),
        result=json.dumps(result),
    )
    response_message = A2AMessage(
        sender_id="SimLingoSchedulerAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict(),
    )
    return response_message.dict()
