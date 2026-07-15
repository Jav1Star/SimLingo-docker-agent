from collections.abc import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
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
SCHEDULER_NUM_PREFIX_LAYERS = int(os.getenv("SCHEDULER_NUM_PREFIX_LAYERS", "10"))
RULE_BASED_CFG_JSON = os.getenv("SCHEDULER_RULE_BASED_CFG_JSON", "").strip() or None

NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://host.docker.internal:4222")
SCHEDULER_BUDGET_IN_SUBJECT = os.getenv(
    "SCHEDULER_BUDGET_IN_SUBJECT",
    os.getenv("NATS_IN_SUBJECT", "workflow.simlingo.scheduler_budget_input"),
)
SCHEDULER_BUDGET_IN_DURABLE = os.getenv(
    "SCHEDULER_BUDGET_IN_DURABLE",
    os.getenv("NATS_IN_DURABLE", "workflow-simlingo-scheduler-budget-input"),
)
SCHEDULER_BUDGET_OUT_SUBJECT = os.getenv(
    "SCHEDULER_BUDGET_OUT_SUBJECT",
    os.getenv("NATS_OUT_SUBJECT", "workflow.simlingo.scheduler_budget_output.llm_prefix_input"),
)
SCHEDULER_PLAN_IN_SUBJECT = os.getenv(
    "SCHEDULER_PLAN_IN_SUBJECT",
    "workflow.simlingo.llm_prefix_output.scheduler_plan_input",
)
SCHEDULER_PLAN_IN_DURABLE = os.getenv(
    "SCHEDULER_PLAN_IN_DURABLE",
    "workflow-simlingo-llm-prefix-output-scheduler-plan-input",
)
SCHEDULER_PLAN_OUT_SUBJECT = os.getenv(
    "SCHEDULER_PLAN_OUT_SUBJECT",
    "workflow.simlingo.scheduler_plan_output.llm_final_input",
)
SCHEDULER_DECISION_UPDATE_IN_SUBJECT = os.getenv(
    "SCHEDULER_DECISION_UPDATE_IN_SUBJECT",
    "workflow.simlingo.llm_final_output.scheduler_decision_update_input",
)
SCHEDULER_DECISION_UPDATE_IN_DURABLE = os.getenv(
    "SCHEDULER_DECISION_UPDATE_IN_DURABLE",
    "workflow-simlingo-llm-final-output-scheduler-decision-update-input",
)
SCHEDULER_DECISION_UPDATE_OUT_SUBJECT = os.getenv(
    "SCHEDULER_DECISION_UPDATE_OUT_SUBJECT",
    "workflow.simlingo.scheduler_decision_update_output",
)

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
    if scheduler_phase == "decision_update":
        return (
            SCHEDULER_DECISION_UPDATE_IN_SUBJECT,
            SCHEDULER_DECISION_UPDATE_IN_DURABLE,
            SCHEDULER_DECISION_UPDATE_OUT_SUBJECT,
        )
    if scheduler_phase == "plan":
        return SCHEDULER_PLAN_IN_SUBJECT, SCHEDULER_PLAN_IN_DURABLE, SCHEDULER_PLAN_OUT_SUBJECT
    return SCHEDULER_BUDGET_IN_SUBJECT, SCHEDULER_BUDGET_IN_DURABLE, SCHEDULER_BUDGET_OUT_SUBJECT


def _subject_tokens(subject: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    parts = [part for part in normalized.split("_") if part]
    tokens = set(parts)
    for size in (2, 3):
        tokens.update("_".join(parts[index : index + size]) for index in range(len(parts) - size + 1))
    return tokens


def _infer_scheduler_phase_from_subject(subject: str) -> str:
    if subject == SCHEDULER_BUDGET_IN_SUBJECT:
        return "budget"
    if subject == SCHEDULER_PLAN_IN_SUBJECT:
        return "plan"
    if subject == SCHEDULER_DECISION_UPDATE_IN_SUBJECT:
        return "decision_update"

    tokens = _subject_tokens(subject)
    if {"scheduler_budget", "budget_input", "encoded_tokens"} & tokens:
        return "budget"
    if {"scheduler_plan", "plan_input", "llm_prefix_output"} & tokens:
        return "plan"
    if {"scheduler_decision_update", "decision_update", "llm_final_output"} & tokens:
        return "decision_update"

    raise HTTPException(
        status_code=422,
        detail=(
            "Unable to infer scheduler phase from NATS subject "
            f"'{subject}'. Include scheduler_budget, scheduler_plan, or scheduler_decision_update in the subject name."
        ),
    )


async def _receive_data_from_nats(
    nats_in_subject: str,
    nats_in_durable: str,
) -> tuple[dict[str, Any], str]:
    try:
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
            return message.payload, message.subject
        raise HTTPException(
            status_code=504,
            detail=f"No messages found on subject '{nats_in_subject}'",
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
        "phase_source": "nats_subject",
        "budget_in_subject": SCHEDULER_BUDGET_IN_SUBJECT,
        "budget_out_subject": SCHEDULER_BUDGET_OUT_SUBJECT,
        "plan_in_subject": SCHEDULER_PLAN_IN_SUBJECT,
        "plan_out_subject": SCHEDULER_PLAN_OUT_SUBJECT,
        "decision_update_in_subject": SCHEDULER_DECISION_UPDATE_IN_SUBJECT,
        "decision_update_out_subject": SCHEDULER_DECISION_UPDATE_OUT_SUBJECT,
        "budget_policy": scheduler_runtime.policy_info() if scheduler_runtime.is_loaded else None,
    }


@app.post("/runtime/budget-policy")
async def update_budget_policy(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", "")).strip().lower()
    fixed_budget = payload.get("fixed_budget", 1.0)
    rule_based_cfg = payload.get("rule_based_cfg", payload.get("rule_based"))

    try:
        policy = await asyncio.to_thread(
            scheduler_runtime.configure_policy,
            mode=mode,
            fixed_budget=float(fixed_budget),
            rule_based_cfg=rule_based_cfg,
        )
    except ValueError as exc:
        logger.warning("Scheduler budget policy validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scheduler budget policy update failed")
        raise HTTPException(status_code=500, detail=f"Scheduler budget policy update failed: {exc}") from exc

    return {
        "status": "success",
        "budget_policy": policy,
    }


async def agent_function(
    nats_in_subject: str,
    nats_in_durable: str,
    nats_out_subject: str | None = None,
) -> dict[str, Any]:
    data, received_subject = await _receive_data_from_nats(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
    )
    scheduler_phase = _infer_scheduler_phase_from_subject(received_subject)
    _, _, default_out_subject = _phase_default_routes(scheduler_phase)
    nats_out_subject = nats_out_subject or default_out_subject
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

    if "nats_in_subject" in metadata and metadata.get("nats_in_subject"):
        nats_in_subject = metadata["nats_in_subject"]
        nats_in_durable = metadata.get("nats_in_durable") or nats_in_subject.replace(".", "-")
    else:
        default_in_subject, default_in_durable, _ = _phase_default_routes("budget")
        nats_in_subject = default_in_subject
        nats_in_durable = default_in_durable

    nats_out_subject = metadata.get("nats_out_subject")

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
        sender_id="SimLingoSchedulerAgent",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict(),
    )
    return response_message.dict()
