"""Shared profiling and trace helpers for the split SimLingo agents."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def profiling_enabled() -> bool:
    return os.getenv("SIMLINGO_EVAL_RECORD_TFLOPS", "1").strip().lower() not in {
        "0", "false", "no", "off", ""
    }


def _find_context(payload: dict[str, Any]) -> tuple[Any, Any, Any]:
    candidates = [payload]
    for key in ("encoded_payload", "llm_payload"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for item in candidates:
        runtime_context = item.get("runtime_context")
        request_id = item.get("pipeline_request_id")
        if isinstance(runtime_context, dict):
            request_id = request_id or runtime_context.get("pipeline_request_id")
        route_key = item.get("route_key")
        frame_id = item.get("frame_id")
        if route_key is not None or frame_id is not None or request_id is not None:
            return route_key, frame_id, request_id
    return None, None, None


def extract_profile_trace(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    trace = payload.get("profile_trace")
    if isinstance(trace, list):
        return [copy.deepcopy(item) for item in trace if isinstance(item, dict)]
    for key in ("encoded_payload", "llm_payload"):
        nested = payload.get(key)
        nested_trace = extract_profile_trace(nested)
        if nested_trace:
            return nested_trace
    return []


def extract_nats_communication_trace(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    trace = payload.get("nats_communication_trace")
    if isinstance(trace, list):
        return [copy.deepcopy(item) for item in trace if isinstance(item, dict)]
    for key in ("encoded_payload", "llm_payload"):
        nested = payload.get(key)
        nested_trace = extract_nats_communication_trace(nested)
        if nested_trace:
            return nested_trace
    return []


def profile_call(
    callback: Callable[[], T], *, agent: str, phase: str, payload: dict[str, Any]
) -> tuple[T, dict[str, Any]]:
    import torch

    route_key, frame_id, request_id = _find_context(payload)
    record: dict[str, Any] = {
        "agent": agent,
        "phase": phase,
        "route_key": route_key,
        "frame_id": frame_id,
        "pipeline_request_id": request_id,
        "profile_success": False,
        "flops": 0,
        "latency_ms": None,
        "error": None,
    }
    if not profiling_enabled():
        record["error"] = "profiling_disabled"
        return callback(), record

    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_available = torch.cuda.is_available()
    start_event = end_event = None
    callback_completed = False
    result: T
    try:
        if cuda_available:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            result = callback()
            callback_completed = True
        if cuda_available:
            end_event.record()
            torch.cuda.synchronize()
            record["latency_ms"] = float(start_event.elapsed_time(end_event))
        record["flops"] = sum(
            int(getattr(event, "flops", 0) or 0)
            for event in profiler.key_averages()
        )
        record["profile_success"] = True
        return result, record
    except Exception as exc:
        record["error"] = str(exc)
        # Never repeat a possibly stateful scheduler/model call. If the model
        # returned and only profiler finalization failed, preserve its result
        # and mark this phase as unprofiled. Model failures still propagate.
        if callback_completed:
            return result, record
        raise


def attach_traces(
    result: dict[str, Any], source_payload: dict[str, Any], profile_record: dict[str, Any]
) -> dict[str, Any]:
    result["profile_trace"] = extract_profile_trace(source_payload) + [profile_record]
    result["nats_communication_trace"] = extract_nats_communication_trace(source_payload)
    return result


def mark_nats_send(payload: dict[str, Any], *, sender: str, receiver: str, link: str) -> None:
    payload["_nats_handoff"] = {
        "sender": sender,
        "receiver": receiver,
        "link": link,
        "sent_wall_ns": time.time_ns(),
    }


def record_nats_receive(payload: dict[str, Any], *, receiver: str) -> None:
    handoff = payload.pop("_nats_handoff", None)
    if not isinstance(handoff, dict):
        return
    sent_ns = handoff.get("sent_wall_ns")
    try:
        latency_ms = max(0.0, (time.time_ns() - int(sent_ns)) / 1_000_000.0)
    except (TypeError, ValueError):
        return
    trace = extract_nats_communication_trace(payload)
    route_key, frame_id, request_id = _find_context(payload)
    trace.append(
        {
            "sender": handoff.get("sender"),
            "receiver": receiver,
            "link": handoff.get("link"),
            "route_key": route_key,
            "frame_id": frame_id,
            "pipeline_request_id": request_id,
            "latency_ms": latency_ms,
            "clock": "time.time_ns",
        }
    )
    payload["nats_communication_trace"] = trace
