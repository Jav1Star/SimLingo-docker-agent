from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from simlingo_agents.common.inference_profile import (
    extract_nats_communication_trace,
    extract_profile_trace,
    mark_nats_send,
    record_nats_receive,
)

class RemoteInferenceError(RuntimeError):
    """Raised when the split-agent pipeline fails."""


@dataclass(frozen=True)
class PipelineEndpoints:
    encoder_url: str
    scheduler_url: str
    llm_url: str


@dataclass(frozen=True)
class AgentInstance:
    cluster_id: str
    agent_id: str
    instance_id: str


def split_stack_profile() -> str:
    profile = os.getenv("SIMLINGO_SPLIT_STACK_PROFILE", "auto").strip().lower()
    if profile in {"k8s", "kubernetes", "cluster"}:
        return "k8s"
    if profile in {"nodeport", "node_port"}:
        return "nodeport"
    if profile in {"local", "localhost"}:
        return "local"
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "k8s"
    return "local"


def default_pipeline_endpoints() -> PipelineEndpoints:
    profile = split_stack_profile()
    if profile == "k8s":
        return PipelineEndpoints(
            encoder_url="http://simlingo-encoder-service:9011/a2a/execute",
            scheduler_url="http://simlingo-scheduler-service:9013/a2a/execute",
            llm_url="http://simlingo-llm-service:9012/a2a/execute",
        )
    if profile == "nodeport":
        host = os.getenv("SIMLINGO_K8S_NODE_HOST", os.getenv("K8S_NODE_HOST", "127.0.0.1"))
        return PipelineEndpoints(
            encoder_url=f"http://{host}:{os.getenv('SIMLINGO_ENCODER_NODEPORT', '30111')}/a2a/execute",
            scheduler_url=f"http://{host}:{os.getenv('SIMLINGO_SCHEDULER_NODEPORT', '30113')}/a2a/execute",
            llm_url=f"http://{host}:{os.getenv('SIMLINGO_LLM_NODEPORT', '30112')}/a2a/execute",
        )
    return PipelineEndpoints(
        encoder_url="http://127.0.0.1:9011/a2a/execute",
        scheduler_url="http://127.0.0.1:9013/a2a/execute",
        llm_url="http://127.0.0.1:9012/a2a/execute",
    )


def configured_pipeline_endpoints() -> PipelineEndpoints:
    defaults = default_pipeline_endpoints()
    return PipelineEndpoints(
        encoder_url=os.getenv("SIMLINGO_ENCODER_AGENT_URL", defaults.encoder_url),
        scheduler_url=os.getenv("SIMLINGO_SCHEDULER_AGENT_URL", defaults.scheduler_url),
        llm_url=os.getenv("SIMLINGO_LLM_AGENT_URL", defaults.llm_url),
    )


def default_nats_server_url() -> str:
    profile = split_stack_profile()
    if profile == "k8s":
        return "nats://nats:4222"
    return f"nats://{os.getenv('SIMLINGO_NATS_HOST', '127.0.0.1')}:{os.getenv('SIMLINGO_NATS_PORT', '4222')}"


def configured_nats_server_url() -> str:
    return os.getenv("NATS_SERVER_URL", default_nats_server_url())


def configured_jetstream_domain() -> str:
    if "NATS_JETSTREAM_DOMAIN" in os.environ:
        return os.environ["NATS_JETSTREAM_DOMAIN"]
    return os.getenv("CLUSTER_ID", "")


def configured_agent_instances() -> dict[str, AgentInstance]:
    cluster_id = os.getenv("CLUSTER_ID", "").strip()
    if not cluster_id:
        raise RemoteInferenceError("CLUSTER_ID is required for instance NATS routing")
    definitions = {
        "encoder": ("simlingo-encoder", "SIMLINGO_ENCODER_INSTANCE_ID"),
        "scheduler": ("simlingo-scheduler", "SIMLINGO_SCHEDULER_INSTANCE_ID"),
        "llm": ("simlingo-llm", "SIMLINGO_LLM_INSTANCE_ID"),
    }
    instances: dict[str, AgentInstance] = {}
    for role, (default_agent_id, instance_env) in definitions.items():
        instance_id = os.getenv(instance_env, "").strip()
        if not instance_id:
            raise RemoteInferenceError(f"{instance_env} is required")
        agent_id = os.getenv(
            f"SIMLINGO_{role.upper()}_AGENT_ID", default_agent_id
        ).strip()
        instances[role] = AgentInstance(cluster_id, agent_id, instance_id)
    return instances


class SplitAgentPipelineClient:
    """Bridges one Bench2Drive frame to the split encoder/scheduler/llm agents."""

    def __init__(
        self,
        *,
        endpoints: PipelineEndpoints | None = None,
        nats_server_url: str | None = None,
        nats_stream: str | None = None,
        nats_stream_subjects: list[str] | None = None,
        nats_jetstream_domain: str | None = None,
        http_timeout_sec: float | None = None,
        nats_timeout_sec: float | None = None,
        subject_prefix: str | None = None,
        sender_id: str | None = None,
    ) -> None:
        self.endpoints = endpoints or configured_pipeline_endpoints()
        self.nats_server_url = nats_server_url or configured_nats_server_url()
        # Kept in the signature for callers during migration. Instance routing
        # derives the target Stream from each full Subject.
        del nats_stream, nats_stream_subjects
        self.nats_jetstream_domain = (
            nats_jetstream_domain
            if nats_jetstream_domain is not None
            else configured_jetstream_domain()
        )
        self.http_timeout_sec = float(http_timeout_sec or os.getenv("SIMLINGO_AGENT_HTTP_TIMEOUT_SEC", "180"))
        self.nats_timeout_sec = float(nats_timeout_sec or os.getenv("SIMLINGO_AGENT_NATS_TIMEOUT_SEC", "120"))
        self.subject_prefix = self._compact_subject_prefix(
            subject_prefix or os.getenv("SIMLINGO_MCP_SUBJECT_PREFIX", "workflow.mcp")
        )
        self.subject_session_id = self._compact_subject_token(
            os.getenv("SIMLINGO_MCP_SESSION_ID", f"session-{uuid.uuid4().hex[:8]}")
        )
        self.sender_id = sender_id or os.getenv("SIMLINGO_MCP_SENDER_ID", "Bench2DriveMCP")
        self.agent_instances = configured_agent_instances()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._infer_async(payload))

    async def _infer_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        route_key = str(payload.get("route_key", "route"))
        request_id = self._build_request_id(
            route_key=route_key,
            frame_id=payload.get("frame_id"),
        )
        channel_id = self._build_channel_id(route_key)
        subjects = self._build_subjects(channel_id)
        stage_durations_ms: dict[str, int] = {}
        nats_cls = self._build_nats_comm()
        nats = nats_cls(
            servers=[self.nats_server_url],
            jetstream_domain=self.nats_jetstream_domain,
        )
        started = time.time()
        try:
            payload = self._attach_pipeline_request_id(payload, request_id)
            mark_nats_send(
                payload,
                sender="bench2drive",
                receiver="encoder",
                link="bench2drive_to_encoder",
            )
            await nats.send(subjects["source_input"], payload)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.encoder_url,
                receiver_id="SimLingoEncoderAgent",
                task_id=f"{request_id}-encoder",
                task_type="vision",
                task_description="Encode Bench2Drive observation",
                metadata={
                    "nats_in_subject": subjects["source_input"],
                    "nats_in_durable": subjects["source_durable"],
                    "nats_out_subject": subjects["encoded_output"],
                },
            )
            stage_durations_ms["encoder"] = int((time.time() - stage_started) * 1000)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.scheduler_url,
                receiver_id="SimLingoSchedulerAgent",
                task_id=f"{request_id}-scheduler-budget",
                task_type="budget",
                task_description="Compute evaluation budget",
                metadata={
                    "nats_in_subject": subjects["encoded_output"],
                    "nats_in_durable": subjects["encoded_durable"],
                    "nats_out_subject": subjects["prefix_input"],
                },
            )
            stage_durations_ms["scheduler_budget"] = int((time.time() - stage_started) * 1000)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.llm_url,
                receiver_id="SimLingoLLMAgent",
                task_id=f"{request_id}-llm-prefix",
                task_type="prefix",
                task_description="Run LLM prefix phase",
                metadata={
                    "nats_in_subject": subjects["prefix_input"],
                    "nats_in_durable": subjects["prefix_input_durable"],
                    "nats_out_subject": subjects["prefix_output"],
                },
            )
            stage_durations_ms["llm_prefix"] = int((time.time() - stage_started) * 1000)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.scheduler_url,
                receiver_id="SimLingoSchedulerAgent",
                task_id=f"{request_id}-scheduler-plan",
                task_type="plan",
                task_description="Compute execution plan",
                metadata={
                    "nats_in_subject": subjects["prefix_output"],
                    "nats_in_durable": subjects["prefix_output_durable"],
                    "nats_out_subject": subjects["final_input"],
                },
            )
            stage_durations_ms["scheduler_plan"] = int((time.time() - stage_started) * 1000)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.llm_url,
                receiver_id="SimLingoLLMAgent",
                task_id=f"{request_id}-llm-final",
                task_type="final",
                task_description="Run LLM final phase",
                metadata={
                    "nats_in_subject": subjects["final_input"],
                    "nats_in_durable": subjects["final_input_durable"],
                    "nats_out_subject": subjects["final_output"],
                    "decision_update_out_subject": subjects["decision_update_input"],
                },
            )
            stage_durations_ms["llm_final"] = int((time.time() - stage_started) * 1000)

            stage_started = time.time()
            self._post_execute(
                self.endpoints.scheduler_url,
                receiver_id="SimLingoSchedulerAgent",
                task_id=f"{request_id}-scheduler-decision-update",
                task_type="decision_update",
                task_description="Update scheduler decision-shift history",
                metadata={
                    "nats_in_subject": subjects["decision_update_input"],
                    "nats_in_durable": subjects["decision_update_input_durable"],
                    "nats_out_subject": subjects["decision_update_output"],
                },
            )
            stage_durations_ms["scheduler_decision_update"] = int((time.time() - stage_started) * 1000)

            update_messages = await nats.receive(
                subject=subjects["decision_update_output"],
                durable=subjects["decision_update_output_durable"],
                batch=1,
                timeout_sec=self.nats_timeout_sec,
            )
            if not update_messages:
                raise RemoteInferenceError(
                    "No scheduler decision-update completion received on "
                    f"'{subjects['decision_update_output']}'"
                )
            update_payload = update_messages[0].payload
            record_nats_receive(update_payload, receiver="bench2drive")
            await update_messages[0].ack()

            messages = await nats.receive(
                subject=subjects["final_output"],
                durable=subjects["final_output_durable"],
                batch=1,
                timeout_sec=self.nats_timeout_sec,
            )
            if not messages:
                raise RemoteInferenceError(
                    f"No final output received on subject '{subjects['final_output']}' within {self.nats_timeout_sec}s"
                )
            message = messages[0]
            record_nats_receive(message.payload, receiver="bench2drive")
            await message.ack()
            decoded = message.payload
            self._merge_terminal_traces(decoded, update_payload)
            decoded.setdefault("pipeline_meta", {})
            decoded["pipeline_meta"].update(
                {
                    "request_id": request_id,
                    "subjects": subjects,
                    "endpoints": self.endpoints.__dict__,
                    "nats_server_url": self.nats_server_url,
                    "nats_stream_mode": "instance",
                    "nats_jetstream_domain": self.nats_jetstream_domain,
                    "stage_durations_ms": stage_durations_ms,
                    "total_duration_ms": int((time.time() - started) * 1000),
                }
            )
            self._validate_final_result(decoded)
            return decoded
        finally:
            await nats.close()

    @staticmethod
    def _merge_terminal_traces(final_payload: dict[str, Any], update_payload: dict[str, Any]) -> None:
        def merge_unique(first: list[dict[str, Any]], second: list[dict[str, Any]], fields: tuple[str, ...]):
            merged: list[dict[str, Any]] = []
            seen: set[tuple[Any, ...]] = set()
            for item in first + second:
                key = tuple(item.get(field) for field in fields)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
            return merged

        final_payload["profile_trace"] = merge_unique(
            extract_profile_trace(final_payload),
            extract_profile_trace(update_payload),
            ("pipeline_request_id", "agent", "phase"),
        )
        final_payload["nats_communication_trace"] = merge_unique(
            extract_nats_communication_trace(final_payload),
            extract_nats_communication_trace(update_payload),
            ("pipeline_request_id", "link"),
        )

    def cleanup_route(self, route_key: str) -> dict[str, Any]:
        """Delete the JetStream resources owned by one completed route."""
        return asyncio.run(self._cleanup_route_async(route_key))

    async def _cleanup_route_async(self, route_key: str) -> dict[str, Any]:
        return {
            "route_key": route_key,
            "nats_cleanup": "not-required",
            "reason": "instance WorkQueue messages are removed by ACK",
            "deleted_consumers": {},
            "purged_subjects": {},
        }

    def _post_execute(
        self,
        url: str,
        *,
        receiver_id: str,
        task_id: str,
        task_type: str,
        task_description: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        body = {
            "sender_id": self.sender_id,
            "receiver_id": receiver_id,
            "message_type": "request",
            "payload": {
                "task_id": task_id,
                "task_type": task_type,
                "task_description": task_description,
                "metadata": metadata,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.http_timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RemoteInferenceError(f"HTTP {exc.code} from {url}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RemoteInferenceError(f"Failed to reach {url}: {exc}") from exc

        try:
            payload = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError as exc:
            raise RemoteInferenceError(f"Non-JSON response from {url}: {response_body}") from exc

        response_payload = payload.get("payload", {})
        status = response_payload.get("status")
        if status and status != "success":
            raise RemoteInferenceError(f"{url} returned task status={status}: {response_payload}")
        return payload

    def _validate_final_result(self, result: dict[str, Any]) -> None:
        if result.get("status") not in {None, "success"}:
            raise RemoteInferenceError(f"Final output status is not success: {result.get('status')}")
        if result.get("phase") != "final":
            raise RemoteInferenceError(f"Expected final LLM phase, got: {result.get('phase')}")
        llm_payload = result.get("llm_payload")
        if not isinstance(llm_payload, dict):
            raise RemoteInferenceError("Final output does not contain llm_payload")
        missing = [key for key in ("speed_wps", "route") if llm_payload.get(key) is None]
        if missing:
            raise RemoteInferenceError(f"Final llm_payload is missing required predictions: {missing}")

    def _build_request_id(self, *, route_key: str, frame_id: Any) -> str:
        route_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", route_key).strip("-") or "route"
        frame_str = str(frame_id if frame_id is not None else "frame")
        return f"{route_key}-{frame_str}-{uuid.uuid4().hex[:10]}"

    def _build_channel_id(self, route_key: str) -> str:
        route_token = self._compact_subject_token(route_key)
        return f"{self.subject_session_id}_{route_token}"

    def _compact_subject_prefix(self, subject_prefix: str) -> str:
        parts = [
            re.sub(r"[^a-zA-Z0-9_-]+", "_", part).strip("_")
            for part in str(subject_prefix).split(".")
        ]
        parts = [part for part in parts if part]
        if not parts:
            parts = ["workflow", "mcp"]
        if len(parts) == 1:
            parts.append("mcp")
        if len(parts) > 2:
            parts = [parts[0], "_".join(parts[1:])]
        return ".".join(parts)

    def _compact_subject_token(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "route"

    def _attach_pipeline_request_id(self, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        payload = dict(payload)
        payload["pipeline_request_id"] = request_id
        runtime_context = payload.get("runtime_context")
        if isinstance(runtime_context, dict):
            runtime_context = dict(runtime_context)
            runtime_context["pipeline_request_id"] = request_id
            payload["runtime_context"] = runtime_context
        return payload

    def _build_nats_comm(self):
        try:
            from simlingo_agents.encoder_agent.protocols import NatsComm
        except ModuleNotFoundError as exc:
            raise RemoteInferenceError(
                "NATS dependencies are missing. Install the runtime requirements used by the split agents "
                "before running the remote Bench2Drive agent."
            ) from exc
        return NatsComm

    def _build_subjects(self, request_id: str) -> dict[str, str]:
        del request_id
        cluster = self.agent_instances["encoder"].cluster_id

        def subject(role: str, operation: str) -> str:
            target = self.agent_instances[role]
            scope = "local" if target.cluster_id == cluster else "global"
            return (
                f"workflow.{scope}.{target.cluster_id}.agent.{target.agent_id}."
                f"instance.{target.instance_id}.{operation}"
            )

        def durable(role: str, operation: str) -> str:
            target = self.agent_instances[role]
            return f"{target.agent_id}-{target.instance_id}-{operation}"

        return {
            "source_input": subject("encoder", "encoder_input"),
            "source_durable": durable("encoder", "encoder-input"),
            "encoded_output": subject("scheduler", "scheduler_budget_input"),
            "encoded_durable": durable("scheduler", "budget-input"),
            "prefix_input": subject("llm", "llm_prefix_input"),
            "prefix_input_durable": durable("llm", "prefix-input"),
            "prefix_output": subject("scheduler", "scheduler_plan_input"),
            "prefix_output_durable": durable("scheduler", "plan-input"),
            "final_input": subject("llm", "llm_final_input"),
            "final_input_durable": durable("llm", "final-input"),
            "final_output": subject("llm", "llm_final_output"),
            "final_output_durable": durable("llm", "final-output"),
            "decision_update_input": subject(
                "scheduler", "scheduler_decision_update_input"
            ),
            "decision_update_input_durable": durable(
                "scheduler", "decision-update-input"
            ),
            "decision_update_output": subject(
                "scheduler", "scheduler_decision_update_output"
            ),
            "decision_update_output_durable": durable(
                "scheduler", "decision-update-output"
            ),
        }
