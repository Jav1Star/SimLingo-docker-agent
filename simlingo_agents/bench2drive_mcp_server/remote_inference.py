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

from simlingo_agents.encoder_agent.utils.numpy_utils import (
    decode_structured_numpy,
    encode_structured_numpy,
)


class RemoteInferenceError(RuntimeError):
    """Raised when the split-agent pipeline fails."""


@dataclass(frozen=True)
class PipelineEndpoints:
    encoder_url: str
    scheduler_url: str
    llm_url: str


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
    if split_stack_profile() in {"k8s", "nodeport"}:
        return "hub"
    return ""


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
        self.nats_stream = nats_stream or os.getenv("NATS_STREAM", "WORKFLOW")
        raw_subjects = os.getenv("NATS_STREAM_SUBJECTS", "workflow.>")
        self.nats_stream_subjects = nats_stream_subjects or [item.strip() for item in raw_subjects.split(",") if item.strip()]
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
            stream=self.nats_stream,
            stream_subjects=self.nats_stream_subjects,
            jetstream_domain=self.nats_jetstream_domain,
        )
        started = time.time()
        try:
            payload = self._attach_pipeline_request_id(payload, request_id)
            await self._cleanup_request_subjects(nats, subjects)
            await nats.send(subjects["source_input"], encode_structured_numpy(payload))

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
                },
            )
            stage_durations_ms["llm_final"] = int((time.time() - stage_started) * 1000)

            messages = await nats.receive(
                subject=subjects["final_output"],
                durable=None,
                batch=1,
                timeout_sec=self.nats_timeout_sec,
            )
            if not messages:
                raise RemoteInferenceError(
                    f"No final output received on subject '{subjects['final_output']}' within {self.nats_timeout_sec}s"
                )
            message = messages[0]
            await message.ack()
            decoded = decode_structured_numpy(message.payload)
            await self._cleanup_request_subjects(nats, subjects)
            decoded.setdefault("pipeline_meta", {})
            decoded["pipeline_meta"].update(
                {
                    "request_id": request_id,
                    "subjects": subjects,
                    "endpoints": self.endpoints.__dict__,
                    "nats_server_url": self.nats_server_url,
                    "nats_stream": self.nats_stream,
                    "nats_jetstream_domain": self.nats_jetstream_domain,
                    "stage_durations_ms": stage_durations_ms,
                    "total_duration_ms": int((time.time() - started) * 1000),
                }
            )
            self._validate_final_result(decoded)
            return decoded
        finally:
            await self._cleanup_request_subjects(nats, subjects)
            await nats.close()

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

    async def _cleanup_request_subjects(self, nats: Any, subjects: dict[str, str]) -> None:
        data_subjects = [
            subject
            for key, subject in subjects.items()
            if not key.endswith("_durable") and "durable" not in key
        ]
        try:
            await nats.purge_subjects(data_subjects)
        except AttributeError:
            return
        except Exception:
            return

    def _build_subjects(self, request_id: str) -> dict[str, str]:
        base = f"{self.subject_prefix}.{request_id}"
        return {
            "source_input": f"{base}.source_input",
            "source_durable": f"{base.replace('.', '-')}-source-input",
            "encoded_output": f"{base}.scheduler_budget_input",
            "encoded_durable": f"{base.replace('.', '-')}-scheduler-budget-input",
            "prefix_input": f"{base}.llm_prefix_input",
            "prefix_input_durable": f"{base.replace('.', '-')}-scheduler-budget-output-llm-prefix-input",
            "prefix_output": f"{base}.llm_prefix_output",
            "prefix_output_durable": f"{base.replace('.', '-')}-llm-prefix-output-scheduler-plan-input",
            "final_input": f"{base}.llm_final_input",
            "final_input_durable": f"{base.replace('.', '-')}-scheduler-plan-output-llm-final-input",
            "final_output": f"{base}.llm_final_output",
            "final_output_durable": f"{base.replace('.', '-')}-llm-final-output",
        }
