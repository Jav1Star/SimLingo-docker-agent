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
        self.endpoints = endpoints or PipelineEndpoints(
            encoder_url=os.getenv("SIMLINGO_ENCODER_AGENT_URL", "http://127.0.0.1:9011/a2a/execute"),
            scheduler_url=os.getenv("SIMLINGO_SCHEDULER_AGENT_URL", "http://127.0.0.1:9013/a2a/execute"),
            llm_url=os.getenv("SIMLINGO_LLM_AGENT_URL", "http://127.0.0.1:9012/a2a/execute"),
        )
        self.nats_server_url = nats_server_url or os.getenv("NATS_SERVER_URL", "nats://127.0.0.1:4222")
        self.nats_stream = nats_stream or os.getenv("NATS_STREAM", "WORKFLOW")
        raw_subjects = os.getenv("NATS_STREAM_SUBJECTS", "workflow.>")
        self.nats_stream_subjects = nats_stream_subjects or [item.strip() for item in raw_subjects.split(",") if item.strip()]
        self.nats_jetstream_domain = (
            nats_jetstream_domain
            if nats_jetstream_domain is not None
            else os.getenv("NATS_JETSTREAM_DOMAIN", "")
        )
        self.http_timeout_sec = float(http_timeout_sec or os.getenv("SIMLINGO_AGENT_HTTP_TIMEOUT_SEC", "60"))
        self.nats_timeout_sec = float(nats_timeout_sec or os.getenv("SIMLINGO_AGENT_NATS_TIMEOUT_SEC", "120"))
        self.subject_prefix = (subject_prefix or os.getenv("SIMLINGO_MCP_SUBJECT_PREFIX", "workflow.mcp")).rstrip(".")
        self.sender_id = sender_id or os.getenv("SIMLINGO_MCP_SENDER_ID", "Bench2DriveMCP")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._infer_async(payload))

    async def _infer_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = self._build_request_id(
            route_key=str(payload.get("route_key", "route")),
            frame_id=payload.get("frame_id"),
        )
        subjects = self._build_subjects(request_id)
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
                durable=subjects["final_output_durable"],
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
            decoded.setdefault("pipeline_meta", {})
            decoded["pipeline_meta"].update(
                {
                    "request_id": request_id,
                    "subjects": subjects,
                    "stage_durations_ms": stage_durations_ms,
                    "total_duration_ms": int((time.time() - started) * 1000),
                }
            )
            return decoded
        finally:
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

    def _build_request_id(self, *, route_key: str, frame_id: Any) -> str:
        route_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", route_key).strip("-") or "route"
        frame_str = str(frame_id if frame_id is not None else "frame")
        return f"{route_key}-{frame_str}-{uuid.uuid4().hex[:10]}"

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
        base = f"{self.subject_prefix}.{request_id}"
        return {
            "source_input": f"{base}.source_input",
            "source_durable": f"{base.replace('.', '-')}-source-input",
            "encoded_output": f"{base}.scheduler_budget_input",
            "encoded_durable": f"{base.replace('.', '-')}-scheduler-budget-input",
            "prefix_input": f"{base}.scheduler_budget_output.llm_prefix_input",
            "prefix_input_durable": f"{base.replace('.', '-')}-scheduler-budget-output-llm-prefix-input",
            "prefix_output": f"{base}.llm_prefix_output.scheduler_plan_input",
            "prefix_output_durable": f"{base.replace('.', '-')}-llm-prefix-output-scheduler-plan-input",
            "final_input": f"{base}.scheduler_plan_output.llm_final_input",
            "final_input_durable": f"{base.replace('.', '-')}-scheduler-plan-output-llm-final-input",
            "final_output": f"{base}.llm_final_output",
            "final_output_durable": f"{base.replace('.', '-')}-llm-final-output",
        }
