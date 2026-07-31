import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import (
    ConsumerConfig,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from nats.js.errors import NotFoundError

try:
    from simlingo_agents.common.nats_codec import decode_message, encode_message
except ModuleNotFoundError:  # Container images expose the shared package at /app/common.
    from common.nats_codec import decode_message, encode_message

logger = logging.getLogger(__name__)


@dataclass
class _SubscriptionLease:
    subscription: Any
    remaining: int
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.remaining -= 1
        if self.remaining > 0:
            return
        self.released = True
        try:
            await self.subscription.unsubscribe()
        except Exception as exc:
            logger.warning("failed to unsubscribe JetStream pull subscription: %s", exc)


@dataclass
class NatsMessage:
    subject: str
    payload: Dict[str, Any]
    stream: Optional[str] = None
    consumer: Optional[str] = None
    stream_seq: Optional[int] = None
    consumer_seq: Optional[int] = None
    _raw: Any = field(default=None, repr=False)
    _subscription_lease: Optional[_SubscriptionLease] = field(default=None, repr=False)

    async def _release_subscription(self) -> None:
        if self._subscription_lease is not None:
            await self._subscription_lease.release()
            self._subscription_lease = None

    async def ack(self) -> None:
        if self._raw is None:
            raise RuntimeError("message does not have a JetStream ack handle")
        try:
            await self._raw.ack()
        finally:
            await self._release_subscription()

    async def nak(self, delay: Optional[float] = None) -> None:
        if self._raw is None:
            raise RuntimeError("message does not have a JetStream ack handle")
        try:
            if delay is None:
                await self._raw.nak()
            else:
                await self._raw.nak(delay=delay)
        finally:
            await self._release_subscription()

    async def in_progress(self) -> None:
        if self._raw is None:
            raise RuntimeError("message does not have a JetStream ack handle")
        await self._raw.in_progress()

    async def term(self) -> None:
        if self._raw is None:
            raise RuntimeError("message does not have a JetStream ack handle")
        try:
            await self._raw.term()
        finally:
            await self._release_subscription()


class NatsComm:
    """
    Runtime communication API for containerized applications.

    Applications use this class inside their own Pod. They do not need to know
    how NATS is deployed; the orchestrator injects NATS_SERVERS as an env var.
    """

    def __init__(
        self,
        servers: Optional[List[str]] = None,
        stream: Optional[str] = None,
        stream_subjects: Optional[List[str]] = None,
        jetstream_domain: Optional[str] = None,
    ):
        self.servers = servers or self._servers_from_env()
        self.cluster_id = os.environ.get("CLUSTER_ID", "").strip()
        self.agent_id = os.environ.get("AGENT_ID", "").strip()
        self.instance_id = os.environ.get("AGENT_INSTANCE_ID", "").strip()
        self.stream_prefix = os.environ.get(
            "NATS_WORKFLOW_STREAM_PREFIX", "WF"
        ).strip() or "WF"
        self.stream = stream or (
            self.workflow_stream_name(self.instance_id)
            if self.instance_id
            else os.environ.get("NATS_STREAM", "")
        )
        self.stream_subjects = stream_subjects or self._stream_subjects_from_env()
        self.jetstream_domain = (
            jetstream_domain
            if jetstream_domain is not None
            else os.environ.get("NATS_JETSTREAM_DOMAIN", "")
        )
        self._nc = NATS()
        self._js = None
        self.stream_storage = os.environ.get("NATS_STREAM_STORAGE", "file").strip().lower()
        self.stream_max_bytes = self._bytes_from_env(
            "NATS_STREAM_MAX_BYTES", 512 * 1024 * 1024
        )
        self.consumer_inactive_threshold = self._float_from_env(
            "NATS_CONSUMER_INACTIVE_THRESHOLD_SEC", 300.0
        )
        self._managed_stream = False

    @staticmethod
    def _servers_from_env() -> List[str]:
        raw = os.environ.get(
            "NATS_SERVERS",
            os.environ.get("NATS_SERVER_URL", "nats://nats:4222"),
        )
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _stream_subjects_from_env() -> List[str]:
        raw = os.environ.get("NATS_STREAM_SUBJECTS", "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _bytes_from_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        units = {
            "kib": 1024,
            "mib": 1024**2,
            "gib": 1024**3,
            "kb": 1000,
            "mb": 1000**2,
            "gb": 1000**3,
        }
        lowered = raw.lower()
        for suffix, multiplier in units.items():
            if lowered.endswith(suffix):
                return int(float(lowered[: -len(suffix)]) * multiplier)
        return int(raw)

    @staticmethod
    def _subject_token(value: str, label: str) -> str:
        token = str(value or "").strip()
        if not token or "." in token or "*" in token or ">" in token:
            raise ValueError(f"invalid NATS {label}: {value!r}")
        return token

    def workflow_stream_name(self, instance_id: str) -> str:
        instance = self._subject_token(instance_id, "instance_id")
        return f"{self.stream_prefix}_{instance}"

    @staticmethod
    def instance_id_from_subject(subject: str) -> str:
        tokens = subject.split(".")
        try:
            marker = tokens.index("instance")
            instance_id = tokens[marker + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"subject is not an instance workflow subject: {subject!r}"
            ) from exc
        return NatsComm._subject_token(instance_id, "instance_id")

    def workflow_stream_subjects(self) -> List[str]:
        cluster = self._subject_token(self.cluster_id, "cluster_id")
        agent = self._subject_token(self.agent_id, "agent_id")
        instance = self._subject_token(self.instance_id, "instance_id")
        return [
            f"workflow.local.{cluster}.agent.{agent}.instance.{instance}.>",
            f"workflow.global.{cluster}.agent.{agent}.instance.{instance}.>",
        ]

    def validate_own_subject(self, subject: str) -> None:
        prefixes = [item[:-1] for item in self.workflow_stream_subjects()]
        if not any(subject.startswith(prefix) for prefix in prefixes):
            raise ValueError(
                f"input subject does not belong to this agent instance: {subject}"
            )

    @staticmethod
    def _int_from_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("invalid integer for %s=%r; using default %s", name, raw, default)
            return default

    @staticmethod
    def _float_from_env(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("invalid float for %s=%r; using default %s", name, raw, default)
            return default

    def _storage_type(self) -> StorageType:
        if self.stream_storage == "file":
            return StorageType.FILE
        return StorageType.MEMORY

    def _stream_config(self) -> StreamConfig:
        return StreamConfig(
            name=self.stream,
            subjects=self.workflow_stream_subjects(),
            storage=self._storage_type(),
            retention=RetentionPolicy.WORK_QUEUE,
            discard=DiscardPolicy.NEW,
            max_bytes=self.stream_max_bytes,
        )

    def _jetstream(self):
        if self.jetstream_domain:
            return self._nc.jetstream(domain=self.jetstream_domain)
        return self._nc.jetstream()

    async def connect(self, ensure_stream: bool = False) -> None:
        if self._nc.is_connected:
            if ensure_stream and self._js is None:
                self._js = self._jetstream()
                await self._ensure_stream()
            return

        await self._nc.connect(
            servers=self.servers,
            connect_timeout=5,
            reconnect_time_wait=2,
            max_reconnect_attempts=10,
        )
        if ensure_stream:
            self._js = self._jetstream()
            await self._ensure_stream()

    async def start(self) -> None:
        if not self.cluster_id or not self.agent_id or not self.instance_id:
            raise RuntimeError(
                "CLUSTER_ID, AGENT_ID and AGENT_INSTANCE_ID are required"
            )
        self.stream = self.workflow_stream_name(self.instance_id)
        await self.connect(ensure_stream=True)
        self._managed_stream = True

    async def close(self) -> None:
        if self._managed_stream and self._js is not None:
            try:
                await self._js.delete_stream(self.stream)
                logger.info("deleted managed workflow stream %s", self.stream)
            except NotFoundError:
                pass
            except Exception:
                logger.exception("failed to delete managed workflow stream %s", self.stream)
            self._managed_stream = False
        if self._nc.is_connected:
            await self._nc.drain()

    async def _ensure_stream(self) -> None:
        try:
            info = await self._js.stream_info(self.stream)
            config = info.config
            desired = self._stream_config()
            changed = False
            for attr in ("subjects", "retention", "discard", "max_bytes", "storage"):
                if getattr(config, attr) != getattr(desired, attr):
                    setattr(config, attr, getattr(desired, attr))
                    changed = True
            if changed:
                await self._js.update_stream(config=config)
                logger.info("updated JetStream stream %s limits", self.stream)
        except NotFoundError:
            await self._js.add_stream(config=self._stream_config())
            logger.info("created JetStream stream %s with subjects=%s", self.stream, self.stream_subjects)
        except Exception as exc:
            logger.warning("failed to ensure JetStream stream %s exists: %s", self.stream, exc)
            raise

    async def send(self, subject: str, payload: Any) -> Dict[str, Any]:
        await self.connect(ensure_stream=False)
        js = self._jetstream()
        ack = await js.publish(subject, encode_message(payload))
        return {
            "subject": subject,
            "stream": ack.stream,
            "seq": ack.seq,
        }

    async def receive(
        self,
        subject: str,
        durable: Optional[str],
        batch: int = 1,
        timeout_sec: float = 5.0,
        ack: bool = False,
    ) -> List[NatsMessage]:
        await self.connect(ensure_stream=False)
        js = self._jetstream()
        stream = self.workflow_stream_name(self.instance_id_from_subject(subject))
        consumer_config = ConsumerConfig(inactive_threshold=self.consumer_inactive_threshold)
        sub = await js.pull_subscribe(
            subject,
            durable=durable,
            stream=stream,
            config=consumer_config,
        )

        try:
            raw_messages = await sub.fetch(batch, timeout=timeout_sec)
        except NatsTimeoutError:
            await sub.unsubscribe()
            return []
        except Exception:
            await sub.unsubscribe()
            raise

        messages: List[NatsMessage] = []
        lease = _SubscriptionLease(subscription=sub, remaining=len(raw_messages))
        try:
            for raw in raw_messages:
                data = decode_message(raw.data)
                metadata = raw.metadata
                messages.append(
                    NatsMessage(
                        subject=raw.subject,
                        payload=data,
                        stream=metadata.stream,
                        consumer=metadata.consumer,
                        stream_seq=metadata.sequence.stream,
                        consumer_seq=metadata.sequence.consumer,
                        _raw=raw,
                        _subscription_lease=None if ack else lease,
                    )
                )
                if ack:
                    await raw.ack()
        except Exception:
            await sub.unsubscribe()
            raise

        if ack or not raw_messages:
            await sub.unsubscribe()

        return messages

    async def delete_consumers(self, consumers: List[str]) -> Dict[str, bool]:
        await self.connect(ensure_stream=False)
        results: Dict[str, bool] = {}
        for consumer in dict.fromkeys(consumers):
            results[consumer] = False
        return results
    
    async def purge_subjects(self, subjects: List[str]) -> Dict[str, bool]:
        return {subject: False for subject in subjects}

    async def serve(
        self,
        subject: str,
        durable: str,
        handler: Callable[[Dict[str, Any]], Any],
        poll_timeout_sec: float = 5.0,
    ) -> None:
        while True:
            messages = await self.receive(
                subject=subject,
                durable=durable,
                batch=1,
                timeout_sec=poll_timeout_sec,
                ack=False,
            )
            for message in messages:
                try:
                    result = handler(message.payload)
                    if asyncio.iscoroutine(result):
                        await result
                    await message.ack()
                except Exception as exc:
                    logger.exception("handler failed for subject=%s payload=%s", subject, message.payload)
                    try:
                        await message.nak()
                    except Exception as nak_exc:
                        logger.warning("failed to nak message after handler error: %s", nak_exc)
                    raise

    async def request(
        self,
        subject: str,
        payload: Dict[str, Any],
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        await self.connect(ensure_stream=False)

        try:
            msg = await self._nc.request(
                subject,
                encode_message(payload),
                timeout=timeout_sec,
            )
        except NatsTimeoutError as exc:
            raise TimeoutError(f"timeout waiting for reply on subject={subject}") from exc

        return decode_message(msg.data)

    async def respond(
        self,
        subject: str,
        handler: Callable[[Dict[str, Any]], Any],
        queue: Optional[str] = None,
    ) -> None:
        await self.connect(ensure_stream=False)

        async def _callback(msg):
            try:
                payload = decode_message(msg.data)
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    result = await result
                await msg.respond(encode_message(result or {}))
            except Exception as exc:
                logger.exception("request handler failed for subject=%s", subject)
                await msg.respond(encode_message({"error": str(exc)}))

        await self._nc.subscribe(subject, queue=queue, cb=_callback)
        while True:
            await asyncio.sleep(3600)
