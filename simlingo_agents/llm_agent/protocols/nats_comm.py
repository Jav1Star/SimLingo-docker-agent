import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import ConsumerConfig, DiscardPolicy, StorageType, StreamConfig
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
        self.stream = stream or os.environ.get("NATS_STREAM", "WORKFLOW")
        self.stream_subjects = stream_subjects or self._stream_subjects_from_env()
        self.jetstream_domain = (
            jetstream_domain
            if jetstream_domain is not None
            else os.environ.get("NATS_JETSTREAM_DOMAIN", "")
        )
        self._nc = NATS()
        self._js = None
        self.stream_storage = os.environ.get("NATS_STREAM_STORAGE", "memory").strip().lower()
        self.stream_max_msgs = self._int_from_env("NATS_STREAM_MAX_MSGS", 128)
        self.stream_max_msgs_per_subject = self._int_from_env("NATS_STREAM_MAX_MSGS_PER_SUBJECT", 1)
        self.stream_max_bytes = self._int_from_env("NATS_STREAM_MAX_BYTES", 512 * 1024 * 1024)
        self.stream_max_age = self._float_from_env("NATS_STREAM_MAX_AGE_SEC", 300.0)
        self.consumer_inactive_threshold = self._float_from_env(
            "NATS_CONSUMER_INACTIVE_THRESHOLD_SEC", 300.0
        )

    @staticmethod
    def _servers_from_env() -> List[str]:
        raw = os.environ.get("NATS_SERVERS", "nats://nats:4222")
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _stream_subjects_from_env() -> List[str]:
        raw = os.environ.get("NATS_STREAM_SUBJECTS", "workflow.>")
        return [item.strip() for item in raw.split(",") if item.strip()]

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
            subjects=self.stream_subjects,
            storage=self._storage_type(),
            discard=DiscardPolicy.OLD,
            max_msgs=self.stream_max_msgs,
            max_msgs_per_subject=self.stream_max_msgs_per_subject,
            max_bytes=self.stream_max_bytes,
            max_age=self.stream_max_age,
        )

    def _jetstream(self):
        if self.jetstream_domain:
            return self._nc.jetstream(domain=self.jetstream_domain)
        return self._nc.jetstream()

    async def connect(self, ensure_stream: bool = True) -> None:
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

    async def close(self) -> None:
        if self._nc.is_connected:
            await self._nc.drain()

    async def _ensure_stream(self) -> None:
        try:
            info = await self._js.stream_info(self.stream)
            config = info.config
            desired = self._stream_config()
            changed = False
            for attr in (
                "subjects",
                "discard",
                "max_msgs",
                "max_msgs_per_subject",
                "max_bytes",
                "max_age",
            ):
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
        await self.connect()
        ack = await self._js.publish(subject, encode_message(payload))
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
        await self.connect()
        consumer_config = ConsumerConfig(inactive_threshold=self.consumer_inactive_threshold)
        sub = await self._js.pull_subscribe(
            subject,
            durable=durable,
            stream=self.stream,
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
        await self.connect()
        results: Dict[str, bool] = {}
        for consumer in dict.fromkeys(consumers):
            try:
                results[consumer] = await self._js.delete_consumer(self.stream, consumer)
            except NotFoundError:
                results[consumer] = False
            except Exception as exc:
                logger.warning("failed to delete stream=%s consumer=%s: %s", self.stream, consumer, exc)
                results[consumer] = False
        return results
    
    async def purge_subjects(self, subjects: List[str]) -> Dict[str, bool]:
        await self.connect()
        results: Dict[str, bool] = {}
        for subject in subjects:
            try:
                results[subject] = await self._js.purge_stream(self.stream, subject=subject)
            except NotFoundError:
                results[subject] = False
            except Exception as exc:
                logger.warning("failed to purge stream=%s subject=%s: %s", self.stream, subject, exc)
                results[subject] = False
        return results

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
