"""Versioned binary codec for SimLingo NATS payloads.

New messages use MessagePack and preserve NumPy arrays as raw bytes.  The
decoder also accepts the previous JSON/Base64 representation so deployments
can be upgraded without invalidating messages already present in JetStream.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import msgpack
import numpy as np

CODEC_ENV_VAR = "SIMLINGO_NATS_CODEC"
MSGPACK_CODEC = "msgpack-v1"
LEGACY_JSON_CODEC = "legacy-json"

_MAGIC = b"SLNG"
_VERSION = 1
_HEADER = _MAGIC + bytes([_VERSION])
_NDARRAY_EXT_CODE = 1


class NatsCodecError(ValueError):
    """Raised when a NATS payload cannot be encoded or decoded."""


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        metadata = msgpack.packb(
            {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data": contiguous.tobytes(),
            },
            use_bin_type=True,
        )
        return msgpack.ExtType(_NDARRAY_EXT_CODE, metadata)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported MessagePack value: {type(value).__name__}")


def _msgpack_ext_hook(code: int, data: bytes) -> Any:
    if code != _NDARRAY_EXT_CODE:
        return msgpack.ExtType(code, data)
    metadata = msgpack.unpackb(data, raw=False, strict_map_key=False)
    try:
        dtype = np.dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])
        raw = metadata["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise NatsCodecError("invalid ndarray extension metadata") from exc
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    except ValueError as exc:
        raise NatsCodecError(
            f"invalid ndarray payload for dtype={dtype} shape={shape}"
        ) from exc


def _decode_legacy_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        if {"shape", "dtype", "data"}.issubset(value):
            try:
                raw = base64.b64decode(value["data"], validate=True)
                return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
            except (TypeError, ValueError) as exc:
                raise NatsCodecError("invalid legacy NumPy payload") from exc
        return {key: _decode_legacy_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_legacy_numpy(item) for item in value]
    return value


def _encode_legacy_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode_legacy_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_legacy_numpy(item) for item in value]
    return value


def configured_codec() -> str:
    codec = os.getenv(CODEC_ENV_VAR, MSGPACK_CODEC).strip().lower()
    aliases = {
        "msgpack": MSGPACK_CODEC,
        "msgpack-v1": MSGPACK_CODEC,
        "json": LEGACY_JSON_CODEC,
        "legacy": LEGACY_JSON_CODEC,
        "legacy-json": LEGACY_JSON_CODEC,
    }
    try:
        return aliases[codec]
    except KeyError as exc:
        raise NatsCodecError(
            f"{CODEC_ENV_VAR} must be one of {sorted(aliases)}, got {codec!r}"
        ) from exc


def encode_message(payload: Any, codec: str | None = None) -> bytes:
    """Serialize one payload, using MessagePack v1 unless configured otherwise."""
    selected = codec or configured_codec()
    if selected == MSGPACK_CODEC:
        packed = msgpack.packb(payload, default=_msgpack_default, use_bin_type=True)
        return _HEADER + packed
    if selected == LEGACY_JSON_CODEC:
        return json.dumps(_encode_legacy_numpy(payload)).encode("utf-8")
    raise NatsCodecError(f"unsupported NATS codec: {selected!r}")


def decode_message(data: bytes | bytearray | memoryview) -> Any:
    """Decode MessagePack v1 or transparently fall back to legacy JSON/Base64."""
    raw = bytes(data)
    if raw.startswith(_MAGIC):
        if len(raw) < len(_HEADER):
            raise NatsCodecError("truncated SimLingo NATS message header")
        version = raw[len(_MAGIC)]
        if version != _VERSION:
            raise NatsCodecError(f"unsupported SimLingo NATS codec version: {version}")
        try:
            return msgpack.unpackb(
                raw[len(_HEADER):],
                raw=False,
                strict_map_key=False,
                ext_hook=_msgpack_ext_hook,
            )
        except (msgpack.MsgpackException, ValueError) as exc:
            raise NatsCodecError("invalid MessagePack NATS payload") from exc

    try:
        legacy = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NatsCodecError("payload is neither msgpack-v1 nor legacy JSON") from exc
    return _decode_legacy_numpy(legacy)
