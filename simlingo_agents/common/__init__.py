"""Shared runtime utilities for the split SimLingo agents."""

from .nats_codec import (
    CODEC_ENV_VAR,
    LEGACY_JSON_CODEC,
    MSGPACK_CODEC,
    decode_message,
    encode_message,
)

__all__ = [
    "CODEC_ENV_VAR",
    "LEGACY_JSON_CODEC",
    "MSGPACK_CODEC",
    "decode_message",
    "encode_message",
]
