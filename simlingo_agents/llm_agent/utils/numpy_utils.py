from __future__ import annotations

import base64
from typing import Any

import numpy as np


def encode_array_to_dict(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array_from_dict(payload: dict[str, Any]) -> np.ndarray:
    if not {"shape", "dtype", "data"}.issubset(payload):
        raise ValueError("array payload must contain shape, dtype, and data")
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def encode_numpy_payload(payload: Any) -> Any:
    if isinstance(payload, np.ndarray):
        return encode_array_to_dict(payload)
    if isinstance(payload, dict):
        return {key: encode_numpy_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [encode_numpy_payload(item) for item in payload]
    return payload


def decode_numpy_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if {"shape", "dtype", "data"}.issubset(payload):
            return decode_array_from_dict(payload)
        return {key: decode_numpy_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [decode_numpy_payload(item) for item in payload]
    return payload


def encode_structured_numpy(payload: Any) -> Any:
    return encode_numpy_payload(payload)


def decode_structured_numpy(payload: Any) -> Any:
    return decode_numpy_payload(payload)
