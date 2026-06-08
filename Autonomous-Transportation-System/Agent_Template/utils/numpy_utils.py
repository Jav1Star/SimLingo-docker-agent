"""
NumPy 数组编解码工具

Agent 间通过 NATS 传输数据时，numpy.ndarray 无法直接 JSON 序列化。
本模块提供递归编码/解码功能，将 numpy 数组转换为传输友好的字典结构:

    编码后格式:
    {
        "shape": (H, W, C),          # 数组形状（tuple → JSON array）
        "dtype": "float32",          # 数据类型字符串
        "data": "base64编码的二进制"  # base64 编码的原始字节
    }

数据流:
    发送方: ndarray → encode_structured_numpy() → dict (JSON 可序列化) → NATS publish
    接收方: NATS receive → dict → decode_structured_numpy() → ndarray (还原)

设计特点:
  - 递归处理: 支持嵌套的 dict / list 结构中的 ndarray
  - 零拷贝还原: np.frombuffer 直接引用解码后的 bytes，无需额外拷贝
  - 类型安全: 保留原始 dtype 信息，还原时精确匹配

使用示例:
    # 编码
    data = {"tensor": np.array([[1, 2], [3, 4]], dtype=np.float32)}
    encoded = encode_structured_numpy(data)
    # → {"tensor": {"shape": [2, 2], "dtype": "float32", "data": "..."}}

    # 解码
    decoded = decode_structured_numpy(encoded)
    # → {"tensor": ndarray([[1, 2], [3, 4]], dtype=float32)}
"""
import base64
import numpy as np

from typing import Any


def encode_structured_numpy(payload: Any) -> Any:
    """
    将 payload 中的 numpy.ndarray 递归编码为传输友好的字典格式。

    编码规则:
      - 遇到 ndarray → 转换为 {"data": base64, "shape": tuple, "dtype": str}
      - 遇到 dict    → 递归处理每个 value
      - 遇到 list    → 递归处理每个 item
      - 其他类型      → 原样返回（int, float, str, bool, None 等）

    参数:
        payload: 任意嵌套结构的数据（dict / list / ndarray / 基本类型）

    返回:
        编码后的结构，其中所有 ndarray 已被替换为字典

    示例:
        >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        >>> encode_structured_numpy({"data": arr})
        {'data': {'data': '...', 'shape': (3,), 'dtype': 'float64'}}
    """
    if isinstance(payload, np.ndarray):
        # 将 ndarray 的原始字节 base64 编码，附带 shape 和 dtype 信息
        return {
            "data": base64.b64encode(payload.tobytes()).decode("utf-8"),
            "shape": payload.shape,
            "dtype": str(payload.dtype),
        }

    if isinstance(payload, dict):
        return {key: encode_structured_numpy(value) for key, value in payload.items()}

    if isinstance(payload, list):
        return [encode_structured_numpy(item) for item in payload]

    # 基本类型直接返回
    return payload


def decode_structured_numpy(payload: Any) -> Any:
    """
    将 encode_structured_numpy() 编码的结构递归还原为 numpy.ndarray。

    解码规则:
      - 遇到包含 shape/dtype/data 三键的 dict → 还原为 ndarray
      - 遇到普通 dict    → 递归处理每个 value
      - 遇到 list        → 递归处理每个 item
      - 其他类型          → 原样返回

    参数:
        payload: 编码后的字典结构

    返回:
        还原后的结构，其中编码字典已被替换回 ndarray

    示例:
        >>> encoded = {"data": {"shape": [3], "dtype": "float64", "data": "..."}}
        >>> decoded = decode_structured_numpy(encoded)
        >>> type(decoded["data"])
        <class 'numpy.ndarray'>
    """
    if isinstance(payload, dict):
        # 检测是否为编码后的数组字典（必须同时包含 shape、dtype、data 三个键）
        if {"shape", "dtype", "data"}.issubset(payload.keys()):
            data = base64.b64decode(payload["data"])
            return np.frombuffer(data, dtype=payload["dtype"]).reshape(payload["shape"])
        # 普通字典，递归处理
        return {k: decode_structured_numpy(v) for k, v in payload.items()}

    if isinstance(payload, list):
        return [decode_structured_numpy(v) for v in payload]

    # 基本类型直接返回
    return payload
