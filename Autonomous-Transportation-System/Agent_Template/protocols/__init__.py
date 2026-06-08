"""
Agent Template 协议定义包

本包包含两类核心模块:
  - a2a_protocol : Agent-to-Agent (A2A) 通信协议，定义消息、请求、响应等 Pydantic 模型
  - nats_comm    : 基于 NATS JetStream 的底层数据传输封装，提供发送/接收/请求-响应等操作
"""

from .a2a_protocol import (
    A2AMessage,
    A2ATaskRequest,
    A2ATaskResponse,
    A2ACapabilityDeclaration,
    A2AProgressNotification
)

from .nats_comm import (
    NatsComm, NatsMessage
)

__all__ = [
    "A2AMessage",
    "A2ATaskRequest",
    "A2ATaskResponse",
    "A2ACapabilityDeclaration",
    "A2AProgressNotification",
    "NatsComm",
    "NatsMessage"
]