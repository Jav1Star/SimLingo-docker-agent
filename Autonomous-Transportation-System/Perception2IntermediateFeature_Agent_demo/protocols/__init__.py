"""
LangManus 协议定义包
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