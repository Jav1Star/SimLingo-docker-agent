from datetime import datetime
from typing import Any, Literal, Optional
import uuid

from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sender_id: str
    receiver_id: str
    message_type: Literal["request", "response", "notification", "error"]
    payload: dict
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None


class A2ATaskRequest(BaseModel):
    task_id: str
    task_type: str
    task_description: str
    context: dict = Field(default_factory=dict)
    timeout: int = 30
    priority: Literal["low", "normal", "high"] = "normal"
    require_stream: bool = False
    metadata: dict = Field(default_factory=dict)


class A2ATaskResponse(BaseModel):
    task_id: str
    status: Literal["success", "error", "timeout", "cancelled"]
    result: Any
    error_message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
