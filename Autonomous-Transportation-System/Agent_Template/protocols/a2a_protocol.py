"""
A2A (Agent-to-Agent) 协议定义

标准化 Agent 间通信的消息格式，用于 L2 ↔ L3 和 L3 ↔ L3 通信
"""

from pydantic import BaseModel, Field
from typing import Literal, Any, Optional
from datetime import datetime
import uuid


class A2AMessage(BaseModel):
    """
    A2A 标准消息格式
    
    所有 Agent 间通信都使用此消息包装
    """
    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sender_id: str  # 发送方 Agent ID
    receiver_id: str  # 接收方 Agent ID
    message_type: Literal["request", "response", "notification", "error"]
    payload: dict
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None  # 用于关联请求-响应


class A2ATaskRequest(BaseModel):
    """
    A2A 任务请求 Payload
    
    L2 Scheduler 发送给 L3 Agent 的任务请求格式
    """
    task_id: str
    task_type: str  # "search", "nlp", "compute", "vision", "code", "web"
    task_description: str
    context: dict = Field(default_factory=dict)  # 上下文信息
    timeout: int = 30  # 超时时间（秒）
    priority: Literal["low", "normal", "high"] = "normal"
    require_stream: bool = False  # 是否需要流式返回
    metadata: dict = Field(default_factory=dict)  # 额外元数据


class A2ATaskResponse(BaseModel):
    """
    A2A 任务响应 Payload
    
    L3 Agent 返回给 L2 Scheduler 的任务执行结果
    """
    task_id: str
    status: Literal["success", "error", "timeout", "cancelled"]
    result: Any
    error_message: Optional[str] = None
    metadata: dict = Field(default_factory=dict)  # 执行时间、成本等信息


class A2ACapabilityDeclaration(BaseModel):
    """
    Agent 能力声明
    
    Agent 向注册中心声明自己支持的任务类型和能力
    """
    agent_id: str
    agent_type: str  # search, nlp, compute, vision, code, web
    capabilities: list[dict]  # 支持的任务类型和参数
    status: Literal["online", "busy", "offline"] = "online"
    load_level: float = 0.0  # 当前负载 0.0-1.0
    max_concurrent_tasks: int = 5
    metadata: dict = Field(default_factory=dict)


class A2AProgressNotification(BaseModel):
    """
    任务进度通知
    
    长时间运行的任务可以通过此消息通知进度
    """
    task_id: str
    progress: float  # 0.0-1.0
    current_step: str
    estimated_time_remaining: Optional[int] = None  # 秒
    metadata: dict = Field(default_factory=dict)


# 辅助函数
def create_task_request(
    task_id: str,
    task_type: str,
    task_description: str,
    **kwargs
) -> A2ATaskRequest:
    """快速创建任务请求"""
    return A2ATaskRequest(
        task_id=task_id,
        task_type=task_type,
        task_description=task_description,
        **kwargs
    )


def create_success_response(
    task_id: str,
    result: Any,
    **kwargs
) -> A2ATaskResponse:
    """快速创建成功响应"""
    return A2ATaskResponse(
        task_id=task_id,
        status="success",
        result=result,
        **kwargs
    )


def create_error_response(
    task_id: str,
    error_message: str,
    **kwargs
) -> A2ATaskResponse:
    """快速创建错误响应"""
    return A2ATaskResponse(
        task_id=task_id,
        status="error",
        result=None,
        error_message=error_message,
        **kwargs
    )
