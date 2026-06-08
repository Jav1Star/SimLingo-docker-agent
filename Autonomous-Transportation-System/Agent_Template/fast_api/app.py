"""
FastAPI 应用主体 — Agent Template 的核心服务

架构概览:
  外部调用方 ──POST──> /a2a/execute ──> 解析 A2AMessage/A2ATaskRequest
                                          │
                                          ├─ 从 metadata 或环境变量获取 NATS 主题
                                          │
                                          ├─ agent_function() ── 从 NATS 拉取上游数据
                                          │                      │
                                          │                      ├─ decode_structured_numpy() 还原 numpy 数组
                                          │                      │
                                          │                      ├─ [此处编写你的业务逻辑]
                                          │                      │
                                          │                      ├─ encode_structured_numpy() 编码结果
                                          │                      │
                                          │                      └─ 将结果发布到 NATS 输出主题
                                          │
                                          └─ 构造 A2ATaskResponse / A2AMessage 返回给调用方

NATS 数据流:
  上游 Agent ──publish──> NATS_IN_SUBJECT (workflow.previousagent.result)
                                                   │
                                        本 Agent 通过 pull_subscribe 消费
                                                   │
  下游 Agent <──publish── NATS_OUT_SUBJECT (workflow.agenttemplate.result)
                                本 Agent 处理完毕后 publish 到此主题

环境变量:
  NATS_SERVER_URL  : NATS 服务器地址，默认 nats://nats:4222
  NATS_IN_SUBJECT  : 输入主题（接收上游 Agent 的数据）
  NATS_IN_DURABLE  : 输入持久化消费者名称
  NATS_OUT_SUBJECT : 输出主题（向下游 Agent 发送数据）

开发者指南:
  1. 在 agent_function() 中的 "模拟处理时间" 位置替换为你的业务逻辑
  2. 在 lifespan() 中的 "模型加载" 位置替换为你的模型初始化代码
  3. 通过请求 metadata 覆盖 NATS 主题，或使用环境变量设置默认值
"""
import os
import json
import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse, NatsComm
from utils.logger_utils import get_logger
from utils.numpy_utils import encode_structured_numpy, decode_structured_numpy


logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# NATS 连接配置（可通过环境变量覆盖）
# ──────────────────────────────────────────────────────────────────────

# NATS 服务器地址，容器环境中通常由编排系统注入
NATS_SERVER_URL = os.getenv("NATS_SERVER_URL", "nats://nats:4222")

# 输入主题: 本 Agent 从此主题拉取上游 Agent 的处理结果
NATS_IN_SUBJECT = os.getenv("NATS_IN_SUBJECT", "workflow.previousagent.result")

# 输入持久化消费者名称: JetStream pull_subscribe 的 durable name
# 确保消息不丢失，消费者重启后可以从上次位置继续消费
NATS_IN_DURABLE = os.getenv("NATS_IN_DURABLE", "workflow-previousagent-result")

# 输出主题: 本 Agent 将处理结果发布到此主题，供下游 Agent 消费
NATS_OUT_SUBJECT = os.getenv("NATS_OUT_SUBJECT", "workflow.agenttemplate.result")

logger.info("NATS communication initialized with server: %s", NATS_SERVER_URL)

# 全局 NATS 通信实例，所有 NATS 操作复用此连接
_nats_comm = NatsComm(servers=[NATS_SERVER_URL])


# ──────────────────────────────────────────────────────────────────────
# NATS 数据收发辅助函数
# ──────────────────────────────────────────────────────────────────────

async def _receive_data_from_nats(
    nats_in_subject: str = NATS_IN_SUBJECT,
    nats_in_durable: str = NATS_IN_DURABLE,
) -> dict:
    """
    从 NATS JetStream 拉取消息。

    工作流程:
      1. 通过 pull_subscribe 订阅指定主题和持久化消费者
      2. 批量拉取消息（默认 1 条，超时 5 秒）
      3. 收到消息后确认 (ack) 并返回 payload
      4. 超时时抛出 HTTP 504；其他异常抛出 HTTP 500
      5. finally 块中关闭 NATS 连接

    参数:
        nats_in_subject: 订阅的主题名
        nats_in_durable: 持久化消费者名称

    返回:
        消息的 payload 字典

    异常:
        HTTPException(504): 超时未收到消息
        HTTPException(500): 接收过程中发生其他异常
    """
    try:
        messages = await _nats_comm.receive(
            subject=nats_in_subject,
            durable=nats_in_durable,
            batch=1,
            timeout_sec=5,
        )
        for message in messages:
            logger.info(f"Received message on subject '{nats_in_subject}'")
            await message.ack()  # 确认消息已处理，防止重复投递
            return message.payload
        # 消息列表为空 → 超时
        raise HTTPException(status_code=504, detail=f"No messages received on subject '{nats_in_subject}' within timeout")
    except Exception as exc:
        logger.exception(f"Error receiving message from NATS subject '{nats_in_subject}'")
        raise HTTPException(status_code=500, detail=f"Failed to receive message: {exc}") from exc
    finally:
        # 确保每次接收后关闭连接，避免连接泄漏
        await _nats_comm.close()


async def _send_data_to_nats(data: dict, nats_out_subject: str = NATS_OUT_SUBJECT) -> None:
    """
    将数据发布到 NATS JetStream 输出主题。

    参数:
        data: 要发送的字典（会被 JSON 序列化）
        nats_out_subject: 目标主题名

    发布成功后记录 ack 信息（stream 名称和序列号）
    """
    ack = await _nats_comm.send(
        subject=nats_out_subject,
        payload=data,
    )
    logger.info(f"Data sent to NATS subject '{nats_out_subject}' with ack: {ack}")


# ──────────────────────────────────────────────────────────────────────
# 应用生命周期管理
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    FastAPI 生命周期上下文管理器。

    启动阶段 (yield 前):
      - 加载模型 / 初始化资源
      - 当前为占位代码，替换为你的实际初始化逻辑

    关闭阶段 (finally):
      - 释放 NATS 连接
    """
    try:
        # TODO: 替换为实际的模型加载代码
        # 例如: model = MyModel.load("path/to/model")
        await asyncio.sleep(1)  # 模拟模型加载时间
        logger.info("Model loaded successfully during startup")
    except Exception as exc:
        logger.exception("Failed to load model during startup")
        raise RuntimeError(f"Startup model loading failed: {exc}") from exc

    try:
        yield  # 在此期间处理 HTTP 请求
    finally:
        # 应用关闭时清理 NATS 连接
        await _nats_comm.close()


# ──────────────────────────────────────────────────────────────────────
# FastAPI 应用实例
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Agent Template API", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────
# 核心 Agent 业务逻辑
# ──────────────────────────────────────────────────────────────────────

async def agent_function(
    nats_in_subject: str = NATS_IN_SUBJECT,
    nats_in_durable: str = NATS_IN_DURABLE,
    nats_out_subject: str = NATS_OUT_SUBJECT,
) -> dict:
    """
    Agent 核心业务处理函数。

    数据流:
      1. 从 NATS 输入主题拉取上游 Agent 的数据
      2. 将 numpy 数组从 base64 编码还原为 ndarray
      3. 执行实际业务处理（模型推理、数据变换等）
      4. 将结果中的 numpy 数组编码为 base64 字典
      5. 将编码后的结果发布到 NATS 输出主题

    参数:
        nats_in_subject: 输入主题名
        nats_in_durable: 持久化消费者名称
        nats_out_subject: 输出主题名

    返回:
        {"status": "success"} 或 {"status": "error", ...}

    开发者注意:
      - 在 "模拟处理时间" 位置替换为实际业务逻辑
      - decode_data 是已还原为 numpy 数组的字典，可直接用于模型推理
      - 结果中的 numpy 数组需通过 encode_structured_numpy() 编码后才能通过 NATS 传输
    """
    # 步骤 1: 从 NATS 接收上游数据
    data = await _receive_data_from_nats(nats_in_subject=nats_in_subject, nats_in_durable=nats_in_durable)

    # 步骤 2: 还原 numpy 数组（base64 → ndarray）
    decode_data = decode_structured_numpy(data)

    # TODO: 在此处编写你的业务逻辑
    # ─── 开发者替换区: 在此处添加你的业务逻辑 ───
    # 示例: result_tensor = model.inference(decode_data)
    await asyncio.sleep(1)  # 模拟处理时间

    result = {
        "status": "success",
        "processed_data": encode_structured_numpy(decode_data),  # 编码 numpy 数组为传输友好格式
    }
    # ─── 开发者替换区结束 ───

    # 步骤 3: 将结果发布到 NATS 输出主题
    await _send_data_to_nats(result, nats_out_subject=nats_out_subject)

    return {
        "status": "success",
    }


# ──────────────────────────────────────────────────────────────────────
# HTTP API 端点
# ──────────────────────────────────────────────────────────────────────

@app.post("/a2a/execute")
async def agent_execute(message: dict) -> dict:
    """
    A2A 执行端点 — Agent 间通信的唯一入口。

    请求流程:
      1. 接收外部传入的 JSON，解析为 A2AMessage
      2. 从 A2AMessage.payload 中提取 A2ATaskRequest
      3. 从 task_request.metadata 中读取 NATS 主题配置（如有）
      4. 调用 agent_function() 执行实际业务逻辑
      5. 将结果封装为 A2ATaskResponse + A2AMessage 返回

    请求体示例:
        {
            "sender_id": "L2_Scheduler",
            "receiver_id": "MyAgent",
            "message_type": "request",
            "payload": {
                "task_id": "task-001",
                "task_type": "vision",
                "task_description": "Process the input image",
                "metadata": {
                    "nats_in_subject": "workflow.vision.input",
                    "nats_out_subject": "workflow.vision.output"
                }
            }
        }

    响应体:
        {
            "message_id": "...",
            "sender_id": "AgentTemplate",
            "receiver_id": "L2_Scheduler",
            "message_type": "response",
            "payload": {
                "task_id": "task-001",
                "status": "success",
                "result": "{...}",
                "error_message": null,
                "metadata": {}
            },
            "timestamp": "..."
        }
    """
    logger.info("Received message: %s", message)

    # 解析外部消息
    request_message = A2AMessage(**message)
    task_request = A2ATaskRequest(**request_message.payload)

    # 从 metadata 中获取 NATS 主题配置（优先级高于环境变量默认值）
    metadata = getattr(task_request, 'metadata', {}) or {}
    if 'nats_in_subject' in metadata and metadata.get('nats_in_subject'):
        # metadata 中指定了输入主题 → 使用它
        nats_in_subject = metadata['nats_in_subject']
        # durable 名称: 如果未指定，则将 subject 中的 '.' 替换为 '-'
        nats_in_durable = metadata.get('nats_in_durable') or nats_in_subject.replace('.', '-')
    else:
        # 回退到环境变量配置的默认值
        nats_in_subject = NATS_IN_SUBJECT
        nats_in_durable = NATS_IN_DURABLE

    nats_out_subject = metadata.get('nats_out_subject') or NATS_OUT_SUBJECT

    # 执行核心 Agent 业务逻辑
    result = await agent_function(
        nats_in_subject=nats_in_subject,
        nats_in_durable=nats_in_durable,
        nats_out_subject=nats_out_subject,
    )

    # 构造 A2A 任务响应
    task_response = A2ATaskResponse(
        task_id=task_request.task_id,
        status=result.get("status", "unknown"),
        result=json.dumps(result)
    )

    # 构造 A2A 消息并返回
    response_message = A2AMessage(
        sender_id="AgentTemplate",
        receiver_id=request_message.sender_id,
        message_type="response",
        payload=task_response.dict()
    )

    return response_message.dict()
