## 写在最前面

需要更改的地方应该只有app.py中的lifespan()和agent_function()函数。lifespan()函数中是模型加载的地方，agent_function()函数中是核心业务逻辑的地方。其他部分都是协议层和通信层的封装，应该不需要改。

agent_function()函数名可以修改成更符合你业务语义的名字，但函数签名（参数和返回值）必须保持不变，以确保与协议层的兼容。

agent的调用和数据流是分离的，调用是通过HTTP POST请求到/a2a/execute端点，数据流是通过NATS消息队列进行的。这是因为数据的传输和处理可能涉及到较大的数据量（如图像、特征张量等），直接通过HTTP传输可能不太合适，而NATS提供了更高效的异步消息传递机制。

NATS的IN或OUT的SUBJECT应随项目需要而动态删减。比如如果需要从上一个智能体处通过NATS获取数据，则保留IN_SUBJECT；如果需要向下一个智能体通过NATS发送数据，则保留OUT_SUBJECT。否则，可以将相关的代码注释掉或删除。SUBJECT的命名不需要更改，调度层调用时会更改成对应的主题名称。

将项目打包成docker镜像就行，k8s文件夹可以不用管，那个文件是给后续部署到k8s集群用的，可以暂时不管那个文件。

# Agent Template

一个基于 **FastAPI + NATS** 的智能体（Agent）模板项目，提供标准化的 Agent-to-Agent (A2A) 通信接口，帮助开发者快速构建可接入多智能体工作流的独立 Agent 服务。

---

## 项目架构

```
外部调用方 ──POST──> /a2a/execute ──> 解析 A2A 消息
                                          │
                                          ├─ 从 NATS 拉取上游数据
                                          │   (workflow.previousagent.result)
                                          │
                                          ├─ 处理数据（模型推理 / 业务逻辑）
                                          │
                                          ├─ 将结果发布到 NATS
                                          │   (workflow.agenttemplate.result)
                                          │
                                          └─ 返回 A2A 响应给调用方
```

### 技术栈

| 组件        | 技术              | 说明                           |
| ----------- | ----------------- | ------------------------------ |
| Web 框架    | FastAPI + Uvicorn | 提供 HTTP API 服务             |
| 消息中间件  | NATS JetStream    | Agent 间异步数据传输           |
| 数据序列化  | JSON + Base64     | 原生类型 JSON 编码，numpy 数组 base64 编码 |
| 数据模型    | Pydantic          | A2A 协议消息校验               |
| 日志        | Python logging    | 统一命名空间的日志系统         |

---

## 项目结构

```
Agent_Template/
├── main.py                  # 项目入口，启动 uvicorn 服务器
├── fast_api/
│   └── app.py               # FastAPI 应用主体（核心业务逻辑在此）
├── protocols/               # 通信协议层（黑盒使用，无需修改）
│   ├── __init__.py          # 包导出
│   ├── a2a_protocol.py      # A2A 消息协议定义
│   └── nats_comm.py         # NATS 通信封装
└── utils/
    ├── __init__.py          # （空）
    ├── logger_utils.py      # 日志工具
    └── numpy_utils.py       # NumPy 数组编解码工具
├── Dockerfile               # Docker 镜像构建文件
└── k8s/
    └── agent-template.yaml   # K8s Deployment + Service 配置
```

---

## API 端点

### POST `/a2a/execute`

Agent 间通信的唯一入口。接收 A2A 格式消息，执行任务并返回结果。

**请求体**（JSON）:

```json
{
    "sender_id": "L2_Scheduler",
    "receiver_id": "MyAgent",
    "message_type": "request",
    "payload": {
        "task_id": "task-001",
        "task_type": "vision",
        "task_description": "Process the input data",
        "context": {},
        "metadata": {
            "nats_in_subject": "workflow.vision.input",
            "nats_out_subject": "workflow.vision.output"
        }
    }
}

{"sender_id": "L2_Scheduler","receiver_id": "MyAgent","message_type": "request","payload": {"task_id": "task-001","task_type": "vision","task_description": "Process the input data","context": {},"metadata": {} } }
```


**响应体**（JSON）:

```json
{
    "message_id": "...",
    "sender_id": "AgentTemplate",
    "receiver_id": "L2_Scheduler",
    "message_type": "response",
    "payload": {
        "task_id": "task-001",
        "status": "success",
        "result": "{\"status\": \"success\", \"processed_data\": {...}}",
        "error_message": null,
        "metadata": {}
    },
    "timestamp": "2025-01-01T00:00:00"
}
```

### 环境变量

| 变量名              | 说明                           | 默认值                           |
| ------------------- | ------------------------------ | -------------------------------- |
| `NATS_SERVER_URL`   | NATS 服务器地址                | `nats://nats:4222`               |
| `NATS_IN_SUBJECT`   | 输入主题（接收上游数据）       | `workflow.previousagent.result`  |
| `NATS_IN_DURABLE`   | 输入持久化消费者名称           | `workflow-previousagent-result`  |
| `NATS_OUT_SUBJECT`  | 输出主题（发送结果到下游）     | `workflow.agenttemplate.result`  |

---

## 如何定制你的 Agent

### 步骤 1: 加载模型

在 [app.py](fast_api/app.py) 的 `lifespan()` 函数中替换模型加载逻辑：

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 加载你的模型
    model = MyModel.load("path/to/model")
    logger.info("Model loaded successfully")

    try:
        yield
    finally:
        await _nats_comm.close()
```

### 步骤 2: 编写业务逻辑

在 `agent_function()` 中实现你的 Agent 核心逻辑：

```python
async def agent_function(...):
    # 1. 从 NATS 接收上游数据（已自动还原 numpy 数组）
    data = await _receive_data_from_nats(...)
    decode_data = decode_structured_numpy(data)

    # 2. ─── 在这里添加你的业务逻辑 ───
    # 例如: result_tensor = model.inference(decode_data)
    result_tensor = ...

    # 3. 编码结果并发布到 NATS
    result = {
        "status": "success",
        "processed_data": encode_structured_numpy(result_tensor),
    }
    await _send_data_to_nats(result, ...)

    return {"status": "success"}
```

---

## 数据流详解

### 完整的 Agent 间通信流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        L2 Scheduler (调度层)                       │
│                                                                  │
│  1. 构造 A2AMessage(request)                                     │
│  2. POST /a2a/execute → L3 Agent                                 │
│  3. 等待响应                                                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     L3 Agent (本模板)                              │
│                                                                  │
│  1. 解析 A2AMessage → A2ATaskRequest                             │
│  2. 从 NATS_IN_SUBJECT 拉取上游数据 (JetStream pull_subscribe)   │
│  3. decode_structured_numpy() 还原 numpy 数组                    │
│  4. 执行模型推理 / 数据处理                                       │
│  5. encode_structured_numpy() 编码结果                            │
│  6. 发布到 NATS_OUT_SUBJECT (JetStream publish)                   │
│  7. 构造 A2AMessage(response) 返回                                │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     下游 Agent / L2 Scheduler                      │
│                                                                  │
│  1. 从 NATS_OUT_SUBJECT 消费数据                                  │
│  2. 处理或使用结果                                                │
└──────────────────────────────────────────────────────────────────┘
```

### NumPy 数组传输

Agent 间传输 numpy 数组时，需要通过编码转换：

```
ndarray → encode_structured_numpy() → {"shape": ..., "dtype": ..., "data": "base64..."}
                                      ↓ JSON 序列化 → NATS 传输 → JSON 反序列化
dict   → decode_structured_numpy() → ndarray (还原)
```

---

## 协议说明（黑盒）

`protocols/` 目录包含两个模块，开发者无需了解其内部实现，只需知道如何使用：

### `A2AMessage` — 消息信封

```python
from protocols import A2AMessage, A2ATaskRequest, A2ATaskResponse

# 构造请求消息
msg = A2AMessage(
    sender_id="me",
    receiver_id="agent",
    message_type="request",
    payload={
        "task_id": "task-1",
        "task_type": "vision",
        "task_description": "...",
    },
)
```

### `NatsComm` — NATS 通信（内部使用）

应用内部已封装了 `_receive_data_from_nats()` 和 `_send_data_to_nats()` 函数，
开发者只需在 `agent_function()` 中调用业务逻辑即可，无需直接使用 `NatsComm`。

---

## 日志

项目使用统一的日志命名空间 `AgentTemplate.<module>`，所有日志通过 `get_logger(__name__)` 获取：

```python
from utils.logger_utils import get_logger

logger = get_logger(__name__)  # → AgentTemplate.fast_api.app
logger.info("Processing request...")
# 输出: 2025-01-01 00:00:00,000 - AgentTemplate.fast_api.app - INFO - Processing request...
```

---

## 容器化部署

### Docker

#### 构建镜像

```bash
# 0.1.0 是版本号，可以根据需要修改
docker build -t agent-template:0.1.0 .
```

#### 运行容器

```bash
docker run agent-template:0.1.0
```

### Kubernetes (Minikube)

#### 部署到集群

```bash
# 将本地镜像加载到 Minikube
minikube image load agent-template:0.1.0

# 应用 K8s 资源（Deployment + Service）
kubectl apply -f k8s/agent-template.yaml
```

#### 访问服务

Service 类型为 `NodePort`，可通过以下地址访问：

```
http://localhost:30091/a2a/execute
```

#### 查看状态

```bash
# 查看 Pod 状态
kubectl get pods -l app=agent-template

# 查看日志
kubectl logs -l app=agent-template
```

#### 移除资源

```bash
kubectl delete -f k8s/agent-template.yaml
```
