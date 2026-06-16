# Bench2Drive MCP Server

这个目录把 Bench2Drive 评测流程包装成一个异步 MCP server，同时新增了一个
`team_code_adaption/agent_simlingo_remote.py` 用于把单体 SimLingo agent 的本地前向，
替换成 `encoder_agent -> scheduler_agent -> llm_agent` 的远程三段式推理链路。

## 设计目标

- `CARLA` 可以单独运行在服务器上，评测器通过 `--remote-carla-host/port` 连接。
- `Bench2Drive` 的完整 route 评测可能持续数分钟，因此不能把“跑完整个 route”做成单个阻塞 MCP tool。
- MCP 只承担“控制面”：
  - `start_evaluation`
  - `get_evaluation_status`
  - `read_evaluation_log`
  - `cancel_evaluation`
- 每一帧的高频推理数据面仍复用现有三 agent 链路：
  - 观测数据编码后发到 NATS
  - 依次触发 encoder / scheduler budget / llm prefix / scheduler plan / llm final
  - 从最终 NATS subject 取回 `speed_wps` 和 `route`

这样可以同时满足：

- 模型推理逻辑不重写
- MCP 调用秒级返回
- Bench2Drive 仍能跑完整闭环评测

## 目录

- `server.py`: MCP server 入口
- `session_manager.py`: 后台评测会话管理
- `remote_inference.py`: 单帧远程三 agent 推理 client
- `requirements.txt`: MCP server 额外依赖

## 新增评测 Agent

`team_code_adaption/agent_simlingo_remote.py`

这个 agent 复用了原 `agent_simlingo.py` 的：

- 传感器定义
- 观测预处理
- prompt 构造
- waypoint / speed 后处理
- PID 控制

只把本地 `self.model(...)` 改成了对 split agents 的远程调用。

## 环境变量

远程推理 agent 默认读取这些环境变量：

- `SIMLINGO_ENCODER_AGENT_URL`
  - 默认 `http://127.0.0.1:9011/a2a/execute`
- `SIMLINGO_SCHEDULER_AGENT_URL`
  - 默认 `http://127.0.0.1:9013/a2a/execute`
- `SIMLINGO_LLM_AGENT_URL`
  - 默认 `http://127.0.0.1:9012/a2a/execute`
- `NATS_SERVER_URL`
  - 默认 `nats://127.0.0.1:4222`
- `NATS_STREAM`
  - 默认 `WORKFLOW`
- `NATS_STREAM_SUBJECTS`
  - 默认 `workflow.>`
- `NATS_JETSTREAM_DOMAIN`
  - 默认 `hub`
- `SIMLINGO_AGENT_HTTP_TIMEOUT_SEC`
  - 默认 `60`
- `SIMLINGO_AGENT_NATS_TIMEOUT_SEC`
  - 默认 `120`

## 安装

```bash
pip install -r simlingo_agents/bench2drive_mcp_server/requirements.txt
```

## 启动 MCP Server

```bash
python -m simlingo_agents.bench2drive_mcp_server.server
```

## MCP Tools

### `start_evaluation`

启动后台评测任务，立即返回 `session_id`。

关键参数：

- `eval_config_path`
- `route_ids`
- `seed`
- `remote_carla_host`
- `remote_carla_port`
- `remote_tm_port`
- `gpu_ids`

说明：

- server 会复制一份 eval yaml 到 runtime 目录
- 自动把 `agent_file` 改成 `team_code_adaption/agent_simlingo_remote.py`
- 自动把 `out_root` 改成独立的 session 输出目录

### `get_evaluation_status`

轮询 session 状态，不会阻塞到 route 结束。

返回内容包含：

- `status`
- `result_summary`
- `stdout_log`
- `stderr_log`
- `expected_routes`

### `read_evaluation_log`

读取 launcher 的 stdout/stderr 尾部内容。

### `cancel_evaluation`

终止一个后台评测 session。

## 远端 CARLA

`start_eval_simlingo_adaption.py` 已补充：

- `--remote-carla-host`
- `--remote-carla-port`
- `--remote-tm-port`

并会把 host 透传给 `leaderboard_evaluator.py --host=...`。

## 推荐运行方式

1. 先在服务器上启动 CARLA。
2. 启动 NATS。
3. 启动三个 docker agent。
4. 启动这个 MCP server。
5. 调用 `start_evaluation(...)`。
6. 轮询 `get_evaluation_status(session_id)`。

## 备注

- 当前 MCP server 负责“完整评测任务的异步管理”，而不是把每一帧观测暴露成独立 MCP tool。
- 如果后面你想做“LLM 直接逐帧调用 MCP tool 决策”，可以在现有 `session_manager` 基础上继续扩成 step-wise tool API。
