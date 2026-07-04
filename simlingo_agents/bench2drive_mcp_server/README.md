# Bench2Drive MCP Server

这个目录把 `Bench2Drive` 评测逻辑打包成一个 MCP 服务，并约定：

- `CARLA` 在本机或远端独立运行，不由 MCP server 内部拉起
- `Bench2Drive` 评测仍走完整 leaderboard 闭环
- 单帧推理继续复用三段式 docker agent：
  - `encoder_agent`
  - `scheduler_agent`
  - `llm_agent`

相对于原始版本，这里补了三件事：

- MCP server 改成“本地独立 CARLA 优先”，默认连接 `127.0.0.1:2000`
- 暴露了运行时检查、eval config/route 枚举、session/log 资源
- 修正了三 agent 的 NATS 生命周期，支持 route 内连续多帧推理

## 目录

- `server.py`: MCP server 定义
- `main.py`: 启动入口，风格与 `pointcloud_mcp_demo/main.py` 对齐
- `start.sh`: 简单的本地启动/停止脚本
- `session_manager.py`: 后台评测任务与运行时校验
- `remote_inference.py`: Bench2Drive 单帧到三 agent 的推理桥接

## 安装

```bash
pip install -r simlingo_agents/bench2drive_mcp_server/requirements.txt
```

完整评测仍依赖仓库原有的 Bench2Drive / CARLA / PyTorch 运行环境。

## 启动

```bash
python -m simlingo_agents.bench2drive_mcp_server.main
```

或：

```bash
simlingo_agents/bench2drive_mcp_server/start.sh start
```

支持的 MCP 环境变量：

- `MCP_APP_NAME`，默认 `bench2drive-mcp`
- `MCP_HOST`，默认 `0.0.0.0`
- `MCP_PORT`，默认 `8124`
- `MCP_TRANSPORT`，默认 `streamable-http`

## 运行约定

1. 先单独启动 `CARLA`。
2. 启动 `NATS`。
3. 启动三个 docker/k8s agent。
4. 启动 Bench2Drive MCP server。
5. 先调用 `validate_runtime(...)`。
6. 再调用 `start_evaluation(...)`。

`start_evaluation(...)` 默认按“外部 CARLA”模式运行，默认连接：

- `carla_host=127.0.0.1`
- `carla_port=2000`
- `traffic_manager_port=8000`
- `use_existing_carla=true`

如果确实想退回旧行为，让评测脚本自行拉起 CARLA，可以显式传 `use_existing_carla=false`。

## 对接 `k8sconfig-test`

新版 `k8sconfig-test` 里的三个 agent 通过 Kubernetes Service 暴露：

- `simlingo-encoder-service:9011`
- `simlingo-llm-service:9012`
- `simlingo-scheduler-service:9013`

MCP 侧通过 `SIMLINGO_SPLIT_STACK_PROFILE` 选择默认寻址方式：

- `local`: 默认连接 `127.0.0.1:9011/9012/9013`，适合本地 docker/端口转发。
- `k8s`: 默认连接上述 Service DNS，适合 MCP 也运行在同一个 k8s namespace。
- `nodeport`: 默认连接 `SIMLINGO_K8S_NODE_HOST` 上的 `30111/30112/30113`，适合 MCP 在集群外访问 `k8sconfig-test` 的 NodePort。
- `auto`: 默认值；如果检测到 `KUBERNETES_SERVICE_HOST` 就按 `k8s`，否则按 `local`。

示例：MCP 运行在同一个 k8s 集群内：

```bash
export SIMLINGO_SPLIT_STACK_PROFILE=k8s
export NATS_SERVER_URL=nats://nats:4222
export NATS_JETSTREAM_DOMAIN=hub
```

示例：MCP 运行在集群外，通过 NodePort 访问三个 agent，并通过本地/端口转发访问 NATS：

```bash
export SIMLINGO_SPLIT_STACK_PROFILE=nodeport
export SIMLINGO_K8S_NODE_HOST=<node-ip>
export NATS_SERVER_URL=nats://127.0.0.1:4222
export NATS_JETSTREAM_DOMAIN=hub
```

如果你的网络环境不同，也可以直接覆盖：

```bash
export SIMLINGO_ENCODER_AGENT_URL=http://<host>:<port>/a2a/execute
export SIMLINGO_SCHEDULER_AGENT_URL=http://<host>:<port>/a2a/execute
export SIMLINGO_LLM_AGENT_URL=http://<host>:<port>/a2a/execute
export NATS_SERVER_URL=nats://<host>:4222
```

## MCP Tools

- `describe_runtime`
  - 返回 repo 路径、split-agent URL、默认 CARLA/NATS 配置、已发现的 eval config
- `validate_runtime`
  - 校验关键脚本、checkpoint/route 目录、三 agent 健康状态、NATS 端口、CARLA 端口
- `list_eval_configs`
  - 列出 `configs/` 下可用的 eval yaml
- `list_routes`
  - 根据某个 eval yaml 展开 route 列表
- `start_evaluation`
  - 后台启动一个完整 Bench2Drive session，立即返回 `session_id`
- `get_evaluation_status`
  - 轮询单个 session
- `list_evaluations`
  - 查看所有 session
- `read_evaluation_log`
  - 读取 launcher stdout/stderr 尾部
- `cancel_evaluation`
  - 终止 session

## MCP Resources

- `bench2drive://runtime`
- `bench2drive://sessions`
- `bench2drive://sessions/{session_id}`
- `bench2drive://sessions/{session_id}/logs/{stream}`

## 推理链路

`team_code_adaption/agent_simlingo_remote.py` 会在每一帧：

1. 构造 Bench2Drive 观测
2. 通过 `remote_inference.py` 发到 NATS
3. 顺序触发：
   - encoder
   - scheduler budget
   - llm prefix
   - scheduler plan
   - llm final
4. 取回 `speed_wps` 和 `route`
5. 继续复用原 SimLingo 的控制与评测逻辑

这样做的结果是：

- MCP server 负责“完整评测任务的控制面和封装”
- 三个 docker agent 负责“高频推理数据面”
- CARLA 保持独立运行，不绑死在 MCP 进程里
