# Perception2IntermediateFeature Agent Demo Docker 化指南

## 1. 前置条件

- 安装 Docker 24+
- 本地存在 conda 环境 `langmanus`（用于导出环境文件）

## 2. 导出 conda 环境文件

在项目根目录执行：

```bash
conda env export -n langmanus --no-builds | sed '/^prefix:/d' > environment.langmanus.yml
```

说明：镜像构建时会根据 `environment.langmanus.yml` 创建同名 conda 环境（`langmanus`）。

## 3. 准备模型目录

项目默认通过环境变量 `MODEL_CHECKPOINT_PATH` 读取模型目录。

建议在项目根目录放置：

- `models/point_pillar_where2comm/`

并确保该目录下至少包含：

- `config.yaml`
- `net_epoch*.pth` 或 `latest.pth`

## 4. 构建镜像

```bash
docker build -t perception2intermediatefeature-agent:latest .
```

## 5. 运行容器（单命令）

```bash
docker run --rm \
  --gpus all \
  -p 9001:9001 \
  --add-host host.docker.internal:host-gateway \
  -e MCP_SERVER_HOST=host.docker.internal \
  -e MCP_SERVER_PORT=8123 \
  -e MODEL_CHECKPOINT_PATH=/app/checkpoints/point_pillar_where2comm/ \
  -e FRONTEND_CALLBACK_URL=http://host.docker.internal:9002/temp/post_data \
  -v $(pwd)/models/point_pillar_where2comm:/app/checkpoints/point_pillar_where2comm:ro \
  --name perception2intermediatefeature-agent \
  perception2intermediatefeature-agent:latest
```

## 6. 使用 Compose 运行

```bash
docker compose up --build
```

## 7. 健康检查与调用

服务默认监听 `9001`：

- 推理接口：`GET /model/forward`
- A2A 接口：`POST /a2a/execute`

示例：

```bash
curl http://127.0.0.1:9001/model/forward
```

## 8. 常见问题

- 启动时报 `Model config not found`：
  - 检查 `MODEL_CHECKPOINT_PATH` 是否指向容器内真实目录（末尾带 `/` 更稳妥）
- `host.docker.internal` 无法访问：
  - Linux 下需要 `--add-host host.docker.internal:host-gateway`
- CUDA 扩展编译失败：
  - 确认主机 GPU 驱动与 CUDA 11.8 兼容
  - 确认构建时网络可访问 PyTorch/cu118 与 pip 源
