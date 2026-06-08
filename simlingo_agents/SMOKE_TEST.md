# SimLingo Agents Smoke Test

下面是一套本机“两阶段 LLM + scheduler”链路最小 smoke test。

## 1. 启动 NATS

```bash
sudo docker rm -f simlingo-nats 2>/dev/null || true
sudo docker run -d --name simlingo-nats -p 4222:4222 nats:latest -js
```

## 2. 启动三个 agent

```bash
sudo docker compose -f simlingo_agents/docker-compose.yml up -d simlingo-encoder-agent simlingo-scheduler-agent simlingo-llm-agent
```

查看健康状态：

```bash
curl http://127.0.0.1:9011/health
curl http://127.0.0.1:9013/health
curl http://127.0.0.1:9012/health
```

## 3. 发布最小输入到 encoder 的上游 subject

使用 `encoder_agent` 容器内自带工具发布一条最小消息：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py publish-minimal-input
```

这条消息会发到默认 subject：

```text
workflow.previousagent.result
```

## 4. 依次触发五个阶段

触发 encoder：

```bash
curl -X POST http://127.0.0.1:9011/a2a/execute \
  -H 'Content-Type: application/json' \
  --data @simlingo_agents/smoke_test/encoder_execute.json
```

触发 scheduler 的 `budget` 阶段。
这一步只消费 `encoder_agent` 输出，并生成 `budget_value`：

```bash
curl -X POST http://127.0.0.1:9013/a2a/execute \
  -H 'Content-Type: application/json' \
  --data @simlingo_agents/smoke_test/scheduler_budget_execute.json
```

触发 llm 的 `prefix` 阶段。
这一步消费 `encoded_payload + budget_value`，运行前 2 层 prefix，并输出 `budget_token_prefix_feature`：

```bash
curl -X POST http://127.0.0.1:9012/a2a/execute \
  -H 'Content-Type: application/json' \
  --data @simlingo_agents/smoke_test/llm_prefix_execute.json
```

触发 scheduler 的 `plan` 阶段。
这一步消费 `budget_token_prefix_feature + budget_value`，生成 `execution_plan`：

```bash
curl -X POST http://127.0.0.1:9013/a2a/execute \
  -H 'Content-Type: application/json' \
  --data @simlingo_agents/smoke_test/scheduler_plan_execute.json
```

触发 llm 的 `final` 阶段。
这一步消费 `encoded_payload + budget_value + execution_plan`，输出最终驾驶决策：

```bash
curl -X POST http://127.0.0.1:9012/a2a/execute \
  -H 'Content-Type: application/json' \
  --data @simlingo_agents/smoke_test/llm_final_execute.json
```

## 5. 查看每一段 NATS 输出

查看 encoder 输出摘要：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.encoded_tokens \
  --durable workflow-simlingo-encoded-tokens-check \
  --summary
```

查看 scheduler `budget` 输出，也就是 llm `prefix` 输入摘要：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.llm_prefix_input \
  --durable workflow-simlingo-llm-prefix-input-check \
  --summary
```

查看 llm `prefix` 输出摘要：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.llm_prefix_output \
  --durable workflow-simlingo-llm-prefix-output-check \
  --summary
```

查看 scheduler `plan` 输出，也就是 llm `final` 输入摘要：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.llm_final_input \
  --durable workflow-simlingo-llm-final-input-check \
  --summary
```

查看 llm `final` 输出摘要：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.llm_output \
  --durable workflow-simlingo-llm-output-check \
  --summary
```

如果想把结构化 numpy 解码后再看摘要，可以加 `--decode`：

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once \
  --subject workflow.simlingo.llm_output \
  --durable workflow-simlingo-llm-output-check-decode \
  --decode \
  --summary
```

## 6. 一次性顺序执行

```bash
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py publish-minimal-input
curl -X POST http://127.0.0.1:9011/a2a/execute -H 'Content-Type: application/json' --data @simlingo_agents/smoke_test/encoder_execute.json
curl -X POST http://127.0.0.1:9013/a2a/execute -H 'Content-Type: application/json' --data @simlingo_agents/smoke_test/scheduler_budget_execute.json
curl -X POST http://127.0.0.1:9012/a2a/execute -H 'Content-Type: application/json' --data @simlingo_agents/smoke_test/llm_prefix_execute.json
curl -X POST http://127.0.0.1:9013/a2a/execute -H 'Content-Type: application/json' --data @simlingo_agents/smoke_test/scheduler_plan_execute.json
curl -X POST http://127.0.0.1:9012/a2a/execute -H 'Content-Type: application/json' --data @simlingo_agents/smoke_test/llm_final_execute.json
sudo docker exec simlingo-encoder-agent python /app/tools/nats_smoke.py fetch-once --subject workflow.simlingo.llm_output --durable workflow-simlingo-llm-output-check --summary
```

## 7. 预期结果

- `encoder_agent` 返回 `status=success`，并向 `workflow.simlingo.encoded_tokens` 发布 `encoded_payload`
- `scheduler_agent` 的 `budget` 阶段返回 `status=success`，并向 `workflow.simlingo.llm_prefix_input` 发布：
  - `encoded_payload`
  - `budget_value`
- `llm_agent` 的 `prefix` 阶段返回 `status=success`，并向 `workflow.simlingo.llm_prefix_output` 发布：
  - `encoded_payload`
  - `budget_value`
  - `budget_token_prefix_feature`
- `scheduler_agent` 的 `plan` 阶段返回 `status=success`，并向 `workflow.simlingo.llm_final_input` 发布：
  - `encoded_payload`
  - `budget_value`
  - `execution_plan`
- `llm_agent` 的 `final` 阶段返回 `status=success`，并向 `workflow.simlingo.llm_output` 发布：
  - `llm_payload.speed_wps`
  - `llm_payload.route`
  - `llm_payload.driving_features`
  - `llm_payload.execution_plan_applied`
