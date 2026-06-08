# SimLingo LLM Agent

负责执行语言侧 Qwen2 推理与 driving 头解析，并支持两阶段推理：

- `prefix` 阶段：输入 `encoded_payload + budget_value`，输出 `budget_token_prefix_feature`
- `final` 阶段：输入 `encoded_payload + budget_value + execution_plan`，输出最终驾驶决策

调用方式与 `Agent_Template` 一致：

- HTTP 触发：`POST /a2a/execute`
- NATS 输入：一次只消费一个 subject 上的一条消息，但消息体按阶段承载不同内容
  - `prefix` 默认输入：`workflow.simlingo.llm_prefix_input`
  - `final` 默认输入：`workflow.simlingo.llm_final_input`
- NATS 输出：
  - `prefix` 默认输出：`workflow.simlingo.llm_prefix_output`
  - `final` 默认输出：`workflow.simlingo.llm_output`

注意：

- `llm_agent` 并不是同时监听两类输入；它每次执行只处理一个阶段。
- `POST /a2a/execute` 时可以通过 `metadata.llm_phase` 选择 `prefix` 或 `final`。
- 如果不传 `metadata.nats_in_subject / nats_out_subject`，agent 会按 `llm_phase` 自动选择对应的默认 subject。

默认端口：`9012`
