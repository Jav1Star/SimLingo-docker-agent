# SimLingo Scheduler Agent

负责两步调度：

- `budget` 阶段：根据 `encoder_agent` 输出估计场景复杂度并给出 `budget_value`
- `plan` 阶段：接收 `llm_agent(prefix)` 输出的 `budget_token_prefix_feature`，生成 `execution_plan`

默认链路：

- `budget` 阶段输入：`workflow.simlingo.encoded_tokens`
- `budget` 阶段输出：`workflow.simlingo.llm_prefix_input`
- `plan` 阶段输入：`workflow.simlingo.llm_prefix_output`
- `plan` 阶段输出：`workflow.simlingo.llm_final_input`
- HTTP 触发：`POST /a2a/execute`
- 默认端口：`9013`

注意：

- `scheduler_agent` 不直接控制 LLM 内部推理。
- 它只输出 `budget_value` 或 `execution_plan`，再交由 `llm_agent` 执行对应阶段。
- `POST /a2a/execute` 时可以通过 `metadata.scheduler_phase` 选择 `budget` 或 `plan`。
