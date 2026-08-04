# SimLingo 视觉 Token 剪枝说明与真实压缩方案调研

## 1. 文档目的

本文说明 SimLingo split-agent 推理链路中现有视觉 token 剪枝的实现、数据流和性能边界，并调研能够真正缩短 token 序列的方法。重点回答两个问题：

1. 当前实现是否减少了输入给 LLM 的视觉信息量、显存占用或推理时间？
2. 如何将逻辑 mask 改成物理 token 压缩，以及如何进一步减少视觉编码器的计算？

这里的 token prune 专指视觉 token，不涉及对话历史或文本上下文裁剪。

## 2. 当前完整推理链路

每个 CARLA 帧按下列顺序执行：

```text
CARLA 图像与车辆状态
  -> Remote Agent 构造 prompt、camera_images 和 token_prune 配置
  -> Encoder Agent：InternVL vision encoder 提取全部视觉 token
  -> VisualTokenPruner：选择保留 token，其他 embedding 置零并生成 keep_mask
  -> Scheduler Agent：根据视觉摘要决定 budget
  -> LLM Agent prefix：应用 keep_mask，运行前若干层并提取 budget token 特征
  -> Scheduler Agent：产生逐层/逐头 execution_plan
  -> LLM Agent final：再次应用同一 keep_mask，重新运行完整 LLM
  -> Driving adaptor：输出 route 和 speed waypoints
  -> PID：输出车辆控制量
```

相关入口：

- `team_code_adaption/agent_simlingo_remote.py::_build_remote_payload`
- `simlingo_agents/bench2drive_mcp_server/remote_inference.py::_infer_async`
- `simlingo_agents/encoder_agent/fast_api/model_runtime.py::encode_visual`
- `simlingo_agents/llm_agent/fast_api/model_runtime.py::run`

## 3. 当前剪枝配置如何生效

评测 YAML 中配置：

```yaml
eval:
  token_prune:
    prune_ratio: 0.25
```

`start_eval_simlingo_adaption.py` 将比例写入 `SIMLINGO_EVAL_TOKEN_PRUNE_RATIO`。Remote Agent 每帧生成：

```python
{
    "mode": "prune2drive",
    "prune_ratio": 0.25,
    "min_keep": 1,
}
```

并将其放入 Encoder 请求的 `payload["token_prune"]`。请求级配置覆盖 Encoder 容器的默认配置，因此不同实验不需要重启 Encoder。

如果请求没有覆盖配置，则使用：

```text
ENCODER_TOKEN_PRUNE_MODE
ENCODER_TOKEN_PRUNE_RATIO
ENCODER_TOKEN_PRUNE_MIN_KEEP
```

需要特别注意：`simlingo_agents/docker-compose.yml` 默认是 `mode=origin, ratio=0.1`。`origin` 会保留全部 token，所以这个默认组合实际不执行剪枝。

## 4. 当前 VisualTokenPruner 算法

核心实现位于：

```text
simlingo_adaption_training/models/encoder/visual_token_pruner.py
```

输入视觉特征为 `[B * num_patches, N, D]`。每个 image patch 独立选择 token，保留数量为：

```text
keep_num = min(N, max(min_keep, ceil(N * (1 - prune_ratio))))
```

支持的模式：

| 模式 | 行为 |
|---|---|
| `origin` | 保留全部 token，忽略非零 `prune_ratio` |
| `random` | 为 token 生成随机分数并保留 top-k，用作消融基线 |
| `prune2drive` | 基于特征余弦距离做贪心多样性选择 |
| `off` | 配置解析时转换为 `origin` |

### 4.1 `prune2drive` 的选择逻辑

算法先对 token 特征做 L2 归一化，再计算两两余弦距离：

```text
d(i, j) = 1 - cosine(x_i, x_j)
```

第一次选择最近邻距离最大的 token；后续每次选择距离已选集合最远的 token，直到得到 `keep_num` 个 token。这接近 greedy farthest-point sampling，目标是保留特征空间中相互差异较大的 token。

该方法当前没有显式使用道路区域、目标点、物体类别、文本 prompt 或 LLM attention。因此它更准确的描述是“视觉特征多样性采样”，而不是任务条件化的驾驶重要性评分。

算法还存在两个工程特征：

- 需要构造 `[N, N]` 距离矩阵，并进行 `keep_num` 轮选择；其选择开销可能抵消一部分后续加速，必须实测。
- 返回的 `visual_token_scores` 是最后一轮的候选距离，不是稳定的全局重要性排序。

## 5. 当前实现为什么不是真正的 token 压缩

选出 `keep_mask` 后，当前代码执行：

```python
masked_features = image_features.clone()
masked_features[~keep_mask] = 0.0
```

Encoder 仍然返回原始形状的 `visual_embeds`，同时返回：

```text
visual_token_keep_mask
visual_token_scores
visual_token_prune
visual_token_keep_ratio
```

LLM Agent 仍然为全部 `<IMG_CONTEXT>` 位置填入视觉 embedding，然后只把被剪位置的 attention mask 设置为 `False`：

```python
updated_mask[batch_idx, visual_positions] = keep_mask[batch_idx, :count]
```

所以当前方案具有以下性质：

| 项目 | 当前实现是否减少 |
|---|---|
| 原始相机像素输入 | 否 |
| Vision Encoder 输入 patch 数 | 否 |
| Vision Encoder 计算量 | 否，剪枝发生在 `extract_feature()` 之后 |
| Encoder 输出张量的 token 数 | 否，形状不变，被剪位置置零 |
| NATS 传输数据量 | 基本不减少，仍传完整 embedding 和额外 mask/scores |
| LLM 的逻辑有效视觉信息 | 是，被剪 token 不能作为有效 key/value 参与注意力 |
| LLM 序列长度 | 否，所有 `<IMG_CONTEXT>` 位置仍存在 |
| Dense attention 理论矩阵尺寸 | 否，仍由原始序列长度决定 |
| 实际推理时间 | 不保证减少，可能只改变输出语义 |

普通 dense Transformer 即使收到带洞的 attention mask，通常仍会为完整序列构造 Q/K/V，并按原序列形状运行 attention 和 FFN。除非底层 kernel 显式执行 unpadding/variable-length attention，否则 mask 不等价于物理减算。

此外，prefix 阶段设置了 `use_cache=False`，final 阶段会从头重新运行模型。因此同一帧中 LLM 前缀层会计算两次，当前 prune mask 也会被应用两次，但不会复用第一次计算结果。

## 6. 当前实现对 Scheduler 的附带影响

Scheduler budget 阶段对视觉 embedding 直接求全 token 平均：

```python
visual_summary = visual_embeds.reshape(B, -1, D).mean(dim=1)
```

由于被剪 token 已置零，但分母仍是原始 token 数，实际得到：

```text
sum(kept_tokens) / original_token_count
```

而不是 masked mean：

```text
sum(kept_tokens) / kept_token_count
```

因此 prune ratio 越高，视觉摘要模长通常越小，并会间接影响 scene novelty、history similarity 和 budget 决策。这种耦合未必是设计意图。改成物理压缩时，应同步明确 Scheduler 需要“保留 token 均值”还是“包含保留比例信息的缩放均值”。

## 7. 真正压缩 token 的候选方法

### 7.1 方案 A：Encoder 后 top-k gather，并重建 LLM 输入序列

这是最适合当前代码、改动风险最低的第一步。

Encoder 完成视觉特征提取和 token 选择后，不再返回原尺寸零向量，而是物理收集保留项：

```python
kept_embeds = visual_embeds[keep_mask]       # [K, D]
kept_positions = original_positions[keep_mask]
```

随后同步缩短：

- `<IMG_CONTEXT>` 的数量；
- `input_ids`；
- `attention_mask`；
- `visual_embeds`；
- 必要的空间位置元数据。

对于当前主要推理场景 `B=1`，可以直接输出紧凑序列。以后支持批处理时，可采用：

- batch 内按最大 `K` padding；或
- packed sequence 加 `cu_seqlens`，配合 variable-length attention。

效果判断：

| 指标 | 预期 |
|---|---|
| 输入给 LLM 的视觉 token 数 | 确实减少为 `K` |
| NATS payload | 明显下降，可不再发送完整零 embedding 和 dense scores |
| LLM attention/FFN 计算 | 随总序列长度下降；attention 部分近似随长度平方下降，FFN 近似线性下降 |
| Vision Encoder 时间 | 不变 |
| 是否必须训练 | 可先不训练验证，但较高压缩率通常需要微调 |

这是本项目最值得优先实现和基准测试的方案。

实现时必须保留 token 的原始空间位置。InternVL 的视觉特征已经包含位置编码，但物理重排或合并后仍应保存 `kept_indices`，用于可视化、调试，以及未来显式位置编码处理。建议按原空间顺序 gather，而不是按重要性分数排序，以减少分布变化。

#### 7.1.1 原理和张量变化

设原始视觉特征为 `V in R^[B,N,D]`，选择器给出每个样本的索引 `I_b`，其长度为 `K_b`。物理压缩不是把未选元素乘零，而是执行索引收集：

```text
V_compact[b] = V[b, I_b, :]          # [K_b, D]
```

若每个样本使用固定压缩率，则通常 `K_b=K`，可直接得到 `[B,K,D]`。如果 `K_b` 随场景变化，则需要 padding 或 packed sequence。为了保持视觉空间顺序，选择 top-k 后应对索引升序排序：

```python
topk = torch.topk(scores, k=keep_num, dim=1).indices
kept_indices = topk.sort(dim=1).values
compact = visual_embeds.gather(
    1, kept_indices.unsqueeze(-1).expand(-1, -1, visual_embeds.size(-1))
)
```

这里的 `scores` 可以来自现有 `prune2drive`、轻量 scorer 或 attention。压缩机制与评分方法应拆成两个独立模块，便于确认收益究竟来自“选得好”还是“序列真的变短”。

#### 7.1.2 如何重建当前 InternVL 输入

当前 tokenizer 在 `<image>` 位置预先展开固定数量的 `<IMG_CONTEXT>`。如果视觉 embedding 从 `N` 变成 `K`，必须让占位符数量也变成 `K`，否则 `_replace_visual_tokens()` 会报数量不匹配，或保留无意义的空位置。

推荐在 Encoder 中采用以下顺序：

1. 先提取视觉特征并选出 `K` 个 token。
2. 根据实际 `K` 构造 `<img>` + `K * <IMG_CONTEXT>` + `</img>`。
3. 再执行 tokenizer，得到真正较短的 `input_ids` 和 `attention_mask`。
4. 返回 `[B,K,D]` 的 `visual_embeds`、`kept_indices` 和原始网格大小。

这比“先生成 N 个占位符，再从 tokenized 序列中删除位置”更不容易破坏 `<img>` 边界、文本 special token 和 waypoint token。若为了减小改动必须后删，则需要对每个 batch 同步 gather `input_ids`、`attention_mask` 和所有与序列位置相关的元数据。

建议的新 payload 结构为：

```python
{
    "visual_embeds": compact_embeds,       # [B, K, D]
    "visual_token_indices": kept_indices,  # [B, K]
    "visual_grid_shape": [grid_h, grid_w],
    "visual_token_original_count": N,
    "visual_token_compact_count": K,
    "tokenized": {
        "input_ids": compact_input_ids,
        "attention_mask": compact_attention_mask,
    },
}
```

#### 7.1.3 为什么能够缩短 LLM 时间

设文本、预算和 driving token 共 `T` 个，视觉 token 从 `N` 降为 `K`。LLM 总长度从 `S_old=T+N` 变为 `S_new=T+K`。对标准 dense Transformer：

```text
attention 主项：O(S^2 * D)
QKV/输出投影与 FFN 主项：O(S * D^2)
```

因此真实收益由总序列缩短比例决定，而不只是视觉 token 的保留比例。例如视觉 token 减半，并不意味着完整 LLM 必然加速一倍；文本和 driving token、固定 kernel 开销、GPU 利用率都会稀释收益。当前 pipeline 还会分别执行 prefix 和 final 两次 forward，因此短序列会同时作用于两次 LLM 调用。

该方案不会缩短 `extract_feature()`，因为选择发生在完整 Vision Encoder 输出之后。但它会减少 Encoder 到 Scheduler/LLM 的序列化、NATS payload 和反序列化成本。

#### 7.1.4 实现风险

- 现有 checkpoint 训练时看到固定数量、固定空间顺序的视觉 token，直接改变数量可能造成分布偏移。
- 对绝对位置敏感的实现不能仅 gather embedding 而丢失位置；至少应保持原顺序，并验证模型内部位置编码方式。
- 多 patch 场景应决定“每个 patch 固定保留 K”还是“所有 patch 竞争总预算 K_total”。后者压缩效率更高，但可能把某个相机/视野全部删掉。
- `prune_ratio=0` 必须严格退化成原始路径，作为数值一致性测试。

### 7.2 方案 B：紧凑序列配合 FlashAttention variable-length kernel

仅仅 gather 后再 padding，批量推理时仍可能被 batch 中最长样本拖累。FlashAttention 官方实现提供 `flash_attn_varlen_func`，可以用 packed token 和 cumulative sequence lengths 跳过 padding，从而让物理 token 数变化落实到 kernel 工作量上。[FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)

对当前项目而言，这一方案有两个前提：

1. InternVL/LLaMA 的 attention 实现必须能接入 varlen kernel，而不是只接受普通二维 mask；
2. 自定义 `execution_plan`、逐头/逐层调度和位置编码逻辑必须保持兼容。

当前 `B=1` 时，先做方案 A 已经能获得真实短序列，varlen 的额外收益有限；它更适合未来的多请求 batching。

#### 7.2.1 Packed/varlen 的基本逻辑

普通 batch 会把每个样本 padding 到 `max(K_b)`：

```text
sample lengths = [80, 32, 64]
dense batch     = [3, 80, D]
```

即使 mask 掉 padding，某些 dense kernel 仍按 `3 * 80` 个位置工作。varlen attention 则把有效 token 连续拼接：

```text
packed_tokens = concat(sample_0, sample_1, sample_2)  # [176, D]
cu_seqlens    = [0, 80, 112, 176]
max_seqlen    = 80
```

kernel 使用 `cu_seqlens` 确定样本边界，只对各自样本内部做 attention。这样既不会跨样本注意，也不会为 padding token 计算 attention。

#### 7.2.2 在本项目中的接入方式

不能只把 `attention_mask` 换个格式；需要从 LLM 每层 attention 的 Q/K/V 入口接入 varlen kernel：

1. 将压缩后的语言序列按有效长度 unpad，得到 packed hidden states。
2. 生成 `indices`、`cu_seqlens` 和 `max_seqlen`。
3. 在每层 attention 中调用 varlen kernel。
4. attention 输出后按 `indices` pad 回原 batch 布局，供残差、FFN、budget position 和 driving adaptor 使用；或者让整层都保持 packed 布局。

更彻底的实现是让 attention 和 FFN 全程使用 packed tokens；仅 attention unpad/pad 会引入额外 gather/scatter，且 FFN 仍可能处理 padding。当前 `execution_plan` 是逐层、逐头控制逻辑，接入时必须确认其 batch/head 维度能映射到 packed 序列。

#### 7.2.3 什么时候值得做

- `B=1` 且已物理缩短序列：几乎没有 padding，方案 A 已提供主要收益。
- `B>1` 且各帧保留数量差异大：varlen 可避免最长样本主导整个 batch。
- token 数很少或 batch 很小：unpad/pad 和 kernel launch 开销可能超过节省的计算，必须 benchmark。

### 7.3 方案 C：Token Merging，而不是直接丢弃

[Token Merging（ToMe）](https://arxiv.org/abs/2210.09461)通过快速匹配合并相似 token，逐层缩短 ViT 序列。与硬删除相比，它将被删 token 的信息聚合进保留 token，通常精度损失更小。论文报告在其测试模型上可接近 2 倍吞吐，并保持较小精度下降；这些数字不能直接外推到 SimLingo，必须在目标 GPU 和驾驶闭环上复测。

可在本项目中采用两种位置：

- **Vision Encoder 输出后合并**：将相似视觉 token 加权合并为 `K` 个 token，再输入 LLM。实现相对简单，只加速 LLM，不加速 Vision Encoder。
- **Vision Encoder 若干层之间逐步合并**：后续 ViT 层处理更短序列，可同时减少视觉编码时间和 LLM 输入；需要侵入 InternVL vision backbone，并验证预训练权重兼容性。

如果担心当前 hard prune 丢失道路边缘、信号灯或小目标，merge 通常比直接 top-k drop 更值得尝试。

#### 7.3.1 ToMe 的核心逻辑

ToMe 不给每个 token简单打一个“删/留”标签，而是把相似 token 成对匹配。典型的一轮做法是：

1. 将 token 分成集合 A 和 B，例如按奇偶位置二分，避免构造完整聚类。
2. 对归一化 token 特征计算 A 到 B 的相似度。
3. 为每个 A token 找到最相似的 B token。
4. 选出相似度最高的 `r` 对进行 merge，其余 token 原样保留。
5. 每 merge 一对，序列长度减少 1。

若 token `x_i` 代表 `s_i` 个原始 patch，token `x_j` 代表 `s_j` 个原始 patch，则合并可写为：

```text
x_merge = (s_i * x_i + s_j * x_j) / (s_i + s_j)
s_merge = s_i + s_j
```

维护 `size` 很重要：普通平均会让经过多次合并的大区域和单 patch 拥有相同权重。ToMe 的 proportional attention 会在 attention logits 中补偿 token size，使合并后的 token 表示其覆盖的原始 token 数。

#### 7.3.2 Encoder 输出后 merge 的实现

这是对当前项目侵入较小的版本：

```python
features = extract_feature(images)       # [B, N, D]
merged, source_map, token_size = tome(features, r=N-K)
# merged: [B, K, D]
```

随后像方案 A 一样只生成 `K` 个 `<IMG_CONTEXT>`。`source_map` 记录每个合并 token 对应哪些原始网格，可用于可视化和定位小目标丢失问题。

这个版本减少 LLM 和 NATS 成本，但 ToMe 自身发生在 Vision Encoder 末端，所以视觉编码时间不变。与当前 `prune2drive` 相比，它避免彻底丢弃非保留 token，但需要测试 LLM 是否能正确解释混合后的 embedding 分布。

#### 7.3.3 Vision Encoder 内逐层 merge

可在若干 ViT block 后执行：

```text
patch embedding: N=256
block 0..5:      N=256
merge 64:        N=192
block 6..11:     N=192
merge 64:        N=128
remaining blocks N=128
```

越早 merge，后续层节省越多，但早层特征语义较弱、误合并风险更大。适合先从中后层、较小 `r` 开始。还需保护 class/global token、特殊 register token，不让它们参与普通 patch merge。

#### 7.3.4 与驾驶任务的关系

merge 比 hard drop 更可能保留背景中累积的信息，但相似特征合并仍可能把空间上相距很远的区域混在一起。自动驾驶对精确空间位置敏感，建议给匹配分数增加空间约束：

```text
match_score(i,j) = cosine(feature_i, feature_j) - lambda * spatial_distance(i,j)
```

或者只允许邻近网格/同一相机 patch 内合并。这样压缩率可能稍低，但更容易保持车道线、信号灯和行人位置。

### 7.4 方案 D：EViT/DynamicViT 式渐进式物理裁剪

[DynamicViT](https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html)在多个 ViT 层加入轻量重要性预测器，逐层移除不重要 token。论文报告裁掉约 66% token 时，FLOPs 降低约 31%--37%，吞吐提升超过 40%，但这些结果来自其模型和分类任务。[EViT](https://arxiv.org/abs/2202.07800)则根据 class-token attention 保留重要 token，并把非重要 token 融合为一个辅助 token；论文同样报告了真实吞吐提升。

这种方法比“Vision Encoder 全算完后再剪”更有机会降低端到端时间，因为被删 token 不再进入后续视觉 Transformer 层。但代价是：

- 需要修改 InternVL vision encoder 的 block forward；
- DynamicViT 的预测器通常需要训练；
- EViT 依赖 class-token attention，未必直接适合驾驶 VLM；
- 驾驶中的小目标、安全关键目标要求专门的训练目标和闭环验证。

若端到端 profile 显示 Vision Encoder 占比很高，应在完成方案 A 后重点评估这一方向。

#### 7.4.1 DynamicViT：学习逐层 keep/drop

DynamicViT 在选定的视觉 Transformer 层后加入轻量 prediction module。预测器同时使用 token 局部特征和全局上下文，输出保留概率：

```text
p_i = softmax(MLP([local_feature_i, global_feature]))
decision_i in {keep, drop}
```

训练时离散采样不可直接反向传播，因此通常使用可微近似或 attention masking 来训练决策模块，并加入 token ratio 约束，使各阶段接近目标保留率。推理时则真正 gather `decision_i=keep` 的 token，让后续 block 接收更短张量。

在 SimLingo 中可把 route/target point/budget feature 拼进 predictor，使选择不仅依赖图像，还依赖驾驶任务：

```text
p_i = scorer(visual_i, global_visual, route_feature, target_point, budget)
```

训练损失建议至少包含原驾驶损失、与未剪模型的 feature/output distillation，以及实际保留率正则。单纯用图像分类式 importance 容易删除面积小但安全关键的目标。

#### 7.4.2 EViT：保留高 attention token并融合剩余信息

EViT 使用 class token 对 patch token 的 attention 作为重要性。选择 top-k 后，不是简单丢掉全部低分 token，而是按 attention 权重把它们融合成一个 inattentive token：

```text
x_fused = sum(a_i * x_i) / sum(a_i),  i in dropped_set
output = [important_tokens, x_fused]
```

这让序列从 `N` 变成约 `K+1`，同时保留被裁区域的全局摘要。SimLingo 的视觉 backbone 若没有适合驾驶语义的 class token，可改用：

- global image token；
- LLM budget/driving token 对视觉 token 的 cross-attention；
- learned driving query 对视觉 token 的 attention。

需要避免直接把训练于图像分类的 class-token 重要性当成驾驶重要性。

#### 7.4.3 物理加速成立的条件

训练阶段可以用 mask 保持固定形状，但部署阶段必须执行 gather，使下一层输入从 `[B,N,D]` 变成 `[B,K,D]`。如果部署仍只传 decision mask，就会回到当前项目的问题：语义被屏蔽，但 dense kernel 的序列长度没有下降。

另外，动态 `K_b` 会影响 batching。可选择每阶段固定 keep ratio，使 batch 内 K 相同；或者配合 7.2 的 varlen kernel。固定 ratio 更容易获得稳定 latency，动态阈值则更适合按场景难度分配计算。

### 7.5 方案 E：TokenLearner / Perceiver Resampler / Q-Former

[TokenLearner](https://arxiv.org/abs/2106.11297)通过少量学习到的空间注意力图，把大量视觉 token 聚合成少数自适应 token。[Flamingo](https://arxiv.org/abs/2204.14198)使用 Perceiver Resampler 将可变数量的视觉特征压到固定数量的 latent token；BLIP-2 的 Q-Former 也属于用少量可学习 query 从视觉特征中提取固定长度表示的路线。

应用到 SimLingo，可以在 InternVL vision encoder 与 LLM 之间增加一个 learned resampler：

```text
[B, N, D] -> learned cross-attention/resampling -> [B, K, D], K << N
```

优点：

- LLM 始终收到固定、较短的视觉序列；
- 比直接丢弃 token 更有机会聚合全局信息；
- 容易控制通信量和 LLM latency。

缺点：

- resampler 自身有计算开销；
- 必须训练或至少微调，不能安全地直接插入现有 checkpoint；
- 如果仍放在完整 Vision Encoder 之后，不能降低视觉编码时间。

该方案适合追求高压缩率和稳定固定长度的长期版本。

#### 7.5.1 TokenLearner 的实现原理

TokenLearner 从二维视觉特征图生成 `K` 张空间权重图，每张权重图负责汇聚一种视觉模式。设视觉特征为 `X[h,w,d]`，第 `k` 个学习 token 为：

```text
A_k = sigmoid/softmax(spatial_scorer_k(X))       # [H,W]
z_k = sum_hw(A_k[h,w] * X[h,w,:]) / sum_hw(A_k) # [D]
```

最终从 `H*W=N` 个 patch 得到固定 `K` 个 token。不同权重图可以分别关注道路、车辆、行人或远处交通设施，但是否形成这些语义取决于训练目标。

在当前项目中，可将 TokenLearner 放在 `extract_feature()` 之后、发送 NATS 之前，并用线性层把输出维度保持为 LLM hidden size。tokenizer 固定展开 K 个 `<IMG_CONTEXT>`。由于所有输入 token 都参与了加权汇聚，信息丢失通常比 hard top-k 更平滑，但它仍不降低此前已完成的 Vision Encoder 计算。

#### 7.5.2 Perceiver Resampler 的实现原理

Resampler维护 `K` 个可学习 latent query `Q in R^[K,D]`，以完整视觉 token 为 key/value 做 cross-attention：

```text
Z = softmax((Q Wq) (V Wk)^T / sqrt(d)) (V Wv)  # [B,K,D]
```

可以堆叠若干 cross-attention + FFN 层。无论原视觉 token 是 256、512 还是更多，输出始终为 K，因此非常适合控制 LLM 输入长度和跨 agent payload。其计算量约包含 `O(K*N*D)` 的 cross-attention；当 `K << N` 时通常小于让 LLM 多层处理 N 个视觉 token。

#### 7.5.3 Q-Former 的实现原理

Q-Former 同样使用固定数量的 learnable query，但通常包含更完整的 Transformer 结构：query 之间做 self-attention，并通过 cross-attention 从冻结的视觉特征读取信息。训练时可采用图文对齐、图文匹配或生成目标，使 query 学会提取对语言模型有用的视觉内容。

迁移到 SimLingo 时，训练目标不应只做图文对齐，可直接用现有 route/speed waypoint loss，并增加 teacher distillation：让压缩版 driving feature 和输出接近未压缩模型。还可让 query 条件化于 route、target point 或当前 budget，从固定的通用视觉摘要变成任务相关摘要。

#### 7.5.4 三者的选择

| 方法 | 聚合方式 | 输出长度 | 额外开销 | 更适合的情况 |
|---|---|---:|---:|---|
| TokenLearner | K 张空间权重图加权池化 | 固定 K | 较低 | 保留二维空间结构、追求轻量 |
| Perceiver Resampler | latent query 对视觉 token cross-attention | 固定 K | 中等 | 输入 patch 数可变、需要较强聚合 |
| Q-Former | query self-attention + 视觉 cross-attention | 固定 K | 较高 | 有训练预算、需要强跨模态/任务对齐 |

三者都必须输出真实 `[B,K,D]` 并缩短 `<IMG_CONTEXT>`，否则即使内部产生 K 个摘要，又把它们填回 N 个位置，也得不到 LLM 加速。

### 7.6 方案 F：降低进入 Vision Encoder 的 patch/pixel 数

如果目标是减少“视觉信息的原始输入量”而不只是 LLM token，需要在 vision encoder 之前处理：

- 降低相机分辨率；
- 减少 dynamic image patches / tile 数；
- 增大 patch size 或提高 InternVL downsample ratio；
- 只输入道路、前景或任务相关 ROI；
- 使用多尺度策略：全局低分辨率图 + 少量高分辨率关键区域；
- 对连续帧做 temporal reuse，仅对变化区域重新编码。

这是唯一能直接降低像素处理、patch embedding 和全部视觉 Transformer 层成本的类别，但也最容易损失远处红绿灯、行人和小目标信息。改变分辨率、patch 网格或相机组合还会造成明显的训练/推理分布偏移，通常需要重新训练或针对驾驶数据微调。

#### 7.6.1 降分辨率或增加 patch size

若图像大小为 `H*W`、patch size 为 `P`，初始 patch token 数近似为：

```text
N = (H/P) * (W/P)
```

图像高宽同时缩小一半，token 数约降为四分之一；patch size 加倍也有类似数量效果。这能减少 patch embedding、全部 Vision Transformer 层、NATS 和 LLM 的计算，是最直接的端到端减算方式。

但 InternVL checkpoint 的位置编码、图像预处理和视觉语义是在特定分辨率/patch 配置下训练的。优先尝试模型原生支持的合法 image size、tile 数或 dynamic patch 配置，不应只在运行时任意修改 `patch_size`。

#### 7.6.2 ROI 与多尺度输入

驾驶场景可把有限 token 预算分配成：

```text
低分辨率全景：保持道路布局和大范围上下文
高分辨率 ROI：交通灯、行人、前车、路口、目标路线附近
```

ROI 可以来自检测器、地图/路线投影、光流变化或轻量 proposal 网络。若每个 ROI 被当作独立 patch，必须给模型相机 ID、原图坐标和尺度元数据，否则模型难以判断局部 crop 在全局中的位置。

该方法不是无条件减少“视觉信息”，而是重新分配信息带宽：牺牲低价值区域的分辨率，把预算留给安全关键区域。ROI 生成器的延迟和漏检率必须计入端到端评测。

#### 7.6.3 Patch 级预筛选

可以在重型 ViT 之前增加很轻的模块，例如低分辨率 CNN、浅层 ViT 或图像梯度/显著性计算，先为 patch 打分，只把 top-k patch 送入主 Vision Encoder。其流程是：

```text
image -> cheap scorer -> selected patch pixels/embeddings -> heavy vision encoder
```

这比在 `extract_feature()` 之后剪枝更早，但主 ViT 通常期望规则网格和固定位置编码。实现上需要支持稀疏 patch embedding、携带原始二维坐标，并对主 Vision Encoder 做微调。若只是先计算完整 patch embedding 再删，仍可节省后续 ViT blocks，但不能节省图像预处理和 patch projection。

#### 7.6.4 时序复用

连续驾驶帧高度相似，可缓存上一帧视觉 token，只重算变化区域：

1. 用 ego motion、光流或特征匹配把上一帧 token warp 到当前坐标。
2. 计算低成本变化分数。
3. 对稳定区域复用缓存 token，对变化区域重新编码。
4. 合并为当前帧紧凑 token，并定期强制全量刷新以限制漂移。

该方法理论上能降低平均视觉输入计算，但比单帧压缩复杂：需要处理车辆运动、遮挡、新出现物体、缓存失效和跨帧误差累积。对安全关键场景应设置强制刷新条件，例如路口、急转、遮挡突变或 novelty 超阈值。

## 8. 方法对“输入量”和“推理时间”的影响对比

| 方法 | LLM 视觉 token 数 | NATS 传输量 | Vision Encoder 时间 | LLM 时间 | 是否通常需训练 |
|---|---:|---:|---:|---:|---:|
| 当前零值 + mask | 不变 | 不变/略增 | 不变 | 不保证下降 | 否 |
| Encoder 后 top-k gather | 减少 | 减少 | 不变 | 可真实下降 | 低压缩率可先免训练验证 |
| Gather + varlen attention | 减少 | 减少 | 不变 | batch 场景更容易真实下降 | 主要是工程适配 |
| Encoder 后 Token Merging | 减少 | 减少 | 不变 | 可真实下降 | 可先免训练，建议微调 |
| ViT 内 ToMe/EViT/DynamicViT | 逐层减少 | 减少 | 可下降 | 可下降 | ToMe 可免训练；其他通常需要 |
| TokenLearner/Resampler/Q-Former | 固定为较小 K | 减少 | 若置于末端则不变 | 可显著下降 | 是 |
| 降分辨率/patch/ROI | 从源头减少 | 减少 | 可显著下降 | 可下降 | 通常需要微调 |

必须区分三种“减少视觉输入”：

1. **减少 LLM 看到的视觉信息**：当前 mask 已经做到语义屏蔽，但没有缩短序列。
2. **减少送入 LLM 的视觉张量/token 数**：需要 gather、merge 或 resampler。
3. **减少 Vision Encoder 接收和处理的视觉输入**：需要降分辨率、减少 patch/ROI，或在 ViT 早期物理裁剪。

## 9. 针对本项目的推荐落地顺序

### 阶段 0：先建立可信基线

分别记录以下阶段的 GPU 时间、峰值显存和 payload 字节数：

```text
vision extract_feature
pruner scoring/select
scheduler budget
LLM prefix
scheduler plan
LLM final
端到端 frame latency
```

使用 CUDA events 测 GPU kernel 时间，并在计时前后正确同步；仅使用当前 HTTP wall-clock 统计无法区分排队、NATS、序列化与 GPU 时间。闭环指标至少应包含 driving score、route completion、infraction、控制稳定性和关键场景失败率。

### 阶段 1：实现物理 compaction

推荐首先把现有 `keep_mask` 改成真实 gather：

1. 保持现有 `prune2drive` 作为选择器，先隔离“选择策略”和“物理压缩”的影响。
2. Encoder 返回 `[B, K, D]` 紧凑视觉 embedding 和 `kept_indices`。
3. tokenization 阶段只生成 `K` 个 `<IMG_CONTEXT>`，或在 Encoder 输出前删除未保留位置。
4. LLM 不再依赖 `visual_token_keep_mask`，而是直接处理短序列。
5. Scheduler 使用紧凑 token 做均值，避免零值分母问题。
6. 默认不再通过 NATS 发送 dense `visual_token_scores`；只在 debug 模式发送。

建议先测试 `prune_ratio = 0, 0.1, 0.25, 0.5`。只有同时看到 LLM CUDA 时间、端到端时间或显存下降，才能认定获得了真实性能收益。

### 阶段 2：改进选择策略

如果 `prune2drive` 的 `[N, N]` 距离矩阵开销明显，依次比较：

- 单层轻量 scorer + top-k；
- 基于已有 attention/feature norm 的 top-k；
- ToMe 式相似 token 合并；
- 驾驶任务条件化 scorer，将 route、target point 或 prompt feature 纳入选择。

### 阶段 3：减少 Vision Encoder 时间

只有 profile 证明视觉编码阶段是主要瓶颈后，再修改 backbone：

- 优先尝试中间层 ToMe，因其可以从现有权重开始评估；
- 然后尝试需要训练的 DynamicViT/EViT/TokenLearner；
- 如果仍需更大收益，再评估低分辨率全图 + 高分辨率 ROI 或时序特征复用。

### 阶段 4：消除 prefix 重算

当前 prefix 和 final 是两次独立 forward，且 prefix 使用 `use_cache=False`。如果 execution plan 的设计允许，可以研究保存 prefix hidden state/KV，并让 final 从 prefix 层继续，而不是重算前若干层。这项优化与 token compaction 正交，可能比小比例 prune 带来更稳定的延迟收益，但需要确认 scheduler plan 对已计算层的语义。

## 10. 验收标准

一个方案只有同时满足以下条件，才应称为“真正 token 压缩”：

1. LLM 收到的 `inputs_embeds.shape[1]` 随保留比例下降；
2. `<IMG_CONTEXT>` 的实际数量同步下降；
3. NATS 中不再携带被删除 token 的零 embedding；
4. profiler 显示 LLM attention/FFN 实际 GPU 时间或显存下降；
5. 端到端 frame latency 在足够样本上有稳定下降，而非只看 FLOPs；
6. 驾驶闭环指标保持在可接受范围；
7. 若宣称 Vision Encoder 加速，还必须证明 `extract_feature()` 时间下降。

## 11. 总结

当前实现确实减少了 LLM 可利用的视觉信息，但没有缩短视觉 embedding 或语言序列，因此不能保证减少推理时间，也没有减少 Vision Encoder 输入与计算。

最可行的近期方案是：**沿用现有选择器，但把保留 token 做物理 gather，同步重建 `<IMG_CONTEXT>` 和 attention 序列**。这会真实减少 Encoder 到 LLM 的传输量和 LLM token 数；在当前 `B=1` 推理下，不必先引入复杂的 varlen batching。

若目标进一步包括缩短 Vision Encoder 时间，应把 token reduction 前移到视觉 Transformer 中间层，采用 ToMe、EViT、DynamicViT 等渐进式方法，或从源头减少图像分辨率/patch/ROI。所有论文中的吞吐数字都只应作为方向性参考，最终结论必须来自本项目模型、目标 GPU、split-agent 通信和 Bench2Drive 闭环评测。

## 12. 参考资料

- Bolya et al., [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)
- Rao et al., [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html)
- Liang et al., [Not All Patches are What You Need: Expediting Vision Transformers via Token Reorganizations](https://arxiv.org/abs/2202.07800)
- Ryoo et al., [TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?](https://arxiv.org/abs/2106.11297)
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- Dao-AILab, [FlashAttention official implementation](https://github.com/Dao-AILab/flash-attention)
