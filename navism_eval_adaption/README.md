# NAVSIM Smart Assigner Eval

这套代码用于在 NAVSIM 本地 PDM 评测里接 `smart_assigner` 的完整 stage2 模型。

## 结构

- `agent.py`
  NAVSIM agent。每个 sample 内按 4 帧顺序重建历史，然后输出第 4 帧轨迹。
- `bootstrap.py`
  加载 stage2 checkpoint 和训练时 `.hydra/config.yaml`。
- `target_points.py`
  用 NAVSIM route centerline 为 4 帧历史分别生成 `target_point / next_target_point`。
- `input_builder.py`
  把单帧 NAVSIM 输入转成 SimLingo 的 `DrivingInput`。
- `trajectory_adapter.py`
  把模型 `route` 重采样成 NAVSIM 评测使用的 4s / 10Hz `Trajectory`。
- `run_metric_caching.py`
  复用 NAVSIM 官方 `run_metric_caching.py` 的本地封装入口。
- `run_pdm_score.py`
  复用 NAVSIM 官方 `run_pdm_score.py` 的本地封装入口。

## 运行

先建 metric cache：

```bash
PYTHONPATH=/home/yyj/simlingo-adaption:/home/yyj/navsim \
conda run -n simlingo python /home/yyj/simlingo-adaption/navism_eval/run_metric_caching.py \
  --config /home/yyj/simlingo-adaption/navism_eval/configs/navtest_stage2_smoke.yaml
```

再跑 PDM 评测：

```bash
PYTHONPATH=/home/yyj/simlingo-adaption:/home/yyj/navsim \
conda run -n simlingo python /home/yyj/simlingo-adaption/navism_eval/run_pdm_score.py \
  --config /home/yyj/simlingo-adaption/navism_eval/configs/navtest_stage2_smoke.yaml
```

额外的 hydra override 可以重复传 `--override key=value`。
配置里的 `navsim_log_path / original_sensor_path / metric_cache_path` 会直接覆盖 NAVSIM 默认目录推断。
`navtest_stage2_full.yaml` 现在默认走 2-GPU `ray_distributed` 并行，比顺序跑快很多。

如果本地 `test_sensor_blobs` 没下完整，可以先补官方测试集：

```bash
cd /home/yyj/navsim/download
bash download_test.sh
```

## 当前约束

- 不改 `smart_assigner` 原始推理逻辑。
- 每个 sample 独立 reset，不跨 sample 传历史。
- `g_hist` 由 4 帧累计；最终决策读到的 `route/speed dev` 是 `1 -> 2`。
- 主导航条件固定走 `target_point`，不使用 navsim `driving_command` 作为 prompt。
- `smoke` 配置默认指向 `2021.05.25.14.16.10_veh-35_01690_02183`，这条 log 在 `test_navsim_logs` 里有充足的可评测 route window。
- 当前本地 `test_sensor_blobs` 下载不完整，和 `smoke` 配置还对不上；需要先重新补齐 `download_test.sh` 才能真正跑通评测。
