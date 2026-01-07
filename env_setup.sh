#!/bin/bash

# === [新增] Hugging Face 国内镜像配置 ===
# 解决 ConnectionResetError 问题
export HF_ENDPOINT="https://hf-mirror.com"
# 启用高速下载 (可选，如果未安装 hf_transfer 库会自动回退到普通下载)
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1. 设置你的项目根目录 (根据你的 ls 输出，是 simlingo-adaption)
export WORK_DIR=~/simlingo-adaption

# 2. 设置 CARLA 根目录 (根据 setup_carla.sh 的安装位置)
# 注意：后续移动到了 /data 目录
export CARLA_ROOT=/data/carla0915

# 3. 设置 PYTHONPATH
# 这让 Python 能找到 CARLA 的库、Bench2Drive 组件以及你的项目代码
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}:${WORK_DIR}

echo "环境配置已激活！"
echo "HF_ENDPOINT: $HF_ENDPOINT (已启用国内镜像)"
echo "CARLA_ROOT: $CARLA_ROOT"
echo "WORK_DIR: $WORK_DIR"