#!/bin/bash

# Configuration for local training on 2x 4090

# Source bashrc to initialize conda
source ~/.bashrc

# Activate environment
# Ensure the 'simlingo' environment is created and available

conda activate simlingo

# Set paths based on start_eval_simlingo.py and local environment
export CARLA_ROOT="/data/carla0915"
export WORK_DIR="/home/yangyujia/simlingo-adaption"
export REPO_ROOT="$WORK_DIR"

# Construct PYTHONPATH
# Including CARLA PythonAPI, the egg file, and the repo root
# Note: Adjust the egg file name if your CARLA version differs slightly (e.g., python version)
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla"
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# Environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export NCCL_DEBUG=INFO
export WANDB__SERVICE_WAIT=300
# Adjust threads for local machine
export OMP_NUM_THREADS=16 # Adjusted for local CPU cores
export OPENBLAS_NUM_THREADS=1

# Print configuration
echo "Work Dir: $WORK_DIR"
echo "Carla Root: $CARLA_ROOT"
echo "Python Path: $PYTHONPATH"

# Training command
# Adjusted for 2 GPUs (gpus=2)
# batch_size=4 is a safe starting point for 24GB VRAM. Increase if memory allows.
echo "Starting training on $(hostname) with 2 GPUs..."

# Run from the repository root
cd "$WORK_DIR"

# gpu = 1 test
# config: expriment/simlingo_seed1.yaml
python simlingo_training/train.py \
    experiment=debug \
    data_module.batch_size=4 \
    gpus=1 \
    name=simlingo_seed1_local
