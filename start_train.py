import os
import sys
from pathlib import Path
import runpy

# ================= 配置区域 =================
# 1. 你的 Carla 路径 (参考你提供的脚本)
CARLA_ROOT = "/data/carla0915" 
PROJECT_DIR  = "/home/yangyujia/simlingo-adaption"

# 2. 想要调试的参数覆盖 (Overrides)
# 这里的写法等同于命令行参数
DEBUG_ARGS = [
    "experiment=debug",         # 实验配置文件名
    "name=debug_session",                # 实验名称
    "data_module.batch_size=2",          # 调小 batch size      
    "gpus=1",                            # [关键] 单卡调试，禁用 DDP 分布式
    "debug=True",                        # 开启 debug 模式
    # 如果路径有问题，可以在这里强制指定绝对路径来测试：
    # "data_module.base_dataset.data_path=/data/simlingo" 
]
# ===========================================

def setup_environment():
    """模拟复杂的环境变量设置"""
    # 设置环境变量 (供 Hydra 或子进程使用)
    os.environ["WORK_DIR"] = str(PROJECT_DIR)
    os.environ["REPO_ROOT"] = str(PROJECT_DIR)
    os.environ["CARLA_ROOT"] = CARLA_ROOT

    scenario_runner_path = os.path.join(PROJECT_DIR, "scenario_runner")
    leaderboard_path = os.path.join(PROJECT_DIR, "leaderboard")
    os.environ["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_path)
    os.environ["LEADERBOARD_ROOT"] = str(leaderboard_path)
    
    paths_to_add = [
        str(PROJECT_DIR),              # Simlingo 根目录
        str(scenario_runner_path),      # Scenario Runner
        str(leaderboard_path),          # Leaderboard
        f"{CARLA_ROOT}/PythonAPI/carla" # Carla PythonAPI
    ]
    # 执行添加路径操作 
    for p in paths_to_add[::-1]:
        if p not in sys.path:
            sys.path.insert(0, p) # 插入头部，确保优先加载
            print(f"Added to sys.path: {p}")

    env_vars = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29505",       # 建议改个非默认端口(29501)，避免冲突
            "NCCL_DEBUG": "INFO",
            "WANDB__SERVICE_WAIT": "300",
            "OMP_NUM_THREADS": "16",      # 限制 CPU 线程数，防止过载
            "OPENBLAS_NUM_THREADS": "1"
        }
    for key, val in env_vars.items():
        os.environ[key] = val

def main():
    setup_environment()

    # 模拟命令行参数 sys.argv
    # 脚本名 + 我们定义的参数
    script_path = os.path.join(PROJECT_DIR, "simlingo_training", "train.py")
    if not os.path.exists(script_path):
        print(f"Error: 找不到训练脚本: {script_path}")
        return
    sys.argv = [str(script_path)] + DEBUG_ARGS
    
    print(f"Starting Training Debug with args:\n{sys.argv}")
    print("-" * 50)

    try:
        # [核心修改] 使用 run_path 替代 import
        # run_name="__main__" 骗过 Hydra，让它以为自己是作为主脚本运行的
        # 这样它就会去文件系统里找 config，而不是去 import 包
        runpy.run_path(str(script_path), run_name="__main__")
        
    except Exception as e:
        print("运行中捕获异常:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()