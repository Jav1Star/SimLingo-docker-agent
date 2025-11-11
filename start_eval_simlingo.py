import os
import sys
import subprocess
import time
import ujson
import shutil
from collections import deque
from tqdm.autonotebook import tqdm

"""
本脚本为在本地工作站（单机，双 4090 GPU）上运行 Bench2Drive 评测的调度器。

关键改动（相对 HPC 版 start_eval_simlingo_forHPC.py）：
- 去除 Slurm 提交，改为本地子进程管理与 GPU 轮转并行。
- 强制优先使用 Bench2Drive 内置的 leaderboard 包，避免被 simlingo 根目录下的同名 leaderboard 冲突。
- 修正端口参数：--port 使用 CARLA 世界端口，--traffic-manager-port 使用 TM 端口。
- 支持双卡并发，默认读取环境变量 SIMLINGO_MAX_JOBS 控制并行数。
- 失败重试与可视化目录清理逻辑与 HPC 版对齐；修复重试时日志文件句柄未关闭导致的泄漏。

使用说明：
1) 确认 CARLA 安装目录、权重路径、路由文件等字段在 configs 中已正确设置。
2) 可通过环境变量 SIMLINGO_MAX_JOBS 控制并行任务数（不超过 GPU 数量）。
3) 日志与结果默认写入 eval_results/Bench2Drive/... 路径，viz 子目录每次提交前会清空。
4) 如遇导入报错（TickRuntimeError 等），本脚本已通过 PYTHONPATH 顺序与工作目录设置避免冲突。
"""


GPU_IDS = [0]  # 本地双 4090；如需限制，可通过环境变量 SIMLINGO_MAX_JOBS 控制并行
MAX_PARALLEL_JOBS = 3
POLL_INTERVAL_S = float(os.getenv("SIMLINGO_POLL_INTERVAL", "5.0"))

carla_world_ports = set(range(10000, 20000, 50))
carla_tm_ports = set(range(30000, 40000, 50))


def expand_path(path: str) -> str:
    return os.path.expanduser(path)


def needs_resubmit(job) -> bool:
    result_file = job["result_file"]
    if not os.path.exists(result_file):
        return True
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            evaluation_data = ujson.load(f)
    except Exception:
        return True

    checkpoint = evaluation_data.get("_checkpoint", {})
    progress = checkpoint.get("progress", [])
    if len(progress) < 2 or progress[0] < progress[1]:
        return True

    failure_statuses = {
        "Failed - Agent couldn't be set up",
        "Failed",
        "Failed - Simulation crashed",
        "Failed - Agent crashed",
    }
    for record in checkpoint.get("records", []):
        if record.get("status") in failure_statuses:
            return True

    return False


def cleanup_viz_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def launch_job(job, gpu_id, world_port, tm_port):
    cfg = job["cfg"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    carla_root = expand_path(cfg["carla_root"])
    repo_root = expand_path(cfg["repo_root"])
    bench_root = os.path.join(repo_root, "Bench2Drive")  # 将子进程工作目录切到 Bench2Drive，减少导入冲突风险

    env["CARLA_ROOT"] = carla_root
    addPYTHOPATH = ":".join(
            [
            f"{carla_root}/PythonAPI/carla",
            f"{carla_root}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg",
            repo_root,
            f"{repo_root}/Bench2Drive/leaderboard",
            f"{repo_root}/Bench2Drive/scenario_runner",
            ]
    )

    env["PYTHONPATH"] = addPYTHOPATH
    env["SCENARIO_RUNNER_ROOT"] = f"{repo_root}/Bench2Drive/scenario_runner"
    env["SAVE_PATH"] = job["viz_path"]

    command = [
        sys.executable,
        "-u",
        f"{repo_root}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
        f"--routes={job['route']}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={job['result_file']}",
        "--timeout=600",
        f"--agent={cfg['agent_file']}",
        f"--agent-config={cfg['checkpoint']}",
        f"--traffic-manager-seed={job['seed']}",
        f"--port={world_port}",                 # 世界端口（与 CARLA server 对应）
        f"--traffic-manager-port={tm_port}"    # 交通管理器端口
        #f"--gpu-rank={gpu_id}"
    ]

    stdout = open(job["log_file"], "w", encoding="utf-8")
    stderr = open(job["err_file"], "w", encoding="utf-8")
    stdout.write(" ".join(command) + "\n")
    stdout.flush()

    process = subprocess.Popen(
        command,
        env=env,
        cwd=bench_root, 
        stderr=stderr,
    )

    job["process"] = process
    job["gpu_id"] = gpu_id
    job["ports"] = {world_port, tm_port}
    job["_stdout_handle"] = stdout
    job["_stderr_handle"] = stderr


def finalize_job(job):
    for handle_key in ("_stdout_handle", "_stderr_handle"):
        handle = job.pop(handle_key, None)
        if handle:
            handle.flush()
            handle.close()
    job.pop("process", None)
    job.pop("ports", None)
    job.pop("gpu_id", None)


def main():
    SimlingoPATH = os.path.expanduser("~/simlingo")
    configs = [
        {
            "agent": "simlingo",
            "checkpoint": f"{SimlingoPATH}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
            "benchmark": "bench2drive",
            "route_path": f"{SimlingoPATH}/leaderboard/data/bench2drive_split",
            "seeds": [3],
            "tries": 1,  # 重试次数,1便于调试
            "out_root": f"{SimlingoPATH}/eval_results/Bench2Drive",
            "carla_root": "/data/carla0915",
            "repo_root": f"{SimlingoPATH}",
            "agent_file": f"{SimlingoPATH}/team_code/agent_simlingo.py",
            "team_code": "team_code",
            "agent_config": "not_used",
            "username": os.getenv("USER", "local_user"),
        }
    ]

    job_queue = []
    for cfg in configs:
        route_path = cfg["route_path"]
        routes = [x for x in os.listdir(route_path) if x.endswith(".xml")]

        fill_zeros = 3 if cfg["benchmark"] == "bench2drive" else 2

        for seed in cfg["seeds"]:
            seed = str(seed)

            base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed)
            os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

            for route in routes:
                route_id = route.split("_")[-1][:-4].zfill(fill_zeros)
                route_file = os.path.join(route_path, route)

                viz_path = os.path.join(base_dir, "viz", route_id)
                os.makedirs(viz_path, exist_ok=True)

                result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
                log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
                err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")

                job = {
                    "cfg": cfg,
                    "route": route_file,
                    "route_id": route_id,
                    "seed": seed,
                    "viz_path": viz_path,
                    "result_file": result_file,
                    "log_file": log_file,
                    "err_file": err_file,
                    "tries_initial": cfg["tries"],
                    "tries_remaining": cfg["tries"],
                    "status": "pending",
                }
                job_queue.append(job)

    pending_jobs = deque(job_queue)
    running_jobs = []
    available_gpus = deque(GPU_IDS)
    progress = tqdm(total=len(job_queue))

    
    while pending_jobs or running_jobs:
        for job in list(running_jobs):
            process = job["process"]
            return_code = process.poll()
            if return_code is None:
                continue

            # 回收 GPU 并关闭日志句柄，避免文件描述符泄漏
            available_gpus.append(job["gpu_id"])
            running_jobs.remove(job)
            finalize_job(job)
            

            if not needs_resubmit(job) and return_code == 0:
                job["status"] = "completed"
                progress.update(1)
                continue

            if job["tries_remaining"] > 0:
                job["status"] = "pending"
                pending_jobs.append(job)
                print(f"Resubmitting job {job['route_id']} (tries left: {job['tries_remaining']}).")
                continue

            job["status"] = "failed"
            progress.update(1)
            print(f"Job {job['route_id']} exhausted retries (return code {return_code}).")
            if return_code != 0:
                return
        if len(running_jobs) >= MAX_PARALLEL_JOBS or not available_gpus:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_started = False
        for _ in range(len(pending_jobs)):
            job = pending_jobs.popleft()

            if job["status"] == "completed":
                continue

            if job["tries_remaining"] <= 0:
                job["status"] = "failed"
                progress.update(1)
                print(f"Skipping job {job['route_id']}: no retries left.")
                continue

            if not needs_resubmit(job):
                job["status"] = "completed"
                progress.update(1)
                continue

            used_ports = set()
            for running in running_jobs:
                used_ports.update(running.get("ports", set()))

            try:
                world_port = next(iter(carla_world_ports - used_ports))
                tm_port = next(iter(carla_tm_ports - used_ports))
            except StopIteration:
                pending_jobs.appendleft(job)
                time.sleep(POLL_INTERVAL_S)
                break

            cleanup_viz_dir(job["viz_path"])
            job["tries_remaining"] -= 1
            job["status"] = "running"

            gpu_id = available_gpus.popleft()
            launch_job(job, gpu_id, world_port, tm_port)
            running_jobs.append(job)

            print(
                f"Started job {job['route_id']} on GPU {gpu_id} "
                f"(tries left after launch: {job['tries_remaining']})."
            )
            job_started = True
            break

        if not job_started:
            time.sleep(POLL_INTERVAL_S)

    progress.close()
    print("All jobs processed.")
    # try:
    # finally:
    #     for job in running_jobs:
    #         process = job.get("process")
    #         if process and process.poll() is None:
    #             process.terminate()
    #         finalize_job(job)


if __name__ == "__main__":
    main()