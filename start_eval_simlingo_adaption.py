import os
import sys
import subprocess
import time
import json
import shutil
import argparse
import yaml
from collections import deque
from tqdm.autonotebook import tqdm

try:
    import ujson
except Exception:
    import json as ujson



VULKAN_GPU_ID = {0: 0, 1: 2}  # 映射到 Vulkan 适配的 GPU ID, 太诡异了，每次只能试试
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

    progress = evaluation_data['_checkpoint']['progress']
    if len(progress) < 2 or progress[0] < progress[1]:
        return True

    failure_statuses = {
        "Failed - Agent couldn't be set up",
        "Failed",
        "Failed - Simulation crashed",
        "Failed - Agent crashed",
    }
    for record in evaluation_data['_checkpoint']['records']:
        if record.get("status") in failure_statuses:
            return True

    return False


def cleanup_viz_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_eval_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_latency_settings(data):
    eval_cfg = data.get("eval", data)
    latency_cfg = data.get("latency", eval_cfg.get("latency", {})) or {}
    default_rule_based_cfg = {
        "k_warmup": 5,
        "eta": 0.03,
        "safe_threshold": 0.35,
        "safe_count_threshold": 2,
        "decay_step": 0.1,
        "base_offset": 0.25,
        "base_scale": 0.75,
        "inst_offset": 0.5,
        "inst_scale": 0.5,
        "weights": {
            "base_mean": {"novelty": 0.23, "speed_shift": 0.35, "route_shift": 0.15},
            "base_max": {"novelty": 0.08, "speed_shift": 0.12, "route_shift": 0.08},
            "inst": {"novelty": 0.40, "speed_shift": 0.40, "route_shift": 0.20},
        },
        "normalization": {
            "novelty": {"q10": 0.0001102686, "q90": 0.0150763988},
            "speed_shift": {"q10": 0.0030981766, "q90": 0.7491058707},
            "route_shift": {"q10": 0.1959435195, "q90": 0.7666570544},
        },
    }
    rule_based_cfg_raw = latency_cfg.get("rule_based", {}) or {}
    if not isinstance(rule_based_cfg_raw, dict):
        rule_based_cfg_raw = {}
    rule_based_cfg = dict(default_rule_based_cfg)
    for key, value in rule_based_cfg_raw.items():
        if isinstance(value, dict) and isinstance(rule_based_cfg.get(key), dict):
            merged = dict(rule_based_cfg[key])
            merged.update(value)
            rule_based_cfg[key] = merged
        else:
            rule_based_cfg[key] = value
    weights_raw = rule_based_cfg_raw.get("weights", {}) if isinstance(rule_based_cfg_raw, dict) else {}
    if isinstance(weights_raw, dict):
        for sub_key in ("base_mean", "base_max", "inst"):
            if isinstance(weights_raw.get(sub_key), dict):
                merged = dict(default_rule_based_cfg["weights"][sub_key])
                merged.update(weights_raw[sub_key])
                rule_based_cfg["weights"][sub_key] = merged
    norm_raw = rule_based_cfg_raw.get("normalization", {}) if isinstance(rule_based_cfg_raw, dict) else {}
    if isinstance(norm_raw, dict):
        for sub_key in ("novelty", "speed_shift", "route_shift"):
            if isinstance(norm_raw.get(sub_key), dict):
                merged = dict(default_rule_based_cfg["normalization"][sub_key])
                merged.update(norm_raw[sub_key])
                rule_based_cfg["normalization"][sub_key] = merged
    # 从 yaml 读取 latency 模式配置（random/fixed/rule_based）
    mode = str(latency_cfg.get("mode", "no mode in config")).strip().lower()
    allowed_modes = {"random", "fixed", "rule_based"}
    if mode not in allowed_modes:
        raise ValueError(f"latency.mode must be one of {sorted(allowed_modes)}, got {mode}")
    fix_latency = float(latency_cfg.get("fix_latency", 1.0))
    if mode == "fixed" and not (0.0 <= fix_latency <= 1.0):
        raise ValueError(f"latency.fix_latency must be in [0, 1] for fixed mode, got {fix_latency}")
    return mode, fix_latency, rule_based_cfg


def build_eval_config(args, no_server_launch):
    eval_yaml = load_eval_yaml(args.eval_config)
    cfg = eval_yaml.get("eval", eval_yaml)
    latency_mode, fix_latency, rule_based_cfg = parse_latency_settings(eval_yaml)

    required_keys = [
        "agent",
        "checkpoint",
        "benchmark",
        "route_path",
        "out_root",
        "carla_root",
        "repo_root",
        "agent_file",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in eval config: {missing}")

    seeds = cfg.get("seeds", [3])
    if args.seed is not None:
        seeds = [args.seed]

    eval_cfg = {
        "agent": cfg["agent"],
        "checkpoint": expand_path(cfg["checkpoint"]),
        "benchmark": cfg["benchmark"],
        "route_path": expand_path(cfg["route_path"]),
        "seeds": seeds,
        "tries": int(cfg.get("tries", 0)),
        "out_root": expand_path(cfg["out_root"]),
        "carla_root": expand_path(cfg["carla_root"]),
        "repo_root": expand_path(cfg["repo_root"]),
        "agent_file": expand_path(cfg["agent_file"]),
        "team_code": cfg.get("team_code", "team_code_adaption"),
        "agent_config": cfg.get("agent_config", "not_used"),
        "username": cfg.get("username", os.getenv("USER", "local_user")),
        "no_server_launch": no_server_launch,
        "latency_mode": latency_mode,
        "fix_latency": fix_latency,
        "rule_based_cfg": rule_based_cfg,
    }
    return eval_cfg


def launch_job(job, gpu_id, world_port, tm_port):
    cfg = job["cfg"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    carla_root = expand_path(cfg["carla_root"])
    repo_root = expand_path(cfg["repo_root"])
    bench_root = os.path.join(repo_root, "Bench2Drive")  # 将子进程工作目录切到 Bench2Drive，减少导入冲突风险

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
    env["CARLA_ROOT"] = carla_root
    env["WORK_DIR"] = repo_root
    env["SCENARIO_RUNNER_ROOT"] = f"{repo_root}/Bench2Drive/scenario_runner"
    env["SAVE_PATH"] = job["viz_path"]
    env["LEADERBOARD_ROOT"] = f"{repo_root}/Bench2Drive/leaderboard"
    # 通过环境变量把 mode 与固定延迟值传给 agent
    env["SIMLINGO_EVAL_LATENCY_MODE"] = str(job["latency_mode"])
    env["SIMLINGO_EVAL_FIXED_LATENCY"] = str(job["fix_latency"])
    env["SIMLINGO_EVAL_RULE_BASED_CFG_JSON"] = json.dumps(job["rule_based_cfg"], ensure_ascii=False)
    
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
        f"--traffic-manager-port={tm_port}",    # 交通管理器端口
        f"--gpu-rank={VULKAN_GPU_ID[gpu_id]}" # carla需要vulkaninfo --summary中对应的device编号
    ]

    if cfg.get("no_server_launch"):
        command.append("--no-server-launch")

    stdout = open(job["log_file"], "w", encoding="utf-8")
    stderr = open(job["err_file"], "w", encoding="utf-8")
    stdout.write(" ".join(command) + "\n")
    stdout.flush()

    process = subprocess.Popen(
        command,
        env=env,
        cwd=bench_root, 
        stderr=stderr,
        stdout=stdout
    )

    job["process"] = process
    job["gpu_id"] = gpu_id
    job["ports"] = {world_port, tm_port,world_port+1,world_port+2} # carla的streaming端口自动占用RPC端口的+1或者+2，需要一起避让
    job["_stdout_handle"] = stdout
    job["_stderr_handle"] = stderr


def finalize_job(job):
    # 1. 强制清理该任务占用的端口 (杀死残留 CARLA)
    if not no_server_launch:
        ports = job.get("ports", set())
        for port in ports:
            try:
                # 使用 fuser 强杀占用端口的进程 (-k: kill, -9: SIGKILL)
                subprocess.run(
                    ["fuser", "-k", "-9", f"{port}/tcp"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass
        
    for handle_key in ("_stdout_handle", "_stderr_handle"):
        handle = job.pop(handle_key, None)
        if handle:
            handle.flush()
            handle.close()
    job.pop("process", None)
    job.pop("ports", None)
    job.pop("gpu_id", None)

def check_and_kill_dead_job(job):
    """
    主动检查一个正在运行的作业是否 "卡死"（基于HPC版本的日志特征）。
    如果是，则终止该进程，以便主循环的重试逻辑可以接管。
    """
    process = job.get("process")
    # 1. 确保进程存在且仍在运行
    if not process or process.poll() is not None:
        return

    log_file = job.get("log_file")
    if not log_file or not os.path.exists(log_file):
        print(f"Warning: there is no {log_file}")
        return

    lines = []
    try:
        # 2. 刷新日志句柄，确保缓冲区内容已写入磁盘
        # (必须在读取前执行，因为本脚本持有文件句柄)
        stdout_handle = job.get("_stdout_handle")
        if stdout_handle and not stdout_handle.closed:
            stdout_handle.flush()
        
        stderr_handle = job.get("_stderr_handle")
        if stderr_handle and not stderr_handle.closed:
            stderr_handle.flush()

        # 3. 读取日志文件内容
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    except Exception as e:
        print(f"Warning: Could not read log {log_file} for dead job check: {e}")
        return

    if not lines:
        return

    # 4. 检查HPC版本中的 "dead job" 特征
    # (增加了更通用的 "Stopping the route, the agent has crashed" 检查)
    CRASH_MARKERS_SUBSTR = (
        "Watchdog exception",
        "Engine crash handling finished; re-raising signal 11",
        "Stopping the route, the agent has crashed",
    )

    # ANSI 颜色行可能形如: "\x1b[91mStopping the route, the agent has crashed:"
    def has_crash_marker(lines):
        for line in lines:
            for m in CRASH_MARKERS_SUBSTR:
                if m in line:
                    return True
        return False


    # 5. 如果检测到卡死，终止进程
    if has_crash_marker(lines):
        print(
            f"Detected dead job {job['route_id']} (PID: {process.pid}) from log. "
            f"Terminating process..."
        )
        process.terminate()



def err_log_has_error(err_file: str) -> bool:
    error_marks = (
        "Traceback (most recent call last)",
        "Segmentation fault",
        "Signal 11",
        "FatalError",
        "LowLevelFatalError",
        "Watchdog exception",
        "Engine crash handling finished",
        "Address already in use",
        "CUDA out of memory",
        "Failed - Agent couldn't be set up",
        "Stopping the route, the agent has crashed",
    )
    if not os.path.exists(err_file):
        return False  # 没有 err 文件 => 尚未评估
    try:
        with open(err_file, "r", encoding="utf-8") as f:
            for line in f:
                for m in error_marks:
                    if m in line:
                        return True
    except Exception:
        return False  # 读取失败时保守地视为不需要重跑
    return False  # 有 err 文件但不含错误标记 => 已评估且正常


def main(args):
    # for fast test
    global no_server_launch
    no_server_launch = False
    if args.remote_carla_port is not None:
        global carla_world_ports, carla_tm_ports
        carla_world_ports = {args.remote_carla_port}
        if args.remote_tm_port is not None:
             carla_tm_ports = {args.remote_tm_port}
        else:
             carla_tm_ports = {args.remote_carla_port + 8000}
        no_server_launch = True
        print(f"Using remote CARLA at port {args.remote_carla_port}")
    
    global GPU_IDS
    GPU_IDS = 0
    if args.gpu is not None:
        if isinstance(args.gpu, int):
            GPU_IDS = [args.gpu]
        else:
            GPU_IDS = args.gpu
    else:
        GPU_IDS = [0]
    global MAX_PARALLEL_JOBS
    MAX_PARALLEL_JOBS = len(GPU_IDS)
    print(f"GPU_IDS: {GPU_IDS}")
    configs = [build_eval_config(args, no_server_launch)]

    job_queue = []
    for cfg in configs:
        route_path = cfg["route_path"]
        routes = [x for x in os.listdir(route_path) if x.endswith(".xml")]

        fill_zeros = 3 if cfg["benchmark"] == "bench2drive" else 2

        for seed in cfg["seeds"]:
            seed = str(seed)
            # 输出目录按 mode 分层；fixed 模式再细分到 lat_xxx
            base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed, cfg["latency_mode"])
            if cfg["latency_mode"] == "fixed":
                base_dir = os.path.join(base_dir, f"lat_{cfg['fix_latency']:.3f}")
            os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)

            for route in routes:
                route_id = route.split("_")[-1][:-4].zfill(fill_zeros)
                route_file = os.path.join(route_path, route)

                viz_path = os.path.join(base_dir, "viz", route_id)
                os.makedirs(viz_path, exist_ok=True)

                log_file = os.path.join(base_dir, "out", f"{route_id}_out.log")
                err_file = os.path.join(base_dir, "err", f"{route_id}_err.log")
                result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
                # 修改后的筛选：基于 res.json 的 status 字段判断
                should_skip = False
                if os.path.exists(result_file):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            res_data = ujson.load(f)
                        # 检查 ["_checkpoint"]["global_record"]["status"]
                        # 如果没有该字段，默认为 "Failed"
                        status = "Failed"
                        if "_checkpoint" in res_data and "global_record" in res_data["_checkpoint"]:
                            status = res_data["_checkpoint"]["global_record"].get("status", "Failed")
                        if status == "Completed":
                            should_skip = True
                            print(f"[skip] route {route_id} mode {cfg['latency_mode']} status is Completed -> skip")
                        else:
                            # 状态是 Failed 或其他，需要重跑
                            print(f"[queue] route {route_id} mode {cfg['latency_mode']} status is {status} -> queue")

                    except Exception as e:
                        # JSON 解析失败或读取错误，视为需要重跑
                        print(f"[queue] route {route_id} mode {cfg['latency_mode']} json invalid ({e}) -> queue")
                        should_skip = False

                if should_skip:
                    continue

                job = {
                    "cfg": cfg,
                    "route": route_file,
                    "route_id": route_id,
                    "seed": seed,
                    "latency_mode": cfg["latency_mode"],
                    "fix_latency": cfg["fix_latency"],
                    "rule_based_cfg": cfg["rule_based_cfg"],
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
                check_and_kill_dead_job(job)
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
            if return_code != 0:
                    infos = [
                        f"Job {job['route_id']} exited with return code {return_code}.",
                        f"route: {job.get('route')}",
                        f"log_file: {job.get('log_file')}",
                        f"err_file: {job.get('err_file')}",
                    ]
                    print(infos)
        if len(running_jobs) >= MAX_PARALLEL_JOBS or not available_gpus:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_started = False
        for _ in range(len(pending_jobs)):
            job = pending_jobs.popleft()

            if job["status"] == "completed":
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
                f"with mode={job['latency_mode']} fix_latency={job['fix_latency']} "
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
    # 1. 创建一个参数解析器
    parser = argparse.ArgumentParser(description="一个接收 seed 参数的脚本。")
    
    # 2. 添加您想要的参数
    default_eval_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        "simlingo_adaption_eval.yaml",
    )
    parser.add_argument("--eval-config", type=str, default=default_eval_config, help="评估配置 YAML 路径")
    parser.add_argument("--seed", type=int, help="用于脚本的随机种子；不传则使用 eval-config 中的 seeds", default=None)
    parser.add_argument("--remote-carla-port", type=int, default=None, help="External CARLA world port")
    parser.add_argument("--remote-tm-port", type=int, default=None, help="External CARLA TM port")
    parser.add_argument("--gpu", type=int, nargs='+', default=None, help="gpu to use")

    # 3. 解析命令行传入的参数
    args = parser.parse_args()

    # 4. 将解析到的参数 (args) 传递给 main 函数
    main(args)
