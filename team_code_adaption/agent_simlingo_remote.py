"""
Bench2Drive agent that keeps the original SimLingo preprocessing/control path,
but delegates model inference to the split encoder/scheduler/llm docker agents.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import time
from collections import deque
from pathlib import Path

import carla
import numpy as np
import torch
import ujson
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from leaderboard.autoagents import autonomous_agent
from transformers import AutoProcessor

import team_code_adaption.transfuser_utils as t_u
from scenario_logger import ScenarioLogger
from simlingo_agents.bench2drive_mcp_server.remote_inference import (
    RemoteInferenceError,
    SplitAgentPipelineClient,
)
from team_code_adaption.behavior_window_logger import BehaviorWindowLogger
from team_code_adaption.agent_simlingo import (
    DEBUG,
    LingoAgent,
    USE_UKF,
    bicycle_model_forward,
    measurement_function_hx,
    measurement_mean,
    residual_measurement_h,
    residual_state_x,
    state_mean,
)
from team_code_adaption.config_simlingo import GlobalConfig
from team_code_adaption.nav_planner import LateralPIDController
from team_code_adaption.scene_visualizer import save_scene_visualization


def get_entry_point():
    return "RemoteSplitLingoAgent"


class RemoteSplitLingoAgent(LingoAgent):
    """Drop-in leaderboard agent that uses the split dockerized inference stack."""

    def setup(self, path_to_conf_file, route_index=None):
        torch.cuda.empty_cache()
        self.track = autonomous_agent.Track.SENSORS
        if "+" in path_to_conf_file:
            self.config_path, self.save_path_root = path_to_conf_file.split("+", 1)
        else:
            self.config_path = path_to_conf_file
            self.save_path_root = route_index

        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DrivingInput = {}
        self.config = GlobalConfig()
        self.eval_budget_mode = os.getenv("SIMLINGO_EVAL_BUDGET_MODE", "fixed").strip().lower()
        if self.eval_budget_mode not in {"random", "fixed", "rule_based", "smart_assigner"}:
            raise ValueError(
                f"SIMLINGO_EVAL_BUDGET_MODE must be random/fixed/rule_based/smart_assigner, got {self.eval_budget_mode}"
            )

        self.fixed_eval_budget = self._load_fixed_eval_budget()
        self.rule_based_cfg = self._load_rule_based_budget_cfg() if self.eval_budget_mode == "rule_based" else None
        self.eval_budget = {
            "mode": self.eval_budget_mode,
            "fixed_budget": float(self.fixed_eval_budget),
            "rule_based_cfg": self.rule_based_cfg if self.eval_budget_mode == "rule_based" else None,
            "last_decision": {},
            "last_update": {},
        }

        self.last_command = -1
        self.last_command_tmp = -1
        self.user_command = None
        self.user_flag = None
        self.running = True
        self.custom_prompt = None
        self.LMDRIVE_AUGM = False
        if self.LMDRIVE_AUGM:
            command_templates_file = "data/augmented_templates/lmdrive.json"
            with open(command_templates_file, "r", encoding="utf-8") as f:
                self.command_templates = ujson.load(f)

        self.route_path = os.environ.get("ROUTES", "")
        self.route_key = self._resolve_route_key(route_index)
        route_type, route_number = self._resolve_route_path_parts()

        self.speed_controller = t_u.PIDController(
            k_p=self.config.speed_kp,
            k_i=self.config.speed_ki,
            k_d=self.config.speed_kd,
            n=self.config.speed_n,
        )
        self.turn_controller = LateralPIDController(inference_mode=False)

        image_fps = 5
        image_history_length = 1
        self.image_buffer = deque(maxlen=image_fps * image_history_length)

        self.carla_frame_rate = 1.0 / 20.0
        self.data_save_freq = 5
        self.lidar_seq_len = 1
        self.logging_freq = 10
        self.logger_region_of_interest = 30.0
        self.dense_route_planner_min_distance = 1.0
        self.dense_route_planner_max_distance = 50.0
        self.log_route_planner_min_distance = 4.0
        self.route_planner_max_distance = 50.0
        self.route_planner_min_distance = 7.5

        self.config_load_path = Path(self.config_path).parent.parent.parent / ".hydra" / "config.yaml"
        with open(self.config_load_path, "r", encoding="utf-8") as file:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(file)
        self.cfg = cfg
        if hasattr(self.cfg, "data_module") and hasattr(self.cfg, "model"):
            self.cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        if self.config.eval_route_as == -1:
            self.config.eval_route_as = getattr(self.cfg.model, "route_as", "target_point_command")

        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
        self.tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<WAYPOINTS>",
                    "<WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS_DIFF>",
                    "<ORG_WAYPOINTS>",
                    "<WAYPOINT_LAST>",
                    "<ROUTE>",
                    "<ROUTE_DIFF>",
                    "<TARGET_POINT>",
                ]
            }
        )
        self.tokenizer.padding_side = "left"

        self.remote_pipeline = SplitAgentPipelineClient()
        self.token_prune_cfg = self._load_eval_token_prune_cfg()
        self.predict_language_cfg = self._load_predict_language_cfg()

        self.iter = self.config_path.split("epoch=")[-1].split("/")[0]
        self.session = self.config_path.split("/")[-4]
        self.T = 1
        self.stuck_detector = 0
        self.force_move = 0
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.target_point_prev = [1e5, 1e5, 1e5]

        if USE_UKF:
            self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
            self.ukf = UKF(
                dim_x=4,
                dim_z=4,
                fx=bicycle_model_forward,
                hx=measurement_function_hx,
                dt=self.carla_frame_rate,
                points=self.points,
                x_mean_fn=state_mean,
                z_mean_fn=measurement_mean,
                residual_x=residual_state_x,
                residual_z=residual_measurement_h,
            )
            self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
            self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
            self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])

            self.filter_initialized = False

        self.state_log = deque(maxlen=max((self.lidar_seq_len * self.data_save_freq), 2))
        self.save_path = os.environ.get("SAVE_PATH") + self.save_path_root
        if self.save_path is not None and route_index is not None:
            self.save_path = pathlib.Path(self.save_path) / route_index
            pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
            self.lon_logger = ScenarioLogger(
                save_path=self.save_path,
                route_index=route_index,
                logging_freq=self.logging_freq,
                log_only=True,
                route_only=False,
                roi=self.logger_region_of_interest,
            )

        save_path_str = str(self.save_path)
        self.debug_save_path = (
            save_path_str
            + "/debug_viz"
            + f"/{self.session}/iter_{self.iter}/{route_type}/{route_number}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        Path(self.debug_save_path).mkdir(parents=True, exist_ok=True)
        self.save_path_metric = self.debug_save_path + "/metric"
        Path(self.save_path_metric).mkdir(parents=True, exist_ok=True)
        self._behavior_logger = self._create_behavior_logger()
        self.save_scene_frames = self._env_flag("SIMLINGO_SAVE_SCENE_FRAMES", False)
        self.save_scene_raw_frames = self._env_flag("SIMLINGO_SAVE_SCENE_RAW_FRAMES", False)
        self.scene_frame_stride = self._load_scene_frame_stride()
        self.scene_frame_format = self._load_scene_frame_format()
        self.scene_frames_dir = Path(self.debug_save_path) / "scene_frames"
        self._scene_frame_warning_emitted = False
        if self.save_scene_frames:
            (self.scene_frames_dir / "annotated").mkdir(parents=True, exist_ok=True)
            if self.save_scene_raw_frames:
                (self.scene_frames_dir / "raw").mkdir(parents=True, exist_ok=True)
        self._route_inference_first_start_perf = None
        self._route_inference_first_start_wall = None
        self._route_inference_last_end_perf = None
        self._route_inference_last_end_wall = None
        self._route_inference_accumulated_sec = 0.0
        self._route_inference_frame_count = 0
        self._route_inference_success_count = 0
        self._route_inference_failed_count = 0
        self._route_inference_timing = {
            "route_key": self.route_key,
            "route_path": self.route_path,
            "save_path_metric": self.save_path_metric,
            "started": False,
            "finalized": False,
            "frame_count": 0,
            "successful_frame_count": 0,
            "failed_frame_count": 0,
            "accumulated_remote_inference_sec": 0.0,
        }
        self.record_tflops = self._env_flag("SIMLINGO_EVAL_RECORD_TFLOPS", True)
        self._split_profile_frames = []
        self._split_profile_errors = []
        if DEBUG:
            self.save_path_img = self.debug_save_path + "/images"
            Path(self.save_path_img).mkdir(parents=True, exist_ok=True)

    def _resolve_route_key(self, route_index=None) -> str:
        if route_index is not None:
            return str(route_index)

        if self.save_path_root:
            return str(self.save_path_root)

        eval_route_id = os.getenv("SIMLINGO_EVAL_ROUTE_ID", "").strip()
        if eval_route_id:
            return f"eval_route/{eval_route_id}"

        if str(getattr(self, "route_path", "") or "").strip():
            route_type, route_number = self._resolve_route_path_parts()
            if route_type and route_number:
                return f"{route_type}/{route_number}"
            if route_number:
                return route_number

        raise RuntimeError(
            "Unable to resolve a non-empty route_key. Ensure leaderboard passes a save_name, "
            "route_index, SIMLINGO_EVAL_ROUTE_ID, or ROUTES."
        )

    def _resolve_route_path_parts(self) -> tuple[str, str]:
        route_path = str(getattr(self, "route_path", "") or "").strip()
        if not route_path:
            return "unknown_route_type", "unknown_route"
        route = pathlib.Path(route_path)
        route_number = route.stem or "unknown_route"
        normalized = route_path.replace("\\", "/")
        if "data/benchmarks/" in normalized:
            route_type = normalized.split("data/benchmarks/")[-1].split("/")[0]
        else:
            route_type = route.parent.name or "routes"
        return route_type, route_number

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            self.tick(input_data)
            return control

        tick_data = self.tick(input_data)
        remote_payload = self._build_remote_payload(timestamp=timestamp, tick_data=tick_data)
        inference_started_perf = time.perf_counter()
        inference_started_wall = time.time()
        self._begin_route_inference_frame(
            timestamp=timestamp,
            remote_payload=remote_payload,
            started_perf=inference_started_perf,
            started_wall=inference_started_wall,
        )
        try:
            remote_result = self.remote_pipeline.infer(remote_payload)
        except RemoteInferenceError as exc:
            self._finish_route_inference_frame(
                timestamp=timestamp,
                remote_payload=remote_payload,
                started_perf=inference_started_perf,
                started_wall=inference_started_wall,
                status="failed",
                error=str(exc),
            )
            raise RuntimeError(f"Remote split-agent inference failed at step {self.step}: {exc}") from exc
        self._finish_route_inference_frame(
            timestamp=timestamp,
            remote_payload=remote_payload,
            started_perf=inference_started_perf,
            started_wall=inference_started_wall,
            status="success",
            remote_result=remote_result,
        )
        self._accumulate_split_profile(remote_result)

        llm_payload = remote_result.get("llm_payload", {})
        self._log_remote_frame_alignment(
            timestamp=timestamp,
            remote_payload=remote_payload,
            remote_result=remote_result,
            llm_payload=llm_payload,
        )
        self._log_scheduler_plan(
            timestamp=timestamp,
            remote_payload=remote_payload,
            remote_result=remote_result,
            llm_payload=llm_payload,
        )
        pred_speed_wps_np = llm_payload.get("speed_wps")
        pred_route_np = llm_payload.get("route")
        if pred_speed_wps_np is None or pred_route_np is None:
            raise RuntimeError(f"Remote pipeline returned incomplete llm payload: {llm_payload.keys()}")

        pred_speed_wps = torch.as_tensor(pred_speed_wps_np, device=self.device, dtype=torch.float32)
        pred_route = torch.as_tensor(pred_route_np, device=self.device, dtype=torch.float32)

        self.eval_budget = {
            "mode": self.eval_budget_mode,
            "fixed_budget": float(self.fixed_eval_budget),
            "budget_value": self._to_jsonable(llm_payload.get("budget_value")),
            "pipeline_meta": remote_result.get("pipeline_meta", {}),
            "last_decision": {},
            "last_update": {
                "step": self.step,
                "timestamp": float(timestamp),
            },
        }

        gt_velocity = tick_data["speed"]
        steer, throttle, brake = self.control_pid(pred_route, gt_velocity, pred_speed_wps)

        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        self._log_remote_control_trace(
            timestamp=timestamp,
            remote_payload=remote_payload,
            remote_result=remote_result,
            llm_payload=llm_payload,
            tick_data=tick_data,
            pred_route=pred_route,
            pred_speed_wps=pred_speed_wps,
            pid_steer=steer,
            pid_throttle=throttle,
            pid_brake=brake,
            returned_control=control,
            stored_control=self.control,
        )
        self._save_scene_frame(
            timestamp=timestamp,
            tick_data=tick_data,
            remote_result=remote_result,
            llm_payload=llm_payload,
            pred_route=pred_route,
            pred_speed_wps=pred_speed_wps,
            control=control,
        )

        metric_info = self.get_metric_info()
        metric_info["eval_budget"] = copy.deepcopy(self.eval_budget)
        self.metric_info[self.step] = metric_info
        if self.save_path_metric is not None:
            with open(f"{self.save_path_metric}/metric_info.json", "w", encoding="utf-8") as outfile:
                json.dump(self.metric_info, outfile, indent=4)

        return control

    def destroy(self, results=None):  # pylint: disable=unused-argument
        self._write_split_tflops_summary()
        self._finalize_route_inference_timing(results=results)
        if getattr(self, "_behavior_logger", None) is not None:
            try:
                self._behavior_logger.finalize()
            except Exception as exc:
                print(f"[behavior-log] failed to finalize route sample: {exc}", flush=True)
        try:
            cleanup_result = self.remote_pipeline.cleanup_route(self.route_key)
            print(
                "[remote-pipeline] route NATS cleanup completed: "
                f"route_key={self.route_key} "
                f"deleted_consumers={sum(cleanup_result['deleted_consumers'].values())} "
                f"purged_subjects={sum(cleanup_result['purged_subjects'].values())}"
            )
        except Exception as exc:  # Cleanup must not hide the route evaluation result.
            print(f"[remote-pipeline] route NATS cleanup failed for {self.route_key}: {exc}")
        del self.config

    def _accumulate_split_profile(self, remote_result: dict) -> None:
        if not self.record_tflops:
            return
        profile_trace = remote_result.get("profile_trace", [])
        communication_trace = remote_result.get("nats_communication_trace", [])
        if not isinstance(profile_trace, list):
            profile_trace = []
        if not isinstance(communication_trace, list):
            communication_trace = []

        inter_agent_records = []
        agent_names = {"encoder", "scheduler", "llm"}
        for item in communication_trace:
            if not isinstance(item, dict):
                continue
            if item.get("sender") in agent_names and item.get("receiver") in agent_names:
                inter_agent_records.append(item)

        valid_profiles = [item for item in profile_trace if isinstance(item, dict)]
        frame_profile_latency = [
            float(item["latency_ms"])
            for item in valid_profiles
            if item.get("profile_success") and item.get("latency_ms") is not None
        ]
        communication_latency = [
            float(item["latency_ms"])
            for item in inter_agent_records
            if item.get("latency_ms") is not None
        ]
        expected_links = {
            "encoder_to_scheduler_budget",
            "scheduler_budget_to_llm_prefix",
            "llm_prefix_to_scheduler_plan",
            "scheduler_plan_to_llm_final",
            "llm_final_to_scheduler_decision_update",
        }
        observed_links = {str(item.get("link")) for item in inter_agent_records}
        communication_complete = expected_links.issubset(observed_links)
        pipeline_meta = remote_result.get("pipeline_meta", {})
        self._split_profile_frames.append(
            {
                "step": int(self.step),
                "pipeline_request_id": pipeline_meta.get("request_id") if isinstance(pipeline_meta, dict) else None,
                "profile_trace": valid_profiles,
                "nats_communication_trace": inter_agent_records,
                "gpu_compute_latency_ms": float(sum(frame_profile_latency)),
                "agent_nats_communication_ms": (
                    float(sum(communication_latency)) if communication_complete else None
                ),
                "nats_communication_complete": communication_complete,
                "end_to_end_latency_ms": (
                    float(pipeline_meta.get("total_duration_ms"))
                    if isinstance(pipeline_meta, dict) and pipeline_meta.get("total_duration_ms") is not None
                    else None
                ),
            }
        )

    @staticmethod
    def _latency_summary(values):
        values = [float(value) for value in values if value is not None]
        return {
            "count": len(values),
            "avg_ms": None if not values else float(sum(values) / len(values)),
            "p50_ms": None if not values else float(np.percentile(values, 50)),
            "p90_ms": None if not values else float(np.percentile(values, 90)),
            "max_ms": None if not values else float(max(values)),
        }

    def _resolve_split_tflops_output_path(self):
        result_file = os.getenv("SIMLINGO_EVAL_RESULT_FILE", "").strip()
        if result_file:
            result_path = Path(result_file)
            if result_path.name.endswith("_res.json"):
                return result_path.with_name(result_path.name.replace("_res.json", "_tflops.json"))
            return result_path.with_suffix(".tflops.json")
        if getattr(self, "save_path", None) is not None:
            return Path(self.save_path) / "route_tflops.json"
        return None

    def _write_split_tflops_summary(self) -> None:
        output_path = self._resolve_split_tflops_output_path()
        if output_path is None:
            return

        frames = list(getattr(self, "_split_profile_frames", []))
        profile_records = [record for frame in frames for record in frame["profile_trace"]]
        communication_records = [
            record for frame in frames for record in frame["nats_communication_trace"]
        ]
        total_flops = sum(
            int(record.get("flops", 0) or 0)
            for record in profile_records
            if record.get("profile_success")
        )
        gpu_values = [frame["gpu_compute_latency_ms"] for frame in frames]
        communication_values = [frame["agent_nats_communication_ms"] for frame in frames]
        end_to_end_values = [frame["end_to_end_latency_ms"] for frame in frames]

        expected_phases = {
            ("encoder", "encode"),
            ("scheduler", "budget"),
            ("llm", "prefix"),
            ("scheduler", "plan"),
            ("llm", "final"),
            ("scheduler", "decision_update"),
        }
        fully_profiled_steps = 0
        for frame in frames:
            successful = {
                (record.get("agent"), record.get("phase"))
                for record in frame["profile_trace"]
                if record.get("profile_success")
            }
            if expected_phases.issubset(successful):
                fully_profiled_steps += 1

        agents = {}
        for agent_name in ("encoder", "scheduler", "llm"):
            agent_records = [r for r in profile_records if r.get("agent") == agent_name]
            phase_names = sorted({str(r.get("phase")) for r in agent_records})
            phases = {}
            for phase_name in phase_names:
                phase_records = [r for r in agent_records if str(r.get("phase")) == phase_name]
                phase_latencies = [r.get("latency_ms") for r in phase_records if r.get("profile_success")]
                phases[phase_name] = {
                    "calls": len(phase_records),
                    "total_flops": sum(int(r.get("flops", 0) or 0) for r in phase_records),
                    "latency": self._latency_summary(phase_latencies),
                }
            agents[agent_name] = {
                "calls": len(agent_records),
                "total_flops": sum(int(r.get("flops", 0) or 0) for r in agent_records),
                "phases": phases,
            }

        link_summaries = {}
        for link in sorted({str(r.get("link")) for r in communication_records}):
            link_values = [r.get("latency_ms") for r in communication_records if str(r.get("link")) == link]
            link_summaries[link] = self._latency_summary(link_values)

        step_count = len(frames)
        avg_flops = None if step_count == 0 else total_flops / step_count
        gpu_summary = self._latency_summary(gpu_values)
        communication_summary = self._latency_summary(communication_values)
        end_to_end_summary = self._latency_summary(end_to_end_values)
        errors = [
            {
                "agent": r.get("agent"),
                "phase": r.get("phase"),
                "step": r.get("frame_id"),
                "error": r.get("error"),
            }
            for r in profile_records
            if not r.get("profile_success")
        ]
        summary = {
            "route_id": os.getenv("SIMLINGO_EVAL_ROUTE_ID"),
            "route_key": getattr(self, "route_key", None),
            "record_tflops": bool(getattr(self, "record_tflops", False)),
            "predict_language": self._to_jsonable(getattr(self, "predict_language_cfg", {})),
            "compute_steps": step_count,
            "inference_profiled_steps": fully_profiled_steps,
            "total_inference_flops": int(total_flops),
            "total_inference_tflops": total_flops / 1e12,
            "avg_inference_flops_per_step": avg_flops,
            "avg_inference_tflops_per_step": None if avg_flops is None else avg_flops / 1e12,
            "forward_latency_ms_values": gpu_values,
            "forward_latency_ms_count": gpu_summary["count"],
            "avg_forward_latency_ms": gpu_summary["avg_ms"],
            "max_forward_latency_ms": gpu_summary["max_ms"],
            "p50_forward_latency_ms": gpu_summary["p50_ms"],
            "p90_forward_latency_ms": gpu_summary["p90_ms"],
            "agent_nats_communication_ms_values": communication_values,
            "agent_nats_communication_ms_count": communication_summary["count"],
            "avg_agent_nats_communication_ms_per_step": communication_summary["avg_ms"],
            "p50_agent_nats_communication_ms_per_step": communication_summary["p50_ms"],
            "p90_agent_nats_communication_ms_per_step": communication_summary["p90_ms"],
            "max_agent_nats_communication_ms_per_step": communication_summary["max_ms"],
            "nats_communication_ms_values": communication_values,
            "nats_communication_profiled_steps": communication_summary["count"],
            "avg_nats_communication_ms_per_step": communication_summary["avg_ms"],
            "total_nats_communication_ms": float(
                sum(value for value in communication_values if value is not None)
            ),
            "end_to_end_latency_ms_values": end_to_end_values,
            "end_to_end_latency": end_to_end_summary,
            "agents": agents,
            "nats_links": link_summaries,
            "profile_errors": errors,
            "profiler": "per-container torch.profiler.profile(with_flops=True)",
            "latency_source": "sum of per-container CUDA events",
            "nats_timing_source": "sender time.time_ns to receiver time.time_ns after decode",
            "nats_timing_note": "Agent-to-agent one-way timings require synchronized host clocks; Bench2Drive boundary links are excluded.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=4, ensure_ascii=False)

    def _load_eval_token_prune_cfg(self):
        value = os.getenv("SIMLINGO_EVAL_TOKEN_PRUNE_RATIO", "").strip()
        if not value:
            return None
        try:
            prune_ratio = float(value)
        except Exception as exc:
            raise ValueError(f"SIMLINGO_EVAL_TOKEN_PRUNE_RATIO must be a float in [0, 1], got {value}") from exc
        if not (0.0 <= prune_ratio <= 1.0):
            raise ValueError(f"SIMLINGO_EVAL_TOKEN_PRUNE_RATIO must be in [0, 1], got {value}")
        return {
            "mode": "prune2drive",
            "prune_ratio": prune_ratio,
            "min_keep": 1,
        }

    def _load_predict_language_cfg(self):
        enabled = self._env_flag("SIMLINGO_PREDICT_LANGUAGE", False)
        max_new_tokens = int(os.getenv("SIMLINGO_PREDICT_LANGUAGE_MAX_NEW_TOKENS", "100"))
        stride = int(os.getenv("SIMLINGO_PREDICT_LANGUAGE_STRIDE", "1"))
        if max_new_tokens < 1:
            raise ValueError(f"SIMLINGO_PREDICT_LANGUAGE_MAX_NEW_TOKENS must be >= 1, got {max_new_tokens}")
        if stride < 1:
            raise ValueError(f"SIMLINGO_PREDICT_LANGUAGE_STRIDE must be >= 1, got {stride}")
        return {
            "enabled": enabled,
            "max_new_tokens": max_new_tokens,
            "stride": stride,
        }

    def _create_behavior_logger(self):
        save_path = os.getenv("SAVE_PATH", "").strip()
        route_id = os.getenv("SIMLINGO_EVAL_ROUTE_ID", "").strip()
        if not save_path or not route_id:
            return None
        try:
            # SAVE_PATH is <base_dir>/viz/<route_id>; behaviors is a sibling of viz.
            base_dir = Path(save_path).parent.parent
            return BehaviorWindowLogger(
                base_dir / "behaviors" / route_id,
                route_id=route_id,
                route_key=self.route_key,
                prune_ratio=(
                    None if self.token_prune_cfg is None else float(self.token_prune_cfg["prune_ratio"])
                ),
                mode=self.eval_budget_mode,
                fixed_budget=self.fixed_eval_budget,
                predict_language_enabled=self.predict_language_cfg["enabled"],
                predict_language_max_new_tokens=self.predict_language_cfg["max_new_tokens"],
                predict_language_stride=self.predict_language_cfg["stride"],
            )
        except Exception as exc:
            print(f"[behavior-log] failed to initialize route sample logger: {exc}", flush=True)
            return None

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _load_scene_frame_stride() -> int:
        value = int(os.getenv("SIMLINGO_SCENE_FRAME_STRIDE", "1"))
        if value < 1:
            raise ValueError(f"SIMLINGO_SCENE_FRAME_STRIDE must be >= 1, got {value}")
        return value

    @staticmethod
    def _load_scene_frame_format() -> str:
        value = os.getenv("SIMLINGO_SCENE_FRAME_FORMAT", "png").strip().lower().lstrip(".")
        if value not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"SIMLINGO_SCENE_FRAME_FORMAT must be png/jpg/jpeg, got {value}")
        return value

    def _save_scene_frame(
        self,
        *,
        timestamp: float,
        tick_data: dict,
        remote_result: dict,
        llm_payload: dict,
        pred_route: torch.Tensor,
        pred_speed_wps: torch.Tensor,
        control: carla.VehicleControl,
    ) -> None:
        if not self.save_scene_frames or self.step % self.scene_frame_stride != 0:
            return
        camera = getattr(self, "camera_for_viz", None)
        if camera is None:
            return

        filename = f"{self.step:06d}.{self.scene_frame_format}"
        annotated_path = self.scene_frames_dir / "annotated" / filename
        raw_path = self.scene_frames_dir / "raw" / filename if self.save_scene_raw_frames else None
        pipeline_meta = remote_result.get("pipeline_meta", {})
        metadata = {
            "route_key": self.route_key,
            "frame_id": int(self.step),
            "timestamp": round(float(timestamp), 3),
            "speed_mps": round(self._scalar_or_none(tick_data.get("speed")), 3),
            "budget_mode": self.eval_budget_mode,
            "budget_value": self._to_jsonable(llm_payload.get("budget_value")),
            "visual_token_keep_ratio": self._to_jsonable(llm_payload.get("visual_token_keep_ratio")),
            "steer": round(float(control.steer), 4),
            "throttle": round(float(control.throttle), 4),
            "brake": round(float(control.brake), 4),
            "stage_durations_ms": pipeline_meta.get("stage_durations_ms", {}),
            "prompt": getattr(self, "prompt", ""),
            "language": self._to_jsonable(llm_payload.get("language")),
        }
        try:
            save_scene_visualization(
                camera_bgr=camera,
                annotated_path=annotated_path,
                raw_path=raw_path,
                target_points=getattr(self, "target_points", None),
                pred_route=pred_route.detach().float().cpu().numpy(),
                pred_speed_wps=pred_speed_wps.detach().float().cpu().numpy(),
                metadata=metadata,
            )
        except Exception as exc:  # Visualization must never abort an evaluation route.
            if not self._scene_frame_warning_emitted:
                print(f"[scene-viz] failed to save frame {self.step}: {exc}", flush=True)
                self._scene_frame_warning_emitted = True

    def _build_remote_payload(self, *, timestamp: float, tick_data: dict) -> dict:
        prompt_label = self.DrivingInput.get("prompt_inference") or self.DrivingInput.get("prompt")
        if prompt_label is None:
            raise RuntimeError("Prompt input is missing from DrivingInput")
        camera_images = self.DrivingInput.get("camera_images")
        if camera_images is None:
            raise RuntimeError("Camera input is missing from DrivingInput")
        if camera_images.ndim != 6:
            raise RuntimeError(f"camera_images must have shape [B,T,NP,C,H,W], got {tuple(camera_images.shape)}")
        num_patches = int(camera_images.shape[2])

        ego_xy = np.asarray(tick_data.get("gps"), dtype=np.float32).tolist() if tick_data.get("gps") is not None else None
        ego_yaw = float(tick_data.get("compass")) if tick_data.get("compass") is not None else None
        runtime_context = {
            "timestamp": float(timestamp),
            "step": int(self.step),
            "route_key": self.route_key,
            "prompt": getattr(self, "prompt", ""),
            "prompt_tp": getattr(self, "prompt_tp", ""),
            "ego_xy": ego_xy,
            "ego_yaw": ego_yaw,
            "gps": ego_xy,
            "compass": ego_yaw,
            "speed_mps": float(tick_data["speed"][0].item()) if hasattr(tick_data.get("speed"), "__getitem__") else None,
        }

        predict_language_cfg = {
            "enabled": (
                self.predict_language_cfg["enabled"]
                and self.step % self.predict_language_cfg["stride"] == 0
            ),
            "max_new_tokens": self.predict_language_cfg["max_new_tokens"],
        }

        payload = {
            "route_key": self.route_key,
            "route_keys": [self.route_key],
            "frame_id": int(self.step),
            "prompt_texts": list(prompt_label.language_string),
            "placeholder_values": prompt_label.placeholder_values,
            "camera_images": camera_images.detach().float().cpu().numpy(),
            "num_patches": num_patches,
            "expand_image_token": False,
            "runtime_context": runtime_context,
            "predict_language": predict_language_cfg,
        }
        if self.token_prune_cfg is not None:
            # 关键调用点：split encoder 按帧接收 prune 配置，避免不同实验需要重启 agent。
            payload["token_prune"] = self.token_prune_cfg
        return payload

    def _begin_route_inference_frame(
        self,
        *,
        timestamp: float,
        remote_payload: dict,
        started_perf: float,
        started_wall: float,
    ) -> None:
        if self._route_inference_first_start_perf is None:
            self._route_inference_first_start_perf = float(started_perf)
            self._route_inference_first_start_wall = float(started_wall)
            self._route_inference_timing.update(
                {
                    "started": True,
                    "finalized": False,
                    "first_frame_id": self._to_int_or_none(remote_payload.get("frame_id")),
                    "first_timestamp": float(timestamp),
                    "first_frame_started_at_unix": float(started_wall),
                    "first_frame_started_at": self._format_unix_ts(started_wall),
                }
            )

        self._route_inference_timing.update(
            {
                "current_frame_id": self._to_int_or_none(remote_payload.get("frame_id")),
                "current_timestamp": float(timestamp),
                "current_frame_started_at_unix": float(started_wall),
                "current_frame_started_at": self._format_unix_ts(started_wall),
                "last_status": "running",
            }
        )
        self._write_route_inference_timing()

    def _finish_route_inference_frame(
        self,
        *,
        timestamp: float,
        remote_payload: dict,
        started_perf: float,
        started_wall: float,
        status: str,
        remote_result: dict | None = None,
        error: str | None = None,
    ) -> None:
        ended_perf = time.perf_counter()
        ended_wall = time.time()
        frame_duration_sec = max(0.0, float(ended_perf - started_perf))
        self._route_inference_last_end_perf = float(ended_perf)
        self._route_inference_last_end_wall = float(ended_wall)
        self._route_inference_accumulated_sec += frame_duration_sec
        self._route_inference_frame_count += 1
        if status == "success":
            self._route_inference_success_count += 1
        else:
            self._route_inference_failed_count += 1

        pipeline_meta = remote_result.get("pipeline_meta", {}) if isinstance(remote_result, dict) else {}
        stage_durations_ms = pipeline_meta.get("stage_durations_ms") if isinstance(pipeline_meta, dict) else None
        pipeline_total_ms = pipeline_meta.get("total_duration_ms") if isinstance(pipeline_meta, dict) else None
        route_elapsed_sec = None
        if self._route_inference_first_start_perf is not None:
            route_elapsed_sec = max(0.0, float(ended_perf - self._route_inference_first_start_perf))

        timing_update = {
            "finalized": False,
            "frame_count": int(self._route_inference_frame_count),
            "successful_frame_count": int(self._route_inference_success_count),
            "failed_frame_count": int(self._route_inference_failed_count),
            "last_status": status,
            "last_frame_id": self._to_int_or_none(remote_payload.get("frame_id")),
            "last_timestamp": float(timestamp),
            "last_frame_started_at_unix": float(started_wall),
            "last_frame_started_at": self._format_unix_ts(started_wall),
            "last_frame_completed_at_unix": float(ended_wall),
            "last_frame_completed_at": self._format_unix_ts(ended_wall),
            "last_frame_remote_inference_sec": frame_duration_sec,
            "accumulated_remote_inference_sec": float(self._route_inference_accumulated_sec),
            "avg_remote_inference_sec": (
                float(self._route_inference_accumulated_sec / self._route_inference_frame_count)
                if self._route_inference_frame_count
                else None
            ),
            "route_elapsed_from_first_frame_start_sec": route_elapsed_sec,
            "last_pipeline_total_duration_ms": pipeline_total_ms,
            "last_pipeline_stage_durations_ms": self._to_jsonable(stage_durations_ms),
        }
        if error:
            timing_update["last_error"] = error
        elif "last_error" in self._route_inference_timing:
            timing_update["last_error"] = None

        self._route_inference_timing.update(timing_update)
        self._write_route_inference_timing()

    def _finalize_route_inference_timing(self, *, results=None) -> None:
        if not hasattr(self, "_route_inference_timing"):
            return

        finalized_wall = time.time()
        finalized_perf = time.perf_counter()
        route_elapsed_sec = self._route_inference_timing.get("route_elapsed_from_first_frame_start_sec")
        if self._route_inference_first_start_perf is not None:
            end_perf = self._route_inference_last_end_perf or finalized_perf
            route_elapsed_sec = max(0.0, float(end_perf - self._route_inference_first_start_perf))

        self._route_inference_timing.update(
            {
                "finalized": True,
                "finalized_at_unix": float(finalized_wall),
                "finalized_at": self._format_unix_ts(finalized_wall),
                "route_elapsed_from_first_frame_start_sec": route_elapsed_sec,
            }
        )
        if results is not None:
            self._route_inference_timing["leaderboard_results_repr"] = repr(results)
        self._write_route_inference_timing()

    def _write_route_inference_timing(self) -> None:
        save_path_metric = getattr(self, "save_path_metric", None)
        if save_path_metric is None:
            return
        try:
            log_path = Path(save_path_metric) / "route_inference_timing.json"
            with open(log_path, "w", encoding="utf-8") as outfile:
                json.dump(self._route_inference_timing, outfile, indent=2, ensure_ascii=True)
        except Exception:
            return

    def _format_unix_ts(self, ts: float) -> str:
        whole = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(ts)))
        millis = int((float(ts) % 1.0) * 1000)
        return f"{whole}.{millis:03d}"

    def _log_remote_frame_alignment(
        self,
        *,
        timestamp: float,
        remote_payload: dict,
        remote_result: dict,
        llm_payload: dict,
    ) -> None:
        if self.save_path_metric is None:
            return

        pipeline_meta = remote_result.get("pipeline_meta", {})
        runtime_context = llm_payload.get("runtime_context", {})
        expected_step = int(self.step)
        payload_frame_id = self._to_int_or_none(remote_payload.get("frame_id"))
        result_frame_id = self._to_int_or_none(remote_result.get("frame_id"))
        llm_frame_id = self._to_int_or_none(llm_payload.get("frame_id"))
        runtime_step = self._to_int_or_none(runtime_context.get("step")) if isinstance(runtime_context, dict) else None
        meta_request_id = pipeline_meta.get("request_id") if isinstance(pipeline_meta, dict) else None
        runtime_request_id = (
            runtime_context.get("pipeline_request_id")
            if isinstance(runtime_context, dict)
            else None
        )

        frame_match = (
            payload_frame_id == expected_step
            and result_frame_id == expected_step
            and llm_frame_id == expected_step
            and runtime_step == expected_step
        )
        request_match = bool(meta_request_id) and meta_request_id == runtime_request_id
        record = {
            "step": expected_step,
            "timestamp": float(timestamp),
            "payload_frame_id": payload_frame_id,
            "result_frame_id": result_frame_id,
            "llm_payload_frame_id": llm_frame_id,
            "runtime_context_step": runtime_step,
            "pipeline_meta_request_id": meta_request_id,
            "runtime_context_pipeline_request_id": runtime_request_id,
            "frame_match": frame_match,
            "request_match": request_match,
            "alignment_ok": frame_match and request_match,
        }

        try:
            log_path = Path(self.save_path_metric) / "remote_frame_alignment.jsonl"
            with open(log_path, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            return

    def _log_scheduler_plan(
        self,
        *,
        timestamp: float,
        remote_payload: dict,
        remote_result: dict,
        llm_payload: dict,
    ) -> None:
        if self.save_path_metric is None:
            return

        pipeline_meta = remote_result.get("pipeline_meta", {})
        runtime_context = llm_payload.get("runtime_context", {})
        execution_plan = llm_payload.get("execution_plan_applied")
        plan_np = None if execution_plan is None else np.asarray(execution_plan)

        record = {
            "step": int(self.step),
            "timestamp": float(timestamp),
            "payload_frame_id": self._to_int_or_none(remote_payload.get("frame_id")),
            "result_frame_id": self._to_int_or_none(remote_result.get("frame_id")),
            "llm_payload_frame_id": self._to_int_or_none(llm_payload.get("frame_id")),
            "pipeline_meta_request_id": pipeline_meta.get("request_id") if isinstance(pipeline_meta, dict) else None,
            "runtime_context_pipeline_request_id": (
                runtime_context.get("pipeline_request_id") if isinstance(runtime_context, dict) else None
            ),
            "budget_value": self._to_jsonable(llm_payload.get("budget_value")),
            "language": self._to_jsonable(llm_payload.get("language")),
            "predict_language": self._to_jsonable(llm_payload.get("predict_language")),
            "scheduler_plan_meta": self._to_jsonable(llm_payload.get("scheduler_plan_meta")),
            "visual_token_prune": self._to_jsonable(llm_payload.get("visual_token_prune")),
            "visual_token_keep_ratio": self._to_jsonable(llm_payload.get("visual_token_keep_ratio")),
            "visual_token_prune_summary": self._visual_token_prune_summary(llm_payload),
            "plan_present": plan_np is not None,
            "plan_shape": None if plan_np is None else list(plan_np.shape),
        }

        if plan_np is not None:
            batch_plan = plan_np[0] if plan_np.ndim == 4 and plan_np.shape[0] > 0 else plan_np
            record["execution_plan"] = self._to_jsonable(plan_np)
            if batch_plan.ndim >= 2:
                layer_axes = tuple(range(1, batch_plan.ndim))
                active = (batch_plan > 0.5).any(axis=layer_axes).astype(np.int64)
                record["layer_active_mask"] = active.tolist()
                record["active_layer_count"] = int(active.sum())
                record["layer_keep_ratio"] = (batch_plan > 0.5).mean(axis=layer_axes).astype(np.float32).tolist()

        behavior_logger = getattr(self, "_behavior_logger", None)
        if behavior_logger is not None:
            try:
                behavior_logger.add_step(
                    step=self.step,
                    timestamp=timestamp,
                    remote_inference_sec=self._route_inference_timing["last_frame_remote_inference_sec"],
                    layer_active_mask=record.get("layer_active_mask"),
                )
            except Exception as exc:
                print(f"[behavior-log] failed to record step {self.step}: {exc}", flush=True)

        try:
            log_path = Path(self.save_path_metric) / "remote_scheduler_plan.jsonl"
            with open(log_path, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            return

    def _log_remote_control_trace(
        self,
        *,
        timestamp: float,
        remote_payload: dict,
        remote_result: dict,
        llm_payload: dict,
        tick_data: dict,
        pred_route: torch.Tensor,
        pred_speed_wps: torch.Tensor,
        pid_steer,
        pid_throttle,
        pid_brake,
        returned_control: carla.VehicleControl,
        stored_control: carla.VehicleControl,
    ) -> None:
        if self.save_path_metric is None:
            return

        pipeline_meta = remote_result.get("pipeline_meta", {})
        runtime_context = llm_payload.get("runtime_context", {})
        initial_delay_active = bool(self.step < self.config.inital_frames_delay)
        returned = self._vehicle_control_to_dict(returned_control)
        stored = self._vehicle_control_to_dict(stored_control)
        record = {
            "step": int(self.step),
            "timestamp": float(timestamp),
            "payload_frame_id": self._to_int_or_none(remote_payload.get("frame_id")),
            "result_frame_id": self._to_int_or_none(remote_result.get("frame_id")),
            "llm_payload_frame_id": self._to_int_or_none(llm_payload.get("frame_id")),
            "pipeline_meta_request_id": pipeline_meta.get("request_id") if isinstance(pipeline_meta, dict) else None,
            "runtime_context_pipeline_request_id": (
                runtime_context.get("pipeline_request_id") if isinstance(runtime_context, dict) else None
            ),
            "gt_velocity": self._scalar_or_none(tick_data.get("speed")),
            "pred_route": self._array_summary(pred_route),
            "pred_speed_wps": self._array_summary(pred_speed_wps),
            "language": self._to_jsonable(llm_payload.get("language")),
            "predict_language": self._to_jsonable(llm_payload.get("predict_language")),
            "visual_token_prune": self._to_jsonable(llm_payload.get("visual_token_prune")),
            "visual_token_keep_ratio": self._to_jsonable(llm_payload.get("visual_token_keep_ratio")),
            "visual_token_prune_summary": self._visual_token_prune_summary(llm_payload),
            "pid": {
                "steer": self._scalar_or_none(pid_steer),
                "throttle": self._scalar_or_none(pid_throttle),
                "brake": self._scalar_or_none(pid_brake),
            },
            "returned_control": returned,
            "stored_control": stored,
            "initial_delay_active": initial_delay_active,
            "returned_control_matches_stored_control": returned == stored,
            "control_in_range": (
                returned is not None
                and -1.0 <= returned["steer"] <= 1.0
                and 0.0 <= returned["throttle"] <= 1.0
                and 0.0 <= returned["brake"] <= 1.0
            ),
        }

        try:
            log_path = Path(self.save_path_metric) / "remote_control_trace.jsonl"
            with open(log_path, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            return

    def _vehicle_control_to_dict(self, control):
        if control is None:
            return None
        return {
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "hand_brake": bool(control.hand_brake),
            "reverse": bool(control.reverse),
            "manual_gear_shift": bool(control.manual_gear_shift),
            "gear": int(control.gear),
        }

    def _array_summary(self, value, max_items: int = 5):
        if value is None:
            return {"present": False}
        if isinstance(value, torch.Tensor):
            array = value.detach().float().cpu().numpy()
        else:
            array = np.asarray(value)
        finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else np.zeros(array.shape, dtype=bool)
        flat = array.reshape(-1) if array.size else array
        return {
            "present": True,
            "shape": list(array.shape),
            "finite": bool(finite.all()) if finite.size else True,
            "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.number) else None,
            "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.number) else None,
            "min": float(np.nanmin(array)) if array.size and np.issubdtype(array.dtype, np.number) else None,
            "max": float(np.nanmax(array)) if array.size and np.issubdtype(array.dtype, np.number) else None,
            "first_values": self._to_jsonable(flat[:max_items]) if array.size else [],
        }

    def _visual_token_prune_summary(self, llm_payload: dict):
        prune_cfg = llm_payload.get("visual_token_prune")
        keep_ratio = llm_payload.get("visual_token_keep_ratio")
        if prune_cfg is None and keep_ratio is None:
            return {"present": False}

        summary = {
            "present": True,
            "config": self._to_jsonable(prune_cfg),
            "keep_ratio": self._array_summary(keep_ratio),
        }
        if isinstance(prune_cfg, dict):
            summary["mode"] = prune_cfg.get("mode")
            summary["prune_ratio"] = self._scalar_or_none(prune_cfg.get("prune_ratio"))
            summary["min_keep"] = self._to_int_or_none(prune_cfg.get("min_keep"))
        return summary

    def _scalar_or_none(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.reshape(-1)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int_or_none(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(-1)[0]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().reshape(-1)[0].item()
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_jsonable(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value
