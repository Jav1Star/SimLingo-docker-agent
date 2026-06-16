from typing import Optional

import numpy as np
import torch

from navsim.common.dataclasses import AgentInput, NAVSIM_INTERVAL_LENGTH
from simlingo_adaption_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_adaption_training.utils.internvl2_utils import (
    get_custom_chat_template,
    get_num_image_tokens_per_patch,
    preprocess_image_batch,
)
from team_code_adaption.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics


class DrivingInputBuilder:
    def __init__(self, runtime):
        self.runtime = runtime
        self.device = runtime.device
        self.use_global_img = runtime.cfg.model.vision_model.use_global_img
        self.encoder_variant = runtime.cfg.model.vision_model.variant
        self.num_image_tokens_total = get_num_image_tokens_per_patch(self.encoder_variant) * 2

    def build(
        self,
        agent_input: AgentInput,
        target_points_by_frame,
        step_idx: int,
        global_ego_pose: Optional[np.ndarray] = None,
        timestamp_s: Optional[float] = None,
    ) -> DrivingInput:
        ego_status = agent_input.ego_statuses[step_idx]
        camera = agent_input.cameras[step_idx].cam_f0.image
        speed = float(np.linalg.norm(ego_status.ego_velocity))
        target_points = target_points_by_frame[step_idx]
        ego_pose = ego_status.ego_pose if global_ego_pose is None else np.asarray(global_ego_pose, dtype=np.float32)
        timestamp_value = float(step_idx * NAVSIM_INTERVAL_LENGTH if timestamp_s is None else timestamp_s)

        image_tensor = torch.from_numpy(np.asarray(camera).transpose(2, 0, 1))
        images_processed = preprocess_image_batch(
            [image_tensor],
            input_size=448,
            use_global_img=self.use_global_img,
            max_num_grid=2,
        )
        processed_image = images_processed["pixel_values"]
        processed_image = processed_image.view(1, 1, processed_image.shape[1], 3, processed_image.shape[3], processed_image.shape[4])

        # 关键调用点：prompt 保持 target_point 语义，不把 navsim driving_command 混进来。
        prompt = (
            f"Current speed: {speed:.1f} m/s. "
            "Target waypoint: <TARGET_POINT><TARGET_POINT>. "
            "Predict the waypoints."
        )
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Waypoints:"}],
            },
        ]
        conversation_dict, question_dict = get_custom_chat_template(
            [conversation],
            self.runtime.tokenizer,
            self.encoder_variant,
            self.num_image_tokens_total,
        )

        placeholder_values = {
            self.runtime.tokenizer.convert_tokens_to_ids("<TARGET_POINT>"): target_points,
        }
        prompt_label = LanguageLabel(
            phrase_ids=conversation_dict["phrase_ids"].to(self.device),
            phrase_valid=conversation_dict["phrase_valid"].to(self.device),
            phrase_mask=conversation_dict["phrase_mask"].to(self.device),
            placeholder_values=[placeholder_values],
            language_string=conversation_dict["language_string"],
            loss_masking=conversation_dict["loss_masking"].to(self.device),
        )
        question_label = LanguageLabel(
            phrase_ids=question_dict["phrase_ids"].to(self.device),
            phrase_valid=question_dict["phrase_valid"].to(self.device),
            phrase_mask=question_dict["phrase_mask"].to(self.device),
            placeholder_values=[placeholder_values],
            language_string=question_dict["language_string"],
            loss_masking=question_dict["loss_masking"].to(self.device),
        )

        height, width = camera.shape[0], camera.shape[1]
        return DrivingInput(
            camera_images=processed_image.to(self.device).bfloat16(),
            image_sizes=images_processed["image_sizes"].to(self.device),
            camera_intrinsics=get_camera_intrinsics(width, height, 110).unsqueeze(0).float().to(self.device),
            camera_extrinsics=get_camera_extrinsics().unsqueeze(0).float().to(self.device),
            vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            target_point=torch.from_numpy(target_points).unsqueeze(0).to(self.device, dtype=torch.float32),
            prompt=prompt_label,
            prompt_inference=question_label,
            # smart_assigner 的历史对齐逻辑在训练时使用世界系 pose；这里保持同一语义。
            ego_xy=torch.from_numpy(np.asarray(ego_pose[:2], dtype=np.float32)).unsqueeze(0).to(self.device),
            ego_yaw=torch.tensor([float(ego_pose[2])], dtype=torch.float32, device=self.device),
            timestamp=torch.tensor([timestamp_value], dtype=torch.float32, device=self.device),
        )
