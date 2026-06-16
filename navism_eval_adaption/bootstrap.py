from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoProcessor

from simlingo_adaption_training.utils.internvl2_utils import resolve_internvl_model_path


def resolve_bootstrap_config_path(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path.parent.parent / ".hydra" / "config.yaml",
        checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(checkpoint_path)


def extract_checkpoint_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


class SmartAssignerRuntime:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.config_path = resolve_bootstrap_config_path(self.checkpoint_path)
        self.device = torch.device(device)

        self.cfg = OmegaConf.load(self.config_path)
        self.cfg.model.vision_model.use_global_img = self.cfg.data_module.use_global_img
        self._patch_legacy_config()

        processor_path = resolve_internvl_model_path(self.cfg.model.vision_model.variant)
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
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

        repo_name = self.cfg.model.vision_model.variant.split("/")[-1]
        cache_dir = str((Path(__file__).resolve().parents[1] / "pretrained" / repo_name).resolve())

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            cfg_data_module=self.cfg.data_module,
            processor=self.processor,
            cache_dir=cache_dir,
            _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)

        checkpoint_obj = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = extract_checkpoint_state_dict(checkpoint_obj)
        self.model.load_state_dict(state_dict, strict=False)
        # 关键调用点：这里固定走 smart_assigner stage2 的原始推理链，不改模型内部时序。
        self.model.set_eval_runtime_profile("smart_assigner")
        if hasattr(self.model, "set_eval_predict_language"):
            self.model.set_eval_predict_language(False)
        else:
            self.model.predict_language = False
        self.model.eval()

    def _patch_legacy_config(self):
        smart_train_cfg = getattr(self.cfg.model, "smart_assigner_train", None)
        if smart_train_cfg is None:
            return
        stage2_cfg = getattr(smart_train_cfg, "stage2", None)
        if stage2_cfg is None:
            return
        if not hasattr(stage2_cfg, "driving_loss_weight"):
            stage2_cfg.driving_loss_weight = 0.1
        if not hasattr(stage2_cfg, "driving_loss_warmup_steps"):
            stage2_cfg.driving_loss_warmup_steps = 1000
        if not hasattr(stage2_cfg, "grpo_prefix_kl_beta"):
            stage2_cfg.grpo_prefix_kl_beta = 0.0
