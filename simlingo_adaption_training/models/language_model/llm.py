from transformers import LlamaModel, LlamaConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import GPTNeoXForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import Qwen2Tokenizer
from .internvl_2_1b.modeling_internvl_chat import InternVLChatModel

from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F

import torch
from torch import Tensor, nn

import os


def _find_local_internvl_path(variant: str, cache_dir: Optional[str] = None) -> str:
    repo_name = variant.split("/")[-1]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    candidate_paths = [
        os.path.join(repo_root, "models", repo_name),
    ]
    if cache_dir:
        candidate_paths.append(cache_dir if os.path.isabs(cache_dir) else os.path.join(repo_root, cache_dir))
    candidate_paths.append(os.path.join(os.path.dirname(__file__), "internvl_2_1b"))

    required_files = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    weight_files = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    missing_by_path = {}
    for path in candidate_paths:
        missing = [name for name in required_files if not os.path.exists(os.path.join(path, name))]
        has_weights = any(os.path.exists(os.path.join(path, name)) for name in weight_files)
        if not missing and has_weights:
            return path
        if os.path.exists(path):
            missing_by_path[path] = missing + ([] if has_weights else ["model weights"])
        else:
            missing_by_path[path] = ["directory"]

    details = "; ".join(f"{path}: missing {', '.join(missing)}" for path, missing in missing_by_path.items())
    raise FileNotFoundError(f"Could not find a complete local InternVL model for {variant}. {details}")


# ... (CONFIGS 字典保持不变) ...
CONFIGS: Dict[str, Dict[str, Any]] = {
    "debug": dict(num_hidden_layers=2, num_attention_heads=2, hidden_size=32, intermediate_size=64),
    "legacy-tiny": dict(num_hidden_layers=8, num_attention_heads=16, hidden_size=2048, intermediate_size=4096),
    "tiny": dict(num_hidden_layers=12, num_attention_heads=8, hidden_size=512, intermediate_size=2048),
    "x-small": dict(num_hidden_layers=14, num_attention_heads=8, hidden_size=1024, intermediate_size=4096),
    "small": dict(num_hidden_layers=22, num_attention_heads=8, hidden_size=1024, intermediate_size=4096),
    "medium": dict(num_hidden_layers=22, num_attention_heads=12, hidden_size=1536, intermediate_size=4096),
    "large": dict(num_hidden_layers=22, num_attention_heads=16, hidden_size=2048, intermediate_size=5632),
    "gaia-large": dict(num_hidden_layers=22, num_attention_heads=16, hidden_size=1536, intermediate_size=4096),
    "7B": dict(num_hidden_layers=32, num_attention_heads=32, hidden_size=4096, intermediate_size=11008),
    "13B": dict(num_hidden_layers=40, num_attention_heads=40, hidden_size=5120, intermediate_size=11008),
    "70B": dict(num_hidden_layers=80, num_attention_heads=64, hidden_size=8192, intermediate_size=28672, num_key_value_heads=8),
    "tiny-llama-1.1b": dict(num_hidden_layers=22, num_attention_heads=32, hidden_size=2048, intermediate_size=5632, num_key_value_heads=4),
    "phi": dict(num_hidden_layers=24, num_attention_heads=32, hidden_size=2048, intermediate_size=8192, partial_rotary_factor=0.5, bias=True, vocab_size=51200, norm_type="layer_norm", mlp_type="gelu_new", parallel_attn_mlp=True),
}

class LLM(nn.Module):
    def __init__(self,
                 **cfg,
            ):
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
        if 'pythia' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            self.variant = f'EleutherAI/{self.variant}'
            self.model = GPTNeoXForCausalLM.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True)
            self.model.embed_tokens = self.model.base_model.embed_in
        elif 'paligemma' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            self.variant = f'google/{self.variant}'
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.variant, torch_dtype="auto", revision="float16").language_model
            self.model.embed_tokens = self.model.base_model.embed_tokens
            self.tokenizer = AutoProcessor.from_pretrained(self.variant, torch_dtype="auto").tokenizer
        elif 'TinyLlama' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            print('Loading pretrained model')
            self.model = AutoModelForCausalLM.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True)
            self.model.embed_tokens = self.model.base_model.embed_tokens
        elif 'llava-v1.6' in self.variant:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            print('Loading pretrained model')
            self.model = LlavaNextForConditionalGeneration.from_pretrained(self.variant, trust_remote_code=True)
            self.tokenizer = LlavaNextProcessor.from_pretrained(self.variant, torch_dtype="auto",  trust_remote_code=True).tokenizer
            self.model = self.model.language_model
            self.model.embed_tokens = self.model.base_model.embed_tokens
        # === [修改部分] 对接 InternVL2-1B (本地修改版) ===
        elif 'internvl' in self.variant.lower():
            local_model_path = _find_local_internvl_path(self.variant, getattr(self, "cache_dir", None))
            print(f'Loading local InternVL model from: {local_model_path}')
            
            # 2. 加载 Wrapper 模型 (InternVLChatModel)
            # 直接使用本地自定义类来初始化
            self.model = InternVLChatModel.from_pretrained(
                local_model_path, 
                torch_dtype=torch.bfloat16
            )
            
            # === [关键修复] 去壳：只保留语言模型部分 ===
            # 这样 self.model 就变成了 Qwen2ForCausalLM (你修改过的那个类)
            self.model = self.model.language_model 
            # ==========================================
            # 3. 加载 Qwen2Tokenizer
            self.tokenizer = Qwen2Tokenizer.from_pretrained(local_model_path)
            
            # 4. 绑定 Embeddings
            # SimLingo 需要访问 embed_tokens 进行采样
            # InternVL 的结构是 Wrapper -> language_model (Qwen2) -> embed_tokens
            # self.model.embed_tokens = self.model.language_model.get_input_embeddings()
            self.model.embed_tokens = self.model.get_input_embeddings()
        else:
            raise ValueError(f"Carefull: Variant {self.variant} not tested.")
            config_overrides = CONFIGS[self.variant].copy()
            configuration = LlamaConfig(**config_overrides)
            self.model = LlamaModel(configuration)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
            self.model.embed_tokens = self.model.base_model.embed_tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.max_position_embeddings = self.model.config.max_position_embeddings
        self.config = self.model.config
        
    def get_lora_model(self):
        if self.lora:
            from peft import get_peft_model
            from peft import LoraConfig
            
            print('Using PEFT model')
            
            layers_to_transform = None
            # [修改] 如果启用了 adaptionTrain，则仅在前 num_prefix_layers 层应用 LoRA
            if self.adaption_train:
                num_prefix = self.num_prefix_layers
                if num_prefix is None:
                    raise ValueError("num_prefix_layers must be set when adaption_train is True.")
                
                layers_to_transform = list(range(num_prefix))
                print(f"AdaptionTrain Enabled: LoRA restricted to first {num_prefix} layers.")
            target_modules_for_qwen = [ # 等效于"all-linear"
                        "q_proj",    # Attention Query
                        "k_proj",    # Attention Key
                        "v_proj",    # Attention Value
                        "o_proj",    # Attention Output
                        "gate_proj", # MLP Gate
                        "up_proj",   # MLP Up-projection
                        "down_proj"  # MLP Down-projection
                    ]
            peft_config = LoraConfig(
                inference_mode=False, 
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                layers_to_transform=layers_to_transform, # 指定LoRA注入的层，其他的层保持原始状态
                target_modules=target_modules_for_qwen,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()



    def forward(self,
        embeddings: Tensor,
        attention_mask: Tensor = None,
        return_dict: bool = True,
        position_ids: Optional[Tensor] = None,
        execution_plan: Optional[Tensor] = None,
        assigner: Optional[object] = None,
        **kwargs
    ) -> Tensor:

        # 1. 准备传递给 Qwen2 (LLM) 的参数
        # 注意：这里只传 inputs_embeds，因为视觉信息已经在 embeddings 里了
        model_inputs = {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "output_hidden_states": True, # AdaLLaVA 需要 hidden states
            "position_ids": position_ids,
            "return_dict": return_dict
        }

        if execution_plan is not None:
            model_inputs["execution_plan"] = execution_plan
        elif assigner is not None:
            model_inputs["assigner"] = assigner

        # 3. 调用模型 (self.model 应该是 Qwen2ForCausalLM 或其变体)
        # 不需要传 pixel_values，因为它只处理 embeddings
        outputs = self.model(**model_inputs)

        # 4. 获取输出 (Qwen2 返回 CausalLMOutputWithPast)
        if hasattr(outputs, "hidden_states"):
            features = outputs.hidden_states[-1]
        else:
            # 兼容 tuple 返回
            features = outputs[-1] 

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        return features, logits

    # ... (后续 sample_categorical 和 greedy_sample 方法保持不变) ...
    # 可以在 greedy_sample 的 forward 调用中也加入 **kwargs 以防万一，
    # 但 greedy_sample 通常只在推理时用，且 SimLingo 的逻辑可能是在 run_step 中手动控制 forward
    
    def sample_categorical(
        self,
        logits: Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
    ):
        if restrict_tokens is not None:
            logits[..., : restrict_tokens[0]] = -float("inf")
            logits[..., restrict_tokens[0] + restrict_tokens[1] :] = -float("inf")

        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=False)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("inf"), logits)

        temperature = max(temperature, 1e-9)
        logits = logits / temperature

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Shift the indices to the right to keep also the first token above the threshold
            mask = (cumulative_probs > top_p).roll(shifts=1, dims=-1)
            mask[..., 0] = False
            logits[mask.gather(-1, sorted_indices.argsort(-1))] = -float("inf")

        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)

    def greedy_sample(
        self,
        input_embeds: Tensor,
        inputs_mask: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        cache_offset: int = 0,
        input_embed_matrix: Optional[Tensor] = None,
        logit_matrix: Optional[Tensor] = None,
        restrict_tokens: Optional[Tuple[int, int]] = None,
        attention_mask = None,
        position_ids = None,
        assigner: Optional[object] = None,
    ) -> Tuple[Tensor, int]:
        
        if input_embed_matrix is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "No input embeddings available because the model doesn't define a vocab. "
                    "Please provide input_embed_matrix. "
                )
            input_embed_matrix = self.embed_tokens.weight
            
        # === [修改开始: 适配 logit_matrix 获取逻辑] ===
        if logit_matrix is None:
            # 1. 优先尝试从 self.model 中获取 (针对 LLM Wrapper 结构)
            # 适配 InternLM2 (使用 .output)
            if hasattr(self.model, "output") and isinstance(self.model.output, nn.Linear):
                logit_matrix = self.model.output.weight
            # 适配 Llama / Qwen 等标准模型 (使用 .lm_head)
            elif hasattr(self.model, "lm_head") and isinstance(self.model.lm_head, nn.Linear):
                logit_matrix = self.model.lm_head.weight
            # 2. 尝试从 self 获取 (以防 lm_head 被挂载到了 Wrapper 上)
            elif hasattr(self, "lm_head") and self.lm_head is not None:
                logit_matrix = self.lm_head.weight
            else:
                raise ValueError(
                    "No logit matrix available. Could not find 'output' or 'lm_head' layer in model."
                    "Please provide logit_matrix explicitely."
                )
        # === [修改结束] ===
        # We generate tokens up to the eos token id if provided. If not provided, we generate until the end.
        # If no eos token id is provided, use -1 instead, so we will never stop generating until 'new_token'.
        sampled_tokens = torch.empty((input_embeds.size(0), max_new_tokens), device=input_embeds.device, dtype=torch.long)
        if eos_token_id is not None:
            sampled_tokens.fill_(eos_token_id)

        # we start with all sequences left to complete
        incomplete_seq_mask = torch.ones(input_embeds.size(0), dtype=torch.bool, device=input_embeds.device)
        for i in range(max_new_tokens):
            # === [修复 2: 在循环调用 forward 时传递参数] ===
            # 如果不传，模型就收不到指令，也就不会剪枝
            features, logits = self.forward(
                embeddings=input_embeds, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                assigner=assigner,
            )
            # ============================================

            last_hidden_state = features[:, -1]

            # sample the next token
            # 注意: 这里 self.lm_head 可能需要适配。
            # InternLM2ForCausalLM 有 self.model.output (作为 head) 或 self.output
            # 原始代码用 logit_matrix，可能是直接传参。
            # 如果没有传，这里可能需要根据新模型调整获取 head 权重的方式
            
            logits = F.linear(last_hidden_state, logit_matrix)
            
            next_token = self.sample_categorical(
                logits, temperature=temperature, top_k=top_k, top_p=top_p, restrict_tokens=restrict_tokens
            )
            x = F.embedding(next_token.unsqueeze(1), input_embed_matrix)

            input_embeds = torch.cat([input_embeds, x], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((input_embeds.size(0), 1), device=input_embeds.device)], dim=1)

            # only update sequences where we haven't predicted the eos token before
            sampled_tokens[incomplete_seq_mask, i] = next_token[incomplete_seq_mask]
            # For completed sequences, we mask them out for future sampling
            # self.cache_mask[:, cache_offset - 1] &= incomplete_seq_mask

            if eos_token_id is not None:
                # only update the mask of incomplete sequences and stop early if an eos token id is provided.
                incomplete_seq_mask = sampled_tokens[:, i] != eos_token_id
                if not incomplete_seq_mask.any():
                    # finished all sequences, early exit
                    sampled_tokens = sampled_tokens[:, : i + 1]
                    break

        return sampled_tokens, input_embeds

if __name__ == "__main__":
    model = LLM(variant="x-small", lora=False) # 修正参数传递以匹配 __init__ 的 **cfg
    print(model)
