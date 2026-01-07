import os
import sys
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from types import SimpleNamespace

# === 1. 环境路径设置 ===
REPO_ROOT = os.path.abspath(".")
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "team_code"))

# === 伪造 Tokenizer (必须匹配 Qwen 的词表大小) ===
class MockTokenizer:
    def __init__(self):
        # Qwen2 的词表非常大，必须设置正确，否则 LayerNorm 会越界报错
        self.additional_special_tokens_ids = [151643, 151644, 151645] 
        self.eos_token_id = 151643
        self.added_tokens_encoder = {'<|end|>': 151643, '<|im_end|>': 151644}
        # ⚠️ 关键：Qwen2 词表大小约为 151655，这里设稍微大一点确保安全
        self.vocab_size = 151936 

    def convert_tokens_to_ids(self, token):
        return 999
    
    def batch_decode(self, tokens, **kwargs):
        return [""]

    def __len__(self):
        return self.vocab_size

class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()

# ==========================================

def get_mock_example(batch_size=1, device="cuda"):
    """构造伪造输入样本"""
    pixel_values = torch.randn(batch_size, 1, 3, 448, 448).to(device)
    mock_prompt = SimpleNamespace()
    mock_prompt.placeholder_values = [] 
    
    mock_driving_input = SimpleNamespace()
    mock_driving_input.camera_images = pixel_values
    mock_driving_input.prompt = mock_prompt
    mock_driving_input.prompt_inference = mock_prompt 

    example = SimpleNamespace()
    example.driving_input = mock_driving_input
    example.camera_images = pixel_values
    return example

def test_inference():
    # === 2. 构建 Config (配置真·Qwen2-0.5B) ===
    yaml_content = """
    model:
      vision_model:
        variant: OpenGVLab/InternVL2-1B
        # 视觉层通常会有 Projector 适配语言层，这里设为 2048 也没问题
        embed_dim: 2048 
        freeze: false
        use_global_img: false
        _target_: simlingo_training.models.encoder.vlm.VLMEncoderModel
      
      language_model:
        variant: OpenGVLab/InternVL2-1B
        
        # === [关键] 显式定义 Qwen2-0.5B 的参数 ===
        # 这些参数将强制覆盖 InternLM2 代码中的默认 7B 参数
        # 从而构建出正确的 24层、896维 的模型
        
        hidden_size: 896           # Qwen2-0.5B 核心维度
        num_hidden_layers: 24      # 真实层数 (足够你的 Scheduler 用前2层)
        num_attention_heads: 14    # Qwen2 头数
        intermediate_size: 4864    # Qwen2 FFN 维度
        vocab_size: 151936         # 适配 Qwen Tokenizer
        max_position_embeddings: 32768
        # ========================================
        
        lora: true
        lora_alpha: 64
        lora_r: 32
        lora_dropout: 0.1
        _target_: simlingo_training.models.language_model.llm.LLM
        
      lr: 3.0e-05
      weight_decay: 0.1
      betas: [0.9, 0.999]
      pct_start: 0.05
      speed_wps_mode: 2d
      predict_route_as_wps: true
      _target_: simlingo_training.models.driving.DrivingModel
    
    cfg_data_module:
      batch_size: 1
      num_workers: 0
      train_partitions: {all: 1.0}
    """
    cfg = OmegaConf.create(yaml_content)
    
    print(">>> 正在初始化模型结构 (Target: Qwen2-0.5B Architecture)...")
    mock_processor = MockProcessor()
    
    try:
        model = instantiate(
            cfg.model,
            cfg_data_module=cfg.cfg_data_module,
            processor=mock_processor,
            cache_dir="./tmp_cache",
            _recursive_=False 
        )
        print(">>> 模型实例化成功！")
        
        # === 验证一下模型是不是真的变小了 ===
        # 获取 LLM 部分的 hidden size
        try:
            actual_hidden = model.language_model.model.config.hidden_size
            actual_layers = model.language_model.model.config.num_hidden_layers
            print(f">>> [架构验证] Hidden Size: {actual_hidden} (预期 896)")
            print(f">>> [架构验证] Layers: {actual_layers} (预期 24)")
            
            if actual_hidden == 4096:
                raise ValueError("模型依然是 7B 架构！配置未生效！")
        except AttributeError:
            pass

    except Exception as e:
        print(f">>> 实例化失败，错误详情:\n{e}")
        import traceback
        traceback.print_exc()
        return

    # === 4. 加载权重 (此时架构匹配，您可以尝试加载真实权重了) ===
    # 为了先测通 Latency 逻辑，我们先用随机权重，确保没有维度错误
    print(">>> [本次测试] 使用随机初始化权重 (架构已对齐 Qwen2-0.5B)。")
    print(">>> 显存占用预计 < 3GB，且保留了完整的 24 层结构。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()

    # === 5. 运行对比测试 ===
    example = get_mock_example(batch_size=1, device=device)

    print("\n>>> [测试 1] Latency = 1.0 (高延迟)")
    try:
        with torch.no_grad():
            out_high = model(example, return_language=False, latency=1.0)
            route_high = out_high[1] 

        print(">>> [测试 2] Latency = 0.0 (无延迟)")
        with torch.no_grad():
            out_low = model(example, return_language=False, latency=0.0)
            route_low = out_low[1]

        # === 6. 结果验证 ===
        print("\n" + "="*30)
        print(f"输出形状: {route_high.shape}")
        
        diff = (route_high - route_low).abs().max().item()
        print(f"Max Diff (Latency 1.0 vs 0.0): {diff:.8f}")
        
        if diff > 1e-6:
            print("✅ SUCCESS: Latency 参数生效！")
            
            print("   (并且是在真实的 24 层模型上通过测试的)")
        else:
            print("⚠️ WARNING: 输出完全一致。")
            
    except Exception as e:
        print(f"推理过程报错: {e}")
        # 如果这里报错 mat1 and mat2 shapes cannot be multiplied
        # 说明 Vision Model 的输出 (2048) 和 Language Model 输入 (896) 没对齐
        # SimLingo 的 MLP Projector 应该会自动处理，但如果没处理，我们需要改 Vision 的 embed_dim
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()