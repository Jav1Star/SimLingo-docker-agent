from transformers import AutoModel, AutoTokenizer

model_path = "/home/yangyujia/models/InternVL2-1B"

print(f"尝试加载模型: {model_path}")
try:
    # 注意：InternVL2 通常需要 trust_remote_code=True
    cfg = {'lora': True, 'lora_alpha': 64, 'lora_r': 32, 'lora_dropout': 0.1, 'cache_dir': 'pretrained/InternVL2-1B'} 
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto",
        **cfg
    )
    print("✅ 模型加载成功！文件是完好的。")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")