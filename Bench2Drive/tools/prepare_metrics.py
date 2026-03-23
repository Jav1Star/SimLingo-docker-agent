import os
import json
import glob

# 1. 你的原始数据根目录
viz_root = "/home/yyj/simlingo-adaption/eval_results/Bench2Drive/simlingo/bench2drive/42/random/viz"
# 2. 你想生成的、给统计脚本用的标准目录
target_metric_dir = "/home/yyj/simlingo-adaption/eval_results/Bench2Drive/simlingo/bench2drive/42/random/metrics_standard"

os.makedirs(target_metric_dir, exist_ok=True)

# 搜索所有的 metric_info.json
metric_files = glob.glob(os.path.join(viz_root, "*/debug_viz/*/*/*/metric/metric_info.json"))

print(f"找到 {len(metric_files)} 个物理指标文件，开始建立映射...")

for src_path in metric_files:
    # 从路径中提取 save_name (去掉前面的 3 位数字 ID)
    # 路径示例：.../viz/000RouteScenario_1711_.../debug_viz/...
    parts = src_path.split('/')
    viz_folder_name = ""
    for p in parts:
        if "RouteScenario" in p:
            viz_folder_name = p
            break
    
    if not viz_folder_name:
        continue
        
    # 去掉前 3 位数字 ID，获取真正的 save_name
    save_name = viz_folder_name[3:] 
    
    # 创建目标子文件夹
    save_dir = os.path.join(target_metric_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 建立软链接
    dst_path = os.path.join(save_dir, "metric_info.json")
    if not os.path.exists(dst_path):
        os.symlink(src_path, dst_path)

print(f"映射完成！现在你可以运行统计脚本了。")