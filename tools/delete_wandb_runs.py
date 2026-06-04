import wandb
import shutil
from pathlib import Path

# 请根据你的实际情况修改你的 Entity(用户名或团队名) 和 Project
# 从你给出的 URL 推断: https://wandb.ai/yujiayang61-bupt/simlingo/...
ENTITY = "yujiayang61-bupt"
PROJECT = "simlingo"
PREFIX_TO_DELETE = "2026_04_24_"
EXCLUDE_RUN_NAME = "2026_04_24_12_46_20_"
OUTPUTS_DIR = Path("outputs")

def main():
    api = wandb.Api()
    
    # 获取项目下的所有 runs
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    deleted_count = 0
    print(f"开始检查项目 {ENTITY}/{PROJECT} 下的 runs...")
    
    for run in runs:
        # 如果 run 的名字以指定的日期开头
        if run.name.startswith(PREFIX_TO_DELETE):
            # 但要保留特定的这次运行，或者以该时间开头的 run
            if run.name.startswith(EXCLUDE_RUN_NAME):
                print(f"跳过被保护的 Run: {run.name} (ID: {run.id})")
                continue
            
            # 删除符合条件的 run
            print(f"正在删除 Run: {run.name} (ID: {run.id})")
            run.delete()
            deleted_count += 1

            # 同步删除 outputs 下同名文件夹
            run_output_dir = OUTPUTS_DIR / run.name
            if run_output_dir.exists() and run_output_dir.is_dir():
                print(f"正在删除本地目录: {run_output_dir}")
                shutil.rmtree(run_output_dir)
            else:
                print(f"未找到同名本地目录，跳过: {run_output_dir}")
            
    print(f"清理完成，共删除了 {deleted_count} 个 runs。")

if __name__ == "__main__":
    main()
