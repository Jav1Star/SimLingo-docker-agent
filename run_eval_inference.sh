#!/bin/bash
set -euo pipefail

WORK_DIR="/home/yyj/1-4-J-104/simlingo-heuristic"
INPUT_DIR="/home/yyj/1-4-J-104/input"
TEMPLATE_CONFIG="${WORK_DIR}/configs/test_algo1-4-J-104.yaml"
GENERATED_CONFIG_DIR="${WORK_DIR}/configs/test_algo1-4-J-104"
START_EVAL="${WORK_DIR}/start_eval_simlingo_adaption.py"

# 开启 nullglob，确保如果没有匹配的文件，数组为空而不是返回通配符字符串
shopt -s nullglob
yaml_files=("${INPUT_DIR}"/*.yaml "${INPUT_DIR}"/*.yml)
shopt -u nullglob

file_count=${#yaml_files[@]}

# 校验目录下文件的数量
if [[ $file_count -eq 0 ]]; then
    echo "Error: No .yaml or .yml files found in ${INPUT_DIR}." >&2
    exit 1
elif [[ $file_count -gt 1 ]]; then
    echo "Error: Multiple .yaml or .yml files found in ${INPUT_DIR}. Expected exactly one." >&2
    for f in "${yaml_files[@]}"; do
        echo "  - $f" >&2
    done
    exit 1
fi

# 获取唯一的配置文件路径
INPUT_PATH="${yaml_files[0]}"
echo "Auto-detected input file: ${INPUT_PATH}"

# 清理网络代理并设置 HF 环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset proxy_url
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

# 生成本次评估的两份独立配置，避免覆盖模板配置。
GENERATED_OUTPUT="$(python - "${INPUT_PATH}" "${TEMPLATE_CONFIG}" "${GENERATED_CONFIG_DIR}" "${TIMESTAMP}" <<'PY'
import sys
from pathlib import Path
import yaml

input_path = Path(sys.argv[1])
template_path = Path(sys.argv[2])
generated_dir = Path(sys.argv[3])
timestamp = sys.argv[4]

with input_path.open("r", encoding="utf-8") as f:
    input_cfg = yaml.safe_load(f) or {}

required_sections = ("route", "output", "test_rule_based", "test_fixed")
missing_sections = [section for section in required_sections if section not in input_cfg]
if missing_sections:
    raise SystemExit(f"Input yaml missing required sections: {', '.join(missing_sections)}")

route_ids = input_cfg["route"].get("route_ids")
out_root = input_cfg["output"].get("out_root")
if not isinstance(route_ids, list) or not route_ids:
    raise SystemExit("Input yaml key 'route.route_ids' must be a non-empty list.")
if not out_root:
    raise SystemExit("Input yaml missing required key: output.out_root")

with template_path.open("r", encoding="utf-8") as f:
    template_cfg = yaml.safe_load(f) or {}

for key_path in (("eval", "route_ids"), ("budget", "mode"), ("token_prune", "prune_ratio")):
    section, key = key_path
    if section not in template_cfg or key not in template_cfg[section]:
        raise SystemExit(f"Template config missing required key: {section}.{key}")

def build_config(test_key):
    test_cfg = input_cfg[test_key]
    for key in ("mode", "prune_ratio"):
        if key not in test_cfg:
            raise SystemExit(f"Input yaml missing required key: {test_key}.{key}")

    eval_cfg = yaml.safe_load(yaml.safe_dump(template_cfg, sort_keys=False))
    eval_cfg["eval"]["route_ids"] = [str(route_id) for route_id in route_ids]
    eval_cfg["eval"]["out_root"] = str(out_root)
    eval_cfg["budget"]["mode"] = str(test_cfg["mode"])
    eval_cfg["token_prune"]["prune_ratio"] = float(test_cfg["prune_ratio"])
    return eval_cfg

def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

generated_dir.mkdir(parents=True, exist_ok=True)
route_name = safe_name("_".join(str(route_id) for route_id in route_ids))
outputs = []
for label, test_key in (("rule_based", "test_rule_based"), ("fixed", "test_fixed")):
    output_path = generated_dir / f"{label}_{route_name}_{timestamp}.yaml"
    eval_cfg = build_config(test_key)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(eval_cfg, f, sort_keys=False, allow_unicode=True)
    outputs.append(output_path)
    print(
        f"Generated {output_path} with route_ids={eval_cfg['eval']['route_ids']} "
        f"out_root={eval_cfg['eval']['out_root']} mode={eval_cfg['budget']['mode']} "
        f"prune_ratio={eval_cfg['token_prune']['prune_ratio']}",
        file=sys.stderr,
    )

for output_path in outputs:
    print(output_path)
PY
)"
mapfile -t GENERATED_CONFIGS <<< "${GENERATED_OUTPUT}"
if [[ ${#GENERATED_CONFIGS[@]} -ne 2 ]]; then
    echo "Error: Expected 2 generated eval configs, got ${#GENERATED_CONFIGS[@]}." >&2
    exit 1
fi

RULE_BASED_CONFIG="${GENERATED_CONFIGS[0]}"
FIXED_CONFIG="${GENERATED_CONFIGS[1]}"

echo "Running rule_based evaluation on GPU 0: ${RULE_BASED_CONFIG}"
python "${START_EVAL}" --eval-config "${RULE_BASED_CONFIG}" --gpu 0

echo "Running fixed evaluation on GPU 0: ${FIXED_CONFIG}"
python "${START_EVAL}" --eval-config "${FIXED_CONFIG}" --gpu 0
