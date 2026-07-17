#!/usr/bin/env bash

# Sequentially evaluate routes 020-100. Each route gets at most 20 minutes.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RES_DIR="${SCRIPT_DIR}/eval_results/split_agents_local/simlingo_remote_split/bench2drive/66/rule_based/res"
SCORE_FILE="${SCRIPT_DIR}/routes_score_100.txt"
TARGET_PERFECT_ROUTES=7

trap 'echo "[runner] interrupted; stopping batch"; exit 130' INT TERM

collect_perfect_routes() {
    python3 - "${RES_DIR}" "${SCORE_FILE}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

res_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
perfect_routes = []

if res_dir.is_dir():
    for result_path in res_dir.glob("*_res.json"):
        match = re.fullmatch(r"(\d+)_res\.json", result_path.name)
        if not match:
            continue
        try:
            with result_path.open("r", encoding="utf-8") as result_file:
                result = json.load(result_file)
            score = result["_checkpoint"]["global_record"]["scores_mean"]["score_composed"]
            if float(score) == 100.0:
                perfect_routes.append(match.group(1))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[score] skip unreadable result {result_path}: {exc}", file=sys.stderr)

perfect_routes.sort(key=int)
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = output_path.with_name(f".{output_path.name}.tmp")
with temporary_path.open("w", encoding="utf-8") as output_file:
    for route_id in perfect_routes:
        output_file.write(f"{route_id}\n")
os.replace(temporary_path, output_path)

print(f"[score] score_composed=100 routes ({len(perfect_routes)}): "
      f"{', '.join(perfect_routes) if perfect_routes else 'none'}")
print(f"[score] updated: {output_path}")
PY
}

stop_if_target_reached() {
    local perfect_count
    perfect_count=$(wc -l < "${SCORE_FILE}")
    if (( perfect_count >= TARGET_PERFECT_ROUTES )); then
        echo "[$(date '+%F %T')] reached ${perfect_count} routes with score_composed=100; stopping batch"
        exit 0
    fi
}

cd "${SCRIPT_DIR}" || exit 1

# Count results that already exist before starting another route.
collect_perfect_routes
stop_if_target_reached

for route_number in $(seq 27 100); do
    route_id=$(printf '%03d' "${route_number}")
    echo "[$(date '+%F %T')] starting route ${route_id}"

    timeout --signal=TERM --kill-after=30s 20m \
        ./start_eval_split_agents_local.py \
        --eval-config configs/simlingo_adaption_eval.yaml \
        --route-id "${route_id}" \
        --seed 66 \
        --gpu 0 \
        --carla-host 127.0.0.1 \
        --carla-port 2000 \
        --traffic-manager-port 8000
    exit_code=$?

    if [[ ${exit_code} -eq 124 || ${exit_code} -eq 137 ]]; then
        echo "[$(date '+%F %T')] route ${route_id} exceeded 20 minutes and was terminated"
    elif [[ ${exit_code} -ne 0 ]]; then
        echo "[$(date '+%F %T')] route ${route_id} failed with exit code ${exit_code}; continuing"
    else
        echo "[$(date '+%F %T')] route ${route_id} completed"
    fi

    collect_perfect_routes
    stop_if_target_reached
done

echo "[$(date '+%F %T')] all routes 027-100 have been processed"
