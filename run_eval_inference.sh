#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="/data/gaoshuo/1-4-J-104/input"
EVAL_SCRIPT="/data/gaoshuo/1-4-J-104/SimLingo-docker-agent/start_eval_split_agents_local.py"

usage() {
    printf 'Usage: %s --rule_based|--fixed\n' "${0##*/}"
    printf '  --rule_based  Use rule_based.yaml as the evaluation config.\n'
    printf '  --fixed       Use fixed.yaml as the evaluation config.\n'
}

if [[ $# -ne 1 ]]; then
    printf 'Error: exactly one mode, --rule_based or --fixed, is required.\n' >&2
    usage >&2
    exit 2
fi

case "$1" in
    --rule_based)
        config_name="rule_based.yaml"
        ;;
    --fixed)
        config_name="fixed.yaml"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        printf 'Error: invalid mode: %s\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -d "$INPUT_DIR" ]]; then
    printf 'Error: input directory does not exist: %s\n' "$INPUT_DIR" >&2
    exit 1
fi

mapfile -d '' test_option_dirs < <(
    find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name 'test_option*' -print0
)

if [[ ${#test_option_dirs[@]} -ne 1 ]]; then
    printf 'Error: expected exactly one test_option* directory under %s, found %d.\n' \
        "$INPUT_DIR" "${#test_option_dirs[@]}" >&2
    if [[ ${#test_option_dirs[@]} -gt 0 ]]; then
        printf 'Found directories:\n' >&2
        printf '  %s\n' "${test_option_dirs[@]}" >&2
    fi
    exit 1
fi

eval_config="${test_option_dirs[0]}/$config_name"
if [[ ! -f "$eval_config" ]]; then
    printf 'Error: evaluation config does not exist: %s\n' "$eval_config" >&2
    exit 1
fi

if [[ ! -x "$EVAL_SCRIPT" ]]; then
    printf 'Error: evaluation script is missing or not executable: %s\n' \
        "$EVAL_SCRIPT" >&2
    exit 1
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1

export ENCODER_UID="630f5a24-ee1a-49e4-89a7-7f57d6de1066"
export LLM_UID="156d26e6-a082-47b4-82b4-d23ba8cae863"
export SCHEDULER_UID="902ae0f7-8b8c-43a9-a211-f2c7a1918950"

printf 'Starting evaluation with config: %s\n' "$eval_config"

exec "$EVAL_SCRIPT" \
    --eval-config "$eval_config" \
    --gpu 0 \
    --carla-host 127.0.0.1 \
    --carla-port 2000 \
    --traffic-manager-port 8000 \
    --agent-profile nodeport \
    --encoder-url http://127.0.0.1:9011/a2a/execute \
    --scheduler-url http://10.112.221.121:9013/a2a/execute \
    --llm-url http://10.112.221.121:9012/a2a/execute \
    --nats-url nats://10.112.221.121:30422 \
    --local-cluster edge-b \
    --nats-jetstream-domain edge-b \
    --encoder-instance-id "$ENCODER_UID" \
    --scheduler-instance-id "$SCHEDULER_UID" \
    --llm-instance-id "$LLM_UID"
