#!/usr/bin/env bash

set -euo pipefail

CARLA_DIR="/data/gaoshuo/data/carla0915"
CONDA_ENV="simlingo"
CARLA_SESSION="simlingo-carla"
ENCODER_SESSION="simlingo-encoder-port-forward"

CARLA_COMMAND="cd '$CARLA_DIR' && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$CONDA_ENV' && exec ./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen"
ENCODER_COMMAND="exec kubectl -n default port-forward svc/simlingo-encoder-service 9011:9011"

usage() {
    printf 'Usage: %s --on|--off\n' "${0##*/}"
    printf '  --on   Start CARLA and the Encoder port-forward in tmux.\n'
    printf '  --off  Stop the two tmux tasks if they are running.\n'
}

start_tmux_session() {
    local session="$1"
    local command="$2"

    if tmux has-session -t "$session" 2>/dev/null; then
        printf 'tmux session already exists: %s\n' "$session"
    else
        tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$command")"
        printf 'Started tmux session: %s\n' "$session"
    fi
}

stop_tmux_session() {
    local session="$1"
    local task_name="$2"

    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
        printf 'Stopped %s (tmux session: %s).\n' "$task_name" "$session"
    else
        printf '%s is not running (tmux session not found: %s).\n' \
            "$task_name" "$session"
    fi
}

if ! command -v tmux >/dev/null 2>&1; then
    printf 'Error: tmux is not installed or is not in PATH.\n' >&2
    exit 1
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    --on)
        start_tmux_session "$CARLA_SESSION" "$CARLA_COMMAND"
        start_tmux_session "$ENCODER_SESSION" "$ENCODER_COMMAND"
        printf 'Attach with: tmux attach -t %s  or  tmux attach -t %s\n' \
            "$CARLA_SESSION" "$ENCODER_SESSION"
        ;;
    --off)
        stop_tmux_session "$CARLA_SESSION" "CARLA"
        stop_tmux_session "$ENCODER_SESSION" "Encoder port-forward"
        ;;
    -h|--help)
        usage
        ;;
    *)
        printf 'Error: unknown option: %s\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
