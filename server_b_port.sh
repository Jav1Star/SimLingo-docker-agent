#!/usr/bin/env bash

set -euo pipefail

NATS_SESSION="server-b-nats-port-forward"
LLM_SESSION="server-b-llm-port-forward"
SCHEDULER_SESSION="server-b-scheduler-port-forward"

NATS_COMMAND="exec kubectl -n default port-forward --address 10.112.221.121 svc/nats 30422:4222"
LLM_COMMAND="exec kubectl -n default port-forward --address 10.112.221.121 svc/simlingo-llm-service 9012:9012"
SCHEDULER_COMMAND="exec kubectl -n default port-forward --address 10.112.221.121 svc/simlingo-scheduler-service 9013:9013"

usage() {
    printf 'Usage: %s --on|--off\n' "${0##*/}"
    printf '  --on   Start the NATS, LLM, and Scheduler port-forwards in tmux.\n'
    printf '  --off  Stop the three tmux port-forward tasks if running.\n'
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

if ! command -v kubectl >/dev/null 2>&1; then
    printf 'Error: kubectl is not installed or is not in PATH.\n' >&2
    exit 1
fi

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    --on)
        start_tmux_session "$NATS_SESSION" "$NATS_COMMAND"
        start_tmux_session "$LLM_SESSION" "$LLM_COMMAND"
        start_tmux_session "$SCHEDULER_SESSION" "$SCHEDULER_COMMAND"
        printf 'Attach to a task with:\n'
        printf '  tmux attach -t %s\n' "$NATS_SESSION"
        printf '  tmux attach -t %s\n' "$LLM_SESSION"
        printf '  tmux attach -t %s\n' "$SCHEDULER_SESSION"
        ;;
    --off)
        stop_tmux_session "$NATS_SESSION" "NATS port-forward"
        stop_tmux_session "$LLM_SESSION" "LLM port-forward"
        stop_tmux_session "$SCHEDULER_SESSION" "Scheduler port-forward"
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
