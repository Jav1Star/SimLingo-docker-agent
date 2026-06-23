#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"
PID_FILE="${PID_DIR}/bench2drive_mcp.pid"
LOG_FILE="${LOG_DIR}/bench2drive_mcp.log"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

is_running() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}")"
        if ps -p "${pid}" >/dev/null 2>&1; then
            return 0
        fi
        rm -f "${PID_FILE}"
    fi
    return 1
}

start_service() {
    if is_running; then
        echo "bench2drive_mcp is already running with pid $(cat "${PID_FILE}")"
        return 0
    fi
    (
        cd "${ROOT_DIR}/../.."
        PYTHONPATH="${ROOT_DIR}/../..:${PYTHONPATH:-}" nohup "${PYTHON_BIN}" -m simlingo_agents.bench2drive_mcp_server.main >"${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
    )
    echo "bench2drive_mcp started with pid $(cat "${PID_FILE}")"
    echo "log: ${LOG_FILE}"
}

stop_service() {
    if ! is_running; then
        echo "bench2drive_mcp is not running"
        return 0
    fi
    local pid
    pid="$(cat "${PID_FILE}")"
    kill "${pid}"
    rm -f "${PID_FILE}"
    echo "bench2drive_mcp stopped"
}

status_service() {
    if is_running; then
        echo "bench2drive_mcp is running with pid $(cat "${PID_FILE}")"
    else
        echo "bench2drive_mcp is not running"
    fi
}

case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
