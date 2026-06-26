#!/bin/bash

# 启动脚本
# 支持启动、停止、重启和状态检查四个子系统

export PYTHONPATH="/src:$PYTHONPATH"

# 配置各模块的Python解释器路径和主程序
POINTCLOUD_MCP_PYTHON="/home/t/anaconda3/envs/langmanus/bin/python3"
POINTCLOUD_MCP_MAIN="main.py"

# 配置日志文件路径
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 配置PID文件路径
PID_DIR="pids"
mkdir -p "$PID_DIR"

POINTCLOUD_MCP_PID="$PID_DIR/pointcloud_mcp.pid"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 单服务脚本：只需要一个命令参数
if [ $# -lt 1 ]; then
    echo "用法: $0 [start|stop|restart|status]"
    exit 1
fi

command="$1"

# 服务名及其相关配置（单模块）
SERVICE_NAME="pointcloud_mcp"
SERVICE_PYTHON="$POINTCLOUD_MCP_PYTHON"
SERVICE_MAIN="$POINTCLOUD_MCP_MAIN"
SERVICE_PID="$POINTCLOUD_MCP_PID"

# 启动单个服务
start_service() {
    local service_name="$1"
    local python_path="$2"
    local main_file="$3"
    local pid_file="$4"
    local log_file="$LOG_DIR/${service_name}.log"
    local run_foreground="$5"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null; then
            echo -e "${YELLOW}$service_name 已经在运行中，PID: $pid${NC}"
            return 1
        else
            echo -e "${YELLOW}发现过时的PID文件，正在清理...${NC}"
            rm -f "$pid_file"
        fi
    fi
    
    echo -e "${GREEN}正在启动 $service_name...${NC}"
    
    if [ "$run_foreground" = "true" ]; then
        # 前台运行（用于collaboration模块）
        "$python_path" "$main_file" &
        echo $! > "$pid_file"
        local pid=$(cat "$pid_file")
        
        echo -e "${GREEN}$service_name 已在前台启动，PID: $pid${NC}"
        echo -e "${YELLOW}提示: 使用 Ctrl+C 停止服务，或在另一个终端使用 '$0 stop $service_name'${NC}"
        
        # 等待服务退出并清理PID文件
        # wait "$pid"
        # rm -f "$pid_file"
        # echo -e "${GREEN}$service_name 已停止${NC}"
    else
        # 后台运行（用于其他模块）
        nohup "$python_path" "$main_file" > "$log_file" 2>&1 &
        echo $! > "$pid_file"
        
        # 验证服务是否成功启动
        sleep 1
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null; then
                echo -e "${GREEN}$service_name 已在后台启动，PID: $pid${NC}"
                echo -e "${GREEN}日志文件: $log_file${NC}"
                return 0
            else
                echo -e "${RED}$service_name 启动失败${NC}"
                rm -f "$pid_file"
                return 1
            fi
        else
            echo -e "${RED}$service_name 启动失败，未生成PID文件${NC}"
            return 1
        fi
    fi
}

# 停止单个服务
stop_service() {
    local service_name="$1"
    local pid_file="$2"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}$service_name 未运行${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    if ! ps -p "$pid" > /dev/null; then
        echo -e "${YELLOW}$service_name 未运行（过时的PID文件）${NC}"
        rm -f "$pid_file"
        return 1
    fi
    
    echo -e "${GREEN}正在停止 $service_name (PID: $pid)...${NC}"
    kill "$pid"
    
    # 等待服务停止
    local timeout=10
    while ps -p "$pid" > /dev/null && [ "$timeout" -gt 0 ]; do
        sleep 1
        timeout=$((timeout - 1))
    done
    
    if ps -p "$pid" > /dev/null; then
        echo -e "${RED}无法停止 $service_name，正在强制终止...${NC}"
        kill -9 "$pid"
        sleep 1
    fi
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}PID文件 $pid_file 不存在${NC}"
    else
        rm -f "$pid_file"
    fi
    
    echo -e "${GREEN}$service_name 已停止${NC}"
    return 0
}

# 检查单个服务状态
check_status() {
    local service_name="$1"
    local pid_file="$2"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${RED}$service_name 未运行${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    if ps -p "$pid" > /dev/null; then
        echo -e "${GREEN}$service_name 正在运行中，PID: $pid${NC}"
        return 0
    else
        echo -e "${RED}$service_name 未运行（过时的PID文件）${NC}"
        rm -f "$pid_file"
        return 1
    fi
}

# 根据命令和模块执行相应操作
case "$command" in
    start)
        echo -e "${GREEN}正在启动 $SERVICE_NAME...${NC}"
        start_service "$SERVICE_NAME" "$SERVICE_PYTHON" "$SERVICE_MAIN" "$SERVICE_PID" "false"
        ;;
    
    stop)
        echo -e "${GREEN}正在停止 $SERVICE_NAME...${NC}"
        stop_service "$SERVICE_NAME" "$SERVICE_PID"
        ;;
    
    restart)
        echo -e "${GREEN}正在重启 $SERVICE_NAME...${NC}"
        $0 stop
        sleep 1
        $0 start
        ;;
    
    status)
        check_status "$SERVICE_NAME" "$SERVICE_PID"
        ;;
    
    *)
        echo -e "${RED}错误: 未知命令 '$command'${NC}"
        exit 1
        ;;
esac

exit 0    
