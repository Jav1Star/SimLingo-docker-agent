"""
Agent Template 项目入口文件

职责:
  - 初始化应用日志
  - 使用 uvicorn 启动 FastAPI 服务，监听 0.0.0.0:9001
  - 捕获未预期的异常并记录日志

开发者可在 main() 中添加额外的启动逻辑（如参数解析、环境变量校验等）。
"""
import uvicorn

from utils.logger_utils import get_logger

logger = get_logger(__name__)

def main():
    """应用主入口函数。启动 uvicorn ASGI 服务器，运行 FastAPI 应用。"""
    logger.info("Starting Agent Template...")

    try:
        # 启动 uvicorn 服务器，监听所有网卡的 9001 端口
        # 模块路径格式: "包.模块:应用实例变量名"
        uvicorn.run("fast_api.app:app", host="0.0.0.0", port=9001)
    except Exception:
        # 记录未预期的异常堆栈后重新抛出
        logger.exception("Agent Template exited unexpectedly")
        raise

if __name__ == "__main__":
    main()
