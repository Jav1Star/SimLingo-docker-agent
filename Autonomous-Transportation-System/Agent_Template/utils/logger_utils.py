"""
日志工具模块

提供统一的日志配置和日志获取函数，确保整个项目使用一致的日志格式和命名空间。

日志命名空间:
  AgentTemplate.<module_name>

  例如:
    AgentTemplate.main          — 主入口模块
    AgentTemplate.fast_api.app  — FastAPI 应用模块
    AgentTemplate.utils.numpy_utils — numpy 工具模块

使用方式:
    from utils.logger_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Hello from %s", __name__)
"""
import logging

# 应用根日志名称，所有子日志器都以此作为前缀
APP_LOGGER_NAME = "AgentTemplate"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _resolve_log_level(level: str | int | None = None) -> int:
    """
    将日志级别字符串或整数解析为 Python logging 模块认可的数值。

    参数:
        level: 日志级别，可以是:
               - 字符串（如 "INFO", "DEBUG", "WARNING"）
               - 整数（如 logging.INFO = 20）
               - None（使用默认级别 INFO）

    返回:
        对应的 logging 级别数值（int）

    示例:
        _resolve_log_level("DEBUG")  → 10
        _resolve_log_level(20)       → 20
        _resolve_log_level(None)     → 20 (INFO)
    """
    if isinstance(level, int):
        return level

    raw_level = level or DEFAULT_LOG_LEVEL
    numeric_level = logging.getLevelName(str(raw_level).upper())
    if isinstance(numeric_level, int):
        return numeric_level
    return logging.INFO


def setup_logging(level: str | int | None = None) -> logging.Logger:
    """
    配置并返回应用根日志器（只配置一次，多次调用不会重复添加 handler）。

    配置内容:
      - 设置日志级别
      - 添加 StreamHandler 输出到标准输出
      - 设置日志格式为: 时间 - 日志器名 - 级别 - 消息
      - 关闭 propagate，避免日志冒泡到根日志器导致重复输出

    参数:
        level: 日志级别，默认 INFO

    返回:
        配置好的应用根 Logger 实例
    """
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(_resolve_log_level(level))

    # 只在首次调用时添加 handler，避免重复
    if not app_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(app_logger.level)
        stream_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        app_logger.addHandler(stream_handler)

    # 防止日志冒泡到 uvicorn/mcp 等根日志器导致重复输出
    app_logger.propagate = False
    return app_logger


def get_logger(module_name: str | None = None) -> logging.Logger:
    """
    获取统一命名空间下的子日志器。

    所有日志器都以 "AgentTemplate" 为前缀，确保日志输出风格一致。

    参数:
        module_name: 模块名称，通常传入 __name__。
                     如果为 None 或 "__main__"，则返回主入口日志器。

    返回:
        命名格式为 AgentTemplate.<module_name> 的 Logger 实例

    示例:
        # 在 main.py 中:
        logger = get_logger(__name__)     # → AgentTemplate.main

        # 在 fast_api/app.py 中:
        logger = get_logger(__name__)     # → AgentTemplate.fast_api.app

        # 在 utils/numpy_utils.py 中:
        logger = get_logger(__name__)     # → AgentTemplate.utils.numpy_utils
    """
    setup_logging()

    if not module_name or module_name == "__main__":
        return logging.getLogger(f"{APP_LOGGER_NAME}.main")

    # 将路径分隔符 "/" 替换为 "."，确保包路径格式正确
    normalized = module_name.replace("/", ".")
    return logging.getLogger(f"{APP_LOGGER_NAME}.{normalized}")
