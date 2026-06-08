import logging
import os

APP_LOGGER_NAME = "Perception2IntermediateFeature_Agent_demo"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _resolve_log_level(level: str | int | None = None) -> int:
    if isinstance(level, int):
        return level

    raw_level = level or os.getenv("PERCEPTION2INTERMEDIATEFEATURE_AGENT_DEMO_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    numeric_level = logging.getLevelName(str(raw_level).upper())
    if isinstance(numeric_level, int):
        return numeric_level
    return logging.INFO


def setup_logging(level: str | int | None = None) -> logging.Logger:
    """Configure and return the application root logger exactly once."""
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(_resolve_log_level(level))

    if not app_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(app_logger.level)
        stream_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        app_logger.addHandler(stream_handler)

    # Prevent duplicate logs from bubbling up to root handlers (e.g. uvicorn/mcp).
    app_logger.propagate = False
    return app_logger


def get_logger(module_name: str | None = None) -> logging.Logger:
    """Return a logger under the unified application namespace."""
    setup_logging()

    if not module_name or module_name == "__main__":
        return logging.getLogger(f"{APP_LOGGER_NAME}.main")

    normalized = module_name.replace("/", ".")
    return logging.getLogger(f"{APP_LOGGER_NAME}.{normalized}")