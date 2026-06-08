import uvicorn

from utils.logger_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting SimLingo Scheduler Agent...")
    uvicorn.run("fast_api.app:app", host="0.0.0.0", port=9013)


if __name__ == "__main__":
    main()
