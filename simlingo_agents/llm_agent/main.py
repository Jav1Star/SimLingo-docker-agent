import uvicorn

from utils.logger_utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting SimLingo LLM Agent...")
    uvicorn.run("fast_api.app:app", host="0.0.0.0", port=9012)


if __name__ == "__main__":
    main()
