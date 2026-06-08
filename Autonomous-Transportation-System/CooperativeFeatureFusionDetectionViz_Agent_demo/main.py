from utils.logger_utils import get_logger
from flask_app.app import CooperativeFeatureFusionWebApp

logger = get_logger(__name__)

def main():
    logger.info("Starting CooperativeFeatureFusionDetectionViz Agent demo...")

    try:
        socketio, app = CooperativeFeatureFusionWebApp().build()
        socketio.run(app, host='0.0.0.0', port=9002, debug=False, allow_unsafe_werkzeug=True)

    except Exception:
        logger.exception("CooperativeFeatureFusionDetectionViz Agent demo exited unexpectedly")
        raise


if __name__ == '__main__':
    main()