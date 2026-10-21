# core/logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

from .paths import state_path


def configure_logging():
    root = logging.getLogger()
    level_name = str(os.getenv("SMART_MIRROR_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level if isinstance(level, int) else logging.INFO)
    # Third-party imports (or a host application) may already have handlers.
    # Only our own marker should prevent duplicate application handlers.
    if any(getattr(handler, "_smart_mirror_handler", False) for handler in root.handlers):
        return
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler._smart_mirror_handler = True
    root.addHandler(stream_handler)

    try:
        file_handler = RotatingFileHandler(
            state_path("logs", "app.log"),
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError:
        root.warning("Could not open the application log; using console logging.", exc_info=True)
    else:
        file_handler.setFormatter(formatter)
        file_handler._smart_mirror_handler = True
        root.addHandler(file_handler)
