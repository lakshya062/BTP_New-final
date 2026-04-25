# core/logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

from .paths import state_path


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        return

    level_name = str(os.getenv("SMART_MIRROR_LOG_LEVEL", "INFO")).upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler = RotatingFileHandler(
        state_path("logs", "app.log"),
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
