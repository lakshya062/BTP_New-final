# core/logging_config.py

import logging
from logging.handlers import RotatingFileHandler

from .paths import project_path


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler = RotatingFileHandler(
        project_path("app.log"),
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
