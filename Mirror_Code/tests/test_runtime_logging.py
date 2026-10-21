import logging
import os
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from unittest.mock import patch

from core.logging_config import configure_logging


class RuntimeLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.original_handlers = self.root.handlers[:]
        self.original_level = self.root.level
        self.root.handlers = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"SMART_MIRROR_STATE_DIR": self.temp_dir.name})
        self.env.start()

    def tearDown(self):
        for handler in self.root.handlers:
            handler.close()
        self.root.handlers = self.original_handlers
        self.root.setLevel(self.original_level)
        self.env.stop()
        self.temp_dir.cleanup()

    def test_existing_third_party_handler_does_not_disable_application_log(self):
        existing = logging.NullHandler()
        self.root.addHandler(existing)
        configure_logging()
        self.assertIn(existing, self.root.handlers)
        file_handlers = [h for h in self.root.handlers if isinstance(h, RotatingFileHandler)]
        self.assertEqual(len(file_handlers), 1)
        self.root.info("Coaching feedback: élévation")
        file_handlers[0].flush()
        with open(file_handlers[0].baseFilename, encoding="utf-8") as log_file:
            self.assertIn("élévation", log_file.read())

    def test_repeated_configuration_does_not_duplicate_handlers(self):
        configure_logging()
        first_handlers = self.root.handlers[:]
        configure_logging()
        self.assertEqual(self.root.handlers, first_handlers)

    def test_unwritable_state_directory_falls_back_to_console(self):
        with patch("core.logging_config.state_path", side_effect=PermissionError("read only")):
            configure_logging()
        self.assertEqual(len(self.root.handlers), 1)
        self.assertIsInstance(self.root.handlers[0], logging.StreamHandler)

    def test_invalid_logging_attribute_uses_info(self):
        with patch.dict(os.environ, {"SMART_MIRROR_LOG_LEVEL": "getLogger"}):
            configure_logging()
        self.assertEqual(self.root.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
