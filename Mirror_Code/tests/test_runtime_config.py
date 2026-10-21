import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui.main_window import MainWindow


class RuntimeConfigurationTests(unittest.TestCase):
    def test_legacy_migration_requires_fresh_server_provisioning(self):
        legacy_data = {
            "server_sync": {
                "enabled": True,
                "server_url": "https://old-install.example",
                "api_key": "old-device-secret",
                "gym_id": "old-gym",
                "main_system_id": "old-main",
            },
            "exercises": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(legacy_data), encoding="utf-8")
            context = SimpleNamespace(
                _read_json_file=MainWindow._read_json_file,
                aruco_dict_type="DICT_5X5_100",
                CONFIG_FILE=str(Path(directory) / "runtime.json"),
            )
            with patch.dict(os.environ, {"SMART_GYM_MAIN_API_KEY": ""}):
                migrated = MainWindow._build_runtime_config_from_legacy(context, str(path))
        self.assertFalse(migrated["server_sync"]["enabled"])
        self.assertEqual(migrated["server_sync"]["server_url"], "http://127.0.0.1:8000")
        self.assertEqual(migrated["server_sync"]["api_key"], "")
        self.assertEqual(migrated["server_sync"]["gym_id"], "")

    def test_string_false_remains_disabled_when_system_ids_are_ensured(self):
        context = SimpleNamespace(
            system_config={},
            server_sync_config={"enabled": "false"},
            _parse_env_bool=MainWindow._parse_env_bool,
        )
        MainWindow._ensure_system_ids(context)
        self.assertFalse(context.server_sync_config["enabled"])


if __name__ == "__main__":
    unittest.main()
