# core/paths.py

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_STATE_DIR_ENV = "SMART_MIRROR_STATE_DIR"
APP_STATE_DIR_NAME = "smart-gym-client"


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def resource_path(*parts):
    return str(PROJECT_ROOT.joinpath("resources", *parts))


def _default_state_root():
    override = str(os.getenv(APP_STATE_DIR_ENV) or "").strip()
    if override:
        return Path(override).expanduser()

    home = Path.home()
    if sys.platform == "win32":
        appdata_root = os.getenv("APPDATA")
        base = Path(appdata_root).expanduser() if appdata_root else home / "AppData" / "Roaming"
        return base / APP_STATE_DIR_NAME
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_STATE_DIR_NAME

    xdg_data_home = str(os.getenv("XDG_DATA_HOME") or "").strip()
    base = Path(xdg_data_home).expanduser() if xdg_data_home else home / ".local" / "share"
    return base / APP_STATE_DIR_NAME


def state_root():
    root = _default_state_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_path(*parts):
    path = state_root().joinpath(*parts)
    if parts:
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
