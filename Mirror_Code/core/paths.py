# core/paths.py

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def resource_path(*parts):
    return str(PROJECT_ROOT.joinpath("resources", *parts))
