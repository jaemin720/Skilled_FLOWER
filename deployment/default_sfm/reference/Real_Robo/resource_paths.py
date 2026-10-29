"""Paths to source-tree resources used by the local runtime applications."""

from importlib.resources import files
from pathlib import Path


def _resource_path(package: str, *parts: str) -> Path:
    """Return a filesystem path for a resource in this source checkout."""
    return Path(str(files(package).joinpath(*parts)))


def config_dir() -> Path:
    return _resource_path("Real_Robo", "configs")


def config_path(name: str) -> Path:
    return config_dir() / name


def mujoco_model_path(name: str) -> Path:
    return _resource_path(
        "external.dual_rb3_mujoco.dual_rb3_mujoco", "models", name
    )
