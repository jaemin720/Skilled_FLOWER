"""Prepare ROS before executable applications import ROS-dependent modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_READY = "_RB_ENV_ROS_READY"
_SETUPS = (
    Path("/opt/ros/humble/setup.bash"),
)
_VENDOR = Path(__file__).resolve().parents[1] / "rb_env" / "_vendor" / "cell_msgs"
_DDS_PROFILE = Path(__file__).resolve().parents[1] / "rb_env" / "config" / "fastdds_shm_only.xml"


def configure_plaif_dds(environment: dict[str, str]) -> None:
    """Match the controller's local SHM transport unless explicitly configured."""
    environment.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    if environment["RMW_IMPLEMENTATION"] != "rmw_fastrtps_cpp":
        return
    if any(environment.get(key) for key in (
        "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE",
    )):
        return
    environment["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(_DDS_PROFILE)
    # Humble's localhost-only participant settings prevent matching the
    # controller's SHM-only participant. The XML itself disables ALL UDP;
    # setting this to 0 does not enable network transport in that profile.
    environment["ROS_LOCALHOST_ONLY"] = "0"


def ensure_ros_environment() -> None:
    """Load the installed overlays and restart with the same Python/arguments.

    Call only from an executable entry point, before importing ROS. Updating
    sys.path alone cannot configure the native loader for rclpy/cell_msgs;
    LD_LIBRARY_PATH must be present when the Python process starts.
    """
    marker = json.dumps([sys.executable, *map(str, _SETUPS), str(_VENDOR), str(_DDS_PROFILE)])
    if os.environ.get(_READY) == marker:
        return
    for setup in _SETUPS:
        if not setup.is_file():
            raise RuntimeError(f"Required ROS setup not found: {setup}")

    # Pass paths as arguments, never interpolate them into shell source.
    result = subprocess.run(
        [
            "/bin/bash", "--noprofile", "--norc", "-c",
            'source "$1" >&2 || exit; '
            'exec "$2" -c \'import json, os; print(json.dumps(dict(os.environ)))\'',
            "rb-env-ros", *map(str, _SETUPS), sys.executable,
        ],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    environment = json.loads(result.stdout)
    configure_plaif_dds(environment)
    for key, path in (
        ("PYTHONPATH", _VENDOR / "python"),
        ("LD_LIBRARY_PATH", _VENDOR / "lib"),
        ("AMENT_PREFIX_PATH", _VENDOR),
    ):
        previous = environment.get(key, "")
        environment[key] = str(path) + (os.pathsep + previous if previous else "")
    environment[_READY] = marker
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
