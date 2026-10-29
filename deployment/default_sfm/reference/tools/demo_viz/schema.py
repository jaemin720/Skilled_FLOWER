"""Decode what demo.py pickles back into named channels.

The demo loop stores whatever the wrapper stack produced, which is a flat
float vector per step plus raw camera frames.  Recovering the meaning of
each column needs the environment config that was used at collection time,
so every helper here takes that config (or the pieces of it that matter).

Layout, for the record.  ``FlattenObservationWrapper`` builds a
``gym.spaces.Dict`` from ``train.proprio_keys`` and flattens it; gymnasium
sorts Dict keys, so the column order is ``sorted(proprio_keys)`` regardless
of how the config lists them.  Each key contributes a fixed width:

    gripper_pose 1 | tcp_force 3 | tcp_pose 6 | tcp_torque 3 | tcp_vel 6

``tcp_pose`` is 6 and not 7 because ``SidePrefixedQuat2EulerWrapper`` has
already turned the quaternion into xyz Euler angles.  With the default dual
config that is 19 columns per arm, 38 in total.

Actions are 7 per arm -- 3 translation deltas, 3 rotation deltas, 1 gripper
command -- concatenated left before right, matching ``DualRb3Env``'s
``_active_arms`` order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

SIDES = ("left", "right")

#: Width of every proprio key the env can expose, after the wrapper stack.
CHANNEL_WIDTHS: Dict[str, int] = {
    "gripper_pose": 1,
    "tcp_force": 3,
    "tcp_pose": 6,
    "tcp_torque": 3,
    "tcp_vel": 6,
}

#: Component labels, used verbatim by the UI.
CHANNEL_COMPONENTS: Dict[str, Sequence[str]] = {
    "gripper_pose": ("pos",),
    "tcp_force": ("fx", "fy", "fz"),
    "tcp_pose": ("x", "y", "z", "roll", "pitch", "yaw"),
    "tcp_torque": ("tx", "ty", "tz"),
    "tcp_vel": ("vx", "vy", "vz", "wx", "wy", "wz"),
}

#: One arm's action block.
ACTION_COMPONENTS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip")
ACTION_BLOCK_DIM = len(ACTION_COMPONENTS)


class SchemaError(RuntimeError):
    """The pickle does not match the config it was supposedly collected with."""


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def active_sides(config: Dict[str, Any]) -> tuple[str, ...]:
    arm_type = (config.get("train") or {}).get("arm_type", "dual")
    return SIDES if arm_type == "dual" else (arm_type,)


def state_layout(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one entry per proprio key: its side, name, slice and labels.

    Sorted exactly the way gymnasium flattens the Dict space, so ``offset``
    indexes straight into ``observations["state"]``.
    """
    proprio_keys = (config.get("train") or {}).get("proprio_keys") or []
    layout: List[Dict[str, Any]] = []
    offset = 0
    for key in sorted(proprio_keys):
        side, _, name = key.partition("/")
        width = CHANNEL_WIDTHS.get(name)
        if width is None:
            raise SchemaError(
                f"proprio key {key!r} has no known width; "
                f"known channels: {sorted(CHANNEL_WIDTHS)}"
            )
        layout.append(
            {
                "key": key,
                "side": side,
                "channel": name,
                "offset": offset,
                "width": width,
                "components": list(CHANNEL_COMPONENTS[name]),
            }
        )
        offset += width
    return layout


def state_width(layout: Sequence[Dict[str, Any]]) -> int:
    return sum(entry["width"] for entry in layout)


def action_layout(sides: Sequence[str]) -> List[Dict[str, Any]]:
    """Return one entry per arm: where its 7-wide action block starts."""
    return [
        {
            "side": side,
            "offset": index * ACTION_BLOCK_DIM,
            "width": ACTION_BLOCK_DIM,
            "components": list(ACTION_COMPONENTS),
        }
        for index, side in enumerate(sides)
    ]


def clip_bounds(config: Dict[str, Any], side: str) -> Dict[str, List[float]] | None:
    """The safety box for one arm, as the UI draws it around the trajectory."""
    env_cfg = (config.get("env") or {}).get(side) or {}
    low, high = env_cfg.get("xyz_clip_low"), env_cfg.get("xyz_clip_high")
    if not low or not high:
        return None
    return {"low": [float(v) for v in low[:3]], "high": [float(v) for v in high[:3]]}


def reset_pose(config: Dict[str, Any], side: str) -> List[float] | None:
    pose = ((config.get("env") or {}).get(side) or {}).get("reset_pose")
    return [float(v) for v in pose] if pose else None


def control_hz(config: Dict[str, Any]) -> float:
    return float((config.get("train") or {}).get("hz", 10))


def describe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the UI needs to interpret the arrays, in one JSON blob."""
    sides = active_sides(config)
    layout = state_layout(config)
    return {
        "sides": list(sides),
        "hz": control_hz(config),
        "state_layout": layout,
        "state_width": state_width(layout),
        "action_layout": action_layout(sides),
        "action_width": len(sides) * ACTION_BLOCK_DIM,
        "clip_bounds": {
            side: clip_bounds(config, side)
            for side in sides
            if clip_bounds(config, side)
        },
        "reset_pose": {
            side: reset_pose(config, side)
            for side in sides
            if reset_pose(config, side)
        },
    }
