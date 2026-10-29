"""Shared environment builder for the Real_Robo application modules.

Camera and gripper backends are selected entirely from the environment
configuration. All current demo and bridge entry points use this builder;
the older per-entry-point builder modules are no longer required.

Camera construction is delegated to `rb_env.camera.build_camera()`, which
looks up the backend by the
`cameras.<name>.type` string in a registry (see rb_env/camera/__init__.py --
the same factory-registry pattern rb_env.gripper_drivers uses
for gripper backends, see below). That registry already includes
"ros2_image_topic" (rb_env/camera/ros2_topic_capture.py), which subscribes
to a PIKA-published ROS2 image topic instead of opening a /dev/video* or
RealSense device -- so a PIKA camera stream can be added to an env's
observations purely by editing the JSON config, e.g.:

    "left_wrist_cam": {
      "type": "ros2_image_topic",
      "params": {"topic": "/camera_l/color/image_raw", "dim": [640, 480]}
    }

(confirm the exact topic name for your bringup with `ros2 topic list` --
see pika_image_viewer.py's module docstring for how single- vs
multi-gripper launches differ). No changes to this file, demo.py,
or any other code are needed to add/swap a camera backend this way.

Gripper command backend (USB DH driver vs ROS2-topic PIKA driver) is
likewise already fully config-driven, one layer down in the stack: it is
NOT something this file (or demo.py) needs to switch on, because
`GripperRobotWrapper` creates the driver for each side from
`config["robot"]["gripper"]["left"/"right"]["type"]` ("dhgripper",
"pikagripper", or "dummygripper") via
`rb_env.gripper_drivers.create_gripper_driver`.

Policy rollout action interpretation is configured here and executed by
`ChunkInterpreter` in the RPC client:

    "train": {
      "action_chunk_mode": "relative_tcp",  # "delta", "delta_flange", "relative_tcp"
      "policy_arm": "left"                  # optional; default = "right"
    }

The optional Action Adapter is configured here too and executed by
`ActionAdapterWrapper` (see rb_env/envs/action_adapter.py for the model and
its calibration CLI):

    "action_adapter": {
      "enable": true,
      "weights_path": "~/weights/action_adapter/default.npz",
      "sides": ["left", "right"],
      "online_adaptation": {"mu": 0.01},
      "calibration": {"trajectories": 10, "horizon": 50}
    }

It maps a policy's world-frame state delta to the command that produces it,
so it only applies to `action_chunk_mode: "delta"`; enabling it alongside
"relative_tcp" is an error rather than a silent no-op.
"""
import json
from collections import OrderedDict
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import rclpy

from rb_env.camera import build_camera
from rb_env.envs.action_adapter import (
    CalibrationSpec,
    MultiSideActionAdapter,
    OnlineAdaptationSpec,
)
from rb_env.envs.chunk_interpreter import DELTA, RELATIVE_TCP
from rb_env.envs.chunking import ChunkingWrapper
from rb_env.envs.dual_rb3_env import DualRb3Env
from rb_env.envs.gripper_wrapper import GripperRobotWrapper
from rb_env.envs.keyboard_wrapper import KeyboardTaskWrapper, resolve_keyboard_config
# Compatibility exports for old collection tools; unsupported and pending removal.
from rb_env.envs.legacy_image_preprocess import (
    adjust_gamma, apply_clahe, build_image_crop_from_config, center_crop,
    legacy_image_kwargs,
)
from rb_env.envs.rb3_env import EnvConfig, Rb3Env
from rb_env.envs.serl_obs_wrappers import FlattenObservationWrapper
from rb_env.envs.wrappers import (
    ActionAdapterWrapper,
    DualSpacemouseIntervention2,
    validate_spacemouse_symlog_temperature,
    DualWrenchFrameWrapper,
    ImageDisplayer,
    SidePrefixedQuat2EulerWrapper,
)


def build_env_config(side_cfg: Dict[str, Any]) -> EnvConfig:
    """Convert one side of the serialized config into an environment config."""
    reset_pose_max = side_cfg.get("reset_pose_max")
    return EnvConfig(
        ROBOT_PREFIX=side_cfg["robot_prefix"],
        RESET_POSE=np.asarray(side_cfg["reset_pose"], dtype=float),
        RESET_POSE_MAX=(
            np.asarray(reset_pose_max, dtype=float)
            if reset_pose_max is not None
            else None
        ),
        ABS_POSE_LIMIT_LOW=np.asarray(side_cfg["xyz_clip_low"], dtype=float),
        ABS_POSE_LIMIT_HIGH=np.asarray(side_cfg["xyz_clip_high"], dtype=float),
        MAX_EPISODE_LENGTH=int(side_cfg["max_episode_length"]),
        DISPLAY_IMAGE=bool(side_cfg["display_image"]),
        GRIPPER_CONTROL=dict(side_cfg.get("gripper_control", {})),
        MAX_TRANSLATION_SPEED_MPS=side_cfg.get("max_translation_speed_mps", 0.05),
        MAX_ROTATION_SPEED_RADPS=side_cfg.get("max_rotation_speed_radps", 0.5),
    )


def load_config(json_path: str | Path) -> Dict[str, Any]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    # Lets `action_adapter.weights_path` be written relative to the config.
    config["_source_path"] = str(json_path.resolve())
    period = float(config.get("train", {}).get("policy_action_period_s", 0.1))
    if not np.isfinite(period) or period <= 0:
        raise ValueError("train.policy_action_period_s must be finite and positive")
    config["train"]["policy_action_period_s"] = period
    return config


def resolve_mjcf_path(robot_cfg: Dict[str, Any]) -> str:
    """Resolve either an explicit absolute path or a packaged model name."""
    model_name = robot_cfg.get("mjcf_model")
    if model_name:
        model = files(
            "external.dual_rb3_mujoco.dual_rb3_mujoco"
        ).joinpath("models", model_name)
        if not model.is_file():
            raise FileNotFoundError(f"Unknown dual_rb3_mujoco model: {model_name}")
        return str(model)

    configured_path = robot_cfg.get("mjcf_path", "")
    if not configured_path:
        return ""

    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "robot.mjcf_path must be absolute. For bundled models, use "
            "robot.mjcf_model with a filename from dual_rb3_mujoco/models."
        )
    return str(path)


SIDES = ("left", "right")


def resolve_policy_sides(config: Dict[str, Any]) -> tuple[str, ...]:
    """Which arm(s) a policy's action chunk describes.

    `train.policy_arm` wins when set ("left", "right" or "dual").  Otherwise
    use "right", matching the original `env_rpc_client.py` behavior.
    """
    declared = (config.get("train") or {}).get("policy_arm")
    if declared:
        declared = str(declared).lower()
        if declared == "dual":
            return SIDES
        if declared in SIDES:
            return (declared,)
        raise ValueError(
            f"train.policy_arm must be 'left', 'right' or 'dual', got {declared!r}"
        )
    return ("right",)


def resolve_action_chunk_mode(config: Dict[str, Any]) -> str:
    """Return the configured policy action chunk representation."""
    train_cfg = config.get("train") or {}
    mode = train_cfg.get("action_chunk_mode", "delta")
    if mode not in ("delta", "delta_flange", RELATIVE_TCP):
        raise ValueError(
            f"train.action_chunk_mode must be 'delta', 'delta_flange' or {RELATIVE_TCP!r}, "
            f"got {mode!r}"
        )
    return mode


ACTION_ADAPTER_KEYS = {
    "enable",
    "weights_path",
    "sides",
    "online_adaptation",
    "calibration",
}


def resolve_action_adapter_config(
    config: Dict[str, Any], require_enabled: bool = True
) -> Dict[str, Any]:
    """Normalize and validate the optional ``action_adapter`` config block.

    Returns a dict with ``enable``, an absolute ``weights_path`` (or ``None``),
    the ``sides`` to adapt, an :class:`OnlineAdaptationSpec`, and the raw
    ``calibration`` block for the calibration CLI.

    ``require_enabled=False`` keeps the block readable (that is what the
    calibration CLI needs) without enforcing the constraints that only matter
    for an environment that will actually run the wrapper.
    """
    adapter_cfg = dict(config.get("action_adapter") or {})
    unknown = set(adapter_cfg) - ACTION_ADAPTER_KEYS
    if unknown:
        raise ValueError(
            f"Unknown action_adapter key(s): {sorted(unknown)}; "
            f"expected {sorted(ACTION_ADAPTER_KEYS)}"
        )

    enable = bool(adapter_cfg.get("enable", False))
    sides_declared = adapter_cfg.get("sides")
    sides = tuple(sides_declared) if sides_declared else SIDES
    invalid = [side for side in sides if side not in SIDES]
    if invalid:
        raise ValueError(
            f"action_adapter.sides must be drawn from {list(SIDES)}, got {invalid}"
        )

    online = OnlineAdaptationSpec.from_config(
        adapter_cfg.get("online_adaptation")
    )
    calibration = CalibrationSpec.from_config(adapter_cfg.get("calibration"))

    weights_path = adapter_cfg.get("weights_path")
    if weights_path:
        path = Path(str(weights_path)).expanduser()
        if not path.is_absolute():
            source = config.get("_source_path")
            base = Path(source).parent if source else Path.cwd()
            path = (base / path).resolve()
        weights_path = path
    else:
        weights_path = None

    if enable and require_enabled:
        # The adapter rewrites world-frame *deltas*; a relative_tcp chunk
        # carries absolute target poses instead, and is passed through
        # untouched.  Fail loudly rather than let an enabled adapter do
        # nothing at all.
        chunk_mode = resolve_action_chunk_mode(config)
        if chunk_mode != DELTA:
            raise ValueError(
                "action_adapter.enable requires train.action_chunk_mode "
                f"{DELTA!r}, got {chunk_mode!r}."
            )
        if weights_path is None:
            raise ValueError(
                "action_adapter.enable requires action_adapter.weights_path."
            )
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"action_adapter.weights_path does not exist: {weights_path}. "
                "Run `python -m rb_env.envs.action_adapter` to calibrate."
            )

    return {
        "enable": enable,
        "weights_path": weights_path,
        "sides": sides,
        "sides_declared": bool(sides_declared),
        "online": online,
        "calibration": calibration,
    }


def build_action_adapter(
    adapter_cfg: Dict[str, Any], sides: tuple[str, ...]
) -> MultiSideActionAdapter:
    """Load one calibrated linear map per arm."""
    return MultiSideActionAdapter.load(
        adapter_cfg["weights_path"],
        sides=sides,
        online=adapter_cfg["online"],
    )


def resolve_action_adapter_sides(
    adapter_cfg: Dict[str, Any], active_sides: tuple[str, ...]
) -> tuple[str, ...]:
    """Which arms the wrapper adapts, given the arms this env actually has."""
    if adapter_cfg["sides_declared"]:
        missing = [side for side in adapter_cfg["sides"] if side not in active_sides]
        if missing:
            raise ValueError(
                f"action_adapter.sides {missing} are not active in this "
                f"environment (active: {list(active_sides)})"
            )
        return tuple(adapter_cfg["sides"])
    return tuple(side for side in active_sides if side in adapter_cfg["sides"])


def resolve_send_chunk_length(config: Dict[str, Any]) -> int:
    """Length on the follow wire; never an environment execution horizon."""
    robot = config.get("robot", {})
    length = (robot.get("plaif") or {}).get("send_chunk_length", 1)
    if type(length) is not int or not 1 <= length <= 50:
        raise ValueError("robot.plaif.send_chunk_length must be an integer in [1, 50]")
    if length > 1 and str(robot.get("backend", "rbpodo")).strip().lower() != "plaif":
        raise ValueError("send_chunk_length > 1 requires robot.backend='plaif'")
    if length > 1 and (config.get("action_adapter") or {}).get("enable", False):
        raise ValueError("ActionAdapter does not support action chunks; disable it for chunk sending")
    return length


def build_environment(
    config: Dict[str, Any],
    fake_env: bool = False,
    save_video: bool = False,
    reward_fn: Callable[[dict], float] | None = None,
):
    resolve_send_chunk_length(config)
    keyboard_cfg = resolve_keyboard_config(config)
    image_kwargs = legacy_image_kwargs(config)
    arm_type = config["train"]["arm_type"]
    if arm_type not in ("dual", *SIDES):
        raise ValueError(f"Unsupported arm_type {arm_type}")
    active_sides = SIDES if arm_type == "dual" else (arm_type,)
    spacemouse_symlog_temperatures = {
        side: validate_spacemouse_symlog_temperature(
            config["env"][side].get("spacemouse_symlog_temperature", 10.0)
        )
        for side in active_sides
    }
    robot_cfg = config["robot"]
    backend = str(robot_cfg.get("backend", "rbpodo")).strip().lower()
    if not rclpy.ok():
        if backend == "plaif":
            from rclpy.signals import SignalHandlerOptions

            # Python's KeyboardInterrupt must reach env.close() while ROS is
            # still alive, so the backend can send and confirm task off.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        else:
            rclpy.init()

    cameras = OrderedDict()
    for name, cam_cfg in config["cameras"].items():
        cam = build_camera(name, cam_cfg)
        cam.start()
        cameras[name] = cam

    if backend == "plaif":
        # Import lazily so the legacy backend need not load bundled cell_msgs.
        from rb_env.envs.plaif_node import (
            PlaifNode,
        )

        plaif_cfg = robot_cfg.get("plaif") or {}
        follow_period = float(plaif_cfg.get("follow_period", 0.1))
        env_period = 1.0 / float(config["train"]["hz"])
        if not np.isclose(env_period, follow_period, atol=1e-3, rtol=0.0):
            raise ValueError(
                "Plaif follow cadence must match train.hz: "
                f"1 / {config['train']['hz']} = {env_period:.6f}s, "
                f"robot.plaif.follow_period = {follow_period:.6f}s"
            )
        arm_node = PlaifNode(
            platform=plaif_cfg.get("platform", "monkey"),
            task=plaif_cfg.get("task", "sku"),
            wait_for_data=bool(plaif_cfg.get("wait_for_data", True)),
            state_timeout=float(plaif_cfg.get("state_timeout", 10.0)),
            velocity_filter_alpha=float(
                plaif_cfg.get("velocity_filter_alpha", 0.25)
            ),
            follow_period=follow_period,
            active_sides=active_sides,
            control_hz=float(config["train"]["hz"]),
        )
    elif backend == "rbpodo":
        from external.dual_rb3_mujoco_ros2.dual_rb3_mujoco_ros2.dual_rb3_mujoco_node import DualRb3MujocoNode

        arm_node = DualRb3MujocoNode(
            resolve_mjcf_path(robot_cfg), is_sim=robot_cfg["is_sim"]
        )
    else:
        raise ValueError(
            f"robot.backend must be 'rbpodo' or 'plaif', got {backend!r}"
        )
    gripper_cfg = robot_cfg.get("gripper") or {}
    node = GripperRobotWrapper(
        arm_node, {side: gripper_cfg[side] for side in active_sides if side in gripper_cfg}
    )

    chunk_mode = resolve_action_chunk_mode(config)
    arm_envs = {
        side: Rb3Env(
            robot_node=node, fake_env=fake_env,
            config=build_env_config(config["env"][side]), hz=config["train"]["hz"],
        )
        for side in active_sides
    }
    # Snapshot the effective driver defaults/calibration, not only JSON overrides.
    from rb_env.gripper_interface import GripperInterface
    interfaces = {side: arm.gripper_interface.contract() for side, arm in arm_envs.items()
                  if isinstance(getattr(arm, 'gripper_interface', None), GripperInterface)}
    if interfaces:
        config['gripper_interfaces'] = interfaces

    env = DualRb3Env(
        arm_envs.get("left"), arm_envs.get("right"),
        cameras=cameras, save_video=save_video, **image_kwargs,
    )

    # PlaifNode consumes the controller's compensated wrench-at-TCP surface
    # and already rotates it from tool axes into the cell base.  The legacy
    # node exposes an FT-frame wrench and still needs the URDF/TF conversion.
    if getattr(arm_node, "wrench_observation_frame", None) != "base_at_tcp":
        env = DualWrenchFrameWrapper(
            env,
            tf_frame_prefix=config["robot"].get("tf_frame_prefix", "dual_rb3"),
        )
    env = ImageDisplayer(env)
    spacemouse_action_scales = {
        side: np.asarray(
            config["env"][side]["spacemouse_action_scale"], dtype=float
        )
        for side in active_sides
    }
    env = DualSpacemouseIntervention2(
        env,
        arm_type=arm_type,
        spacemouse_action_scales=spacemouse_action_scales,
        spacemouse_symlog_temperatures=spacemouse_symlog_temperatures,
    )
    # Outside the spacemouse wrapper on purpose: policy deltas are adapted,
    # human teleoperation commands reach the robot as they are.
    adapter_cfg = resolve_action_adapter_config(config)
    if adapter_cfg["enable"]:
        adapter_sides = resolve_action_adapter_sides(adapter_cfg, active_sides)
        env = ActionAdapterWrapper(
            env,
            build_action_adapter(adapter_cfg, adapter_sides),
            sides=adapter_sides,
            chunk_mode=chunk_mode,
        )
        print(
            f"[Env Builder] action adapter enabled for {adapter_sides} "
            f"from {adapter_cfg['weights_path']}"
        )
    env = SidePrefixedQuat2EulerWrapper(env)
    env = FlattenObservationWrapper(
        env, proprio_keys=config["train"]["proprio_keys"]
    )
    env = ChunkingWrapper(
        env,
        obs_horizon=config["train"]["wrappers"]["serlobs"]["obs_horizon"],
    )

    if keyboard_cfg["enable"]:
        # Record the same observations/actions that the application sees.
        env = KeyboardTaskWrapper(env, keyboard_cfg, collection_config=config)

    return env
