"""Robot-side observation/action adaptation, independent of model architecture."""
import numpy as np
from rb_env.envs.builder_integration import resolve_policy_sides, resolve_action_chunk_mode
from rb_env.gripper_drivers import compute_noop_gripper_action

POLICY_BLOCK_DIM = 7

class PolicyIO:
    """Select policy observations and adapt policy actions to the environment."""

    def __init__(
        self,
        config: dict,
        action_dim: int,
        observation_keys=None,
        policy_sides=None,
        preprocessor=None,
        lighting_correction=None,
    ):
        self.config = config
        self.preprocessor = preprocessor
        self.lighting = None
        if lighting_correction is not None:
            from .lighting_correction import PolicyLightingCorrection
            self.lighting = PolicyLightingCorrection(lighting_correction, config, preprocessor)
        if preprocessor and observation_keys is not None:
            raise ValueError("Bundle observation mapping cannot be overridden by --observation_keys")
        self.action_dim = int(action_dim)
        self.policy_sides = tuple(policy_sides or resolve_policy_sides(config))
        self.policy_dim = POLICY_BLOCK_DIM * len(self.policy_sides)
        if self.policy_dim > self.action_dim:
            raise ValueError(
                f"policy sides {self.policy_sides} need {self.policy_dim} action "
                f"dims but the env action space has {self.action_dim}."
            )

        self.observation_keys = (
            None if observation_keys is None else tuple(observation_keys)
        )
        if self.observation_keys is not None:
            if not self.observation_keys:
                raise ValueError("observation_keys must contain at least one key.")
            if len(set(self.observation_keys)) != len(self.observation_keys):
                raise ValueError(
                    f"observation_keys contains duplicates: {self.observation_keys}"
                )
        self.idle_action = self._build_idle_action()

    # -- observations ----------------------------------------------------

    def observation(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            raise TypeError("Observation must be a dict.")
        if self.lighting:
            obs = self.lighting.before_preprocess(obs)
        if self.preprocessor:
            from .protocol import validate_observation
            processed = self.preprocessor.observation(obs, collection_config=self.config)
            if self.lighting:
                processed = self.lighting.after_preprocess(processed)
            return validate_observation(self.preprocessor, processed)
        if self.observation_keys is None:
            return obs

        missing = [key for key in self.observation_keys if key not in obs]
        if missing:
            raise KeyError(
                f"Observation key(s) {missing} not found; available keys: "
                f"{sorted(obs)}"
            )
        return {key: obs[key] for key in self.observation_keys}

    # -- actions ---------------------------------------------------------

    def observation_poses(self, observation):
        """Full recorded pose remains the anchor even if model state omits pose columns."""
        from preprocess.convert import StateDecoder
        from rb_env.envs.action_utils import rpy_to_pose
        decoder = StateDecoder(self.config)
        return {side: rpy_to_pose(decoder.channel(observation, f'{side}/tcp_pose'))
                for side in self.policy_sides}

    def physical_actions(self, actions):
        if self.preprocessor:
            from .protocol import validate_actions
            values = self.preprocessor.denormalize_actions(validate_actions(self.preprocessor, actions))
            values = self.preprocessor.compose_actions(values)
            if not np.all(np.isfinite(values)):
                raise ValueError("Denormalized actions contain NaN or infinity")
            return values
        return self.as_policy_chunk(actions)

    def gripper_context(self, observation, previous_openings=None):
        if not self.preprocessor or not self.preprocessor.uses_opening_gripper:
            return None
        from preprocess.convert import StateDecoder
        from rb_env.gripper_interface import GripperInterface
        decoder = StateDecoder(self.config)
        observed = [float(GripperInterface.from_collection(self.config, side).observed_opening(
                    decoder.channel(observation, side + '/gripper_pose')[0])) for side in self.policy_sides]
        return dict(observed_gripper=observed, previous_openings=previous_openings)

    def execution_actions(self, labels, gripper_context=None):
        if not self.preprocessor or not self.preprocessor.uses_opening_gripper:
            return labels
        from rb_env.gripper_interface import GripperInterface
        targets = self.preprocessor.decode_gripper_actions(labels, **(gripper_context or {}))
        for i, side in enumerate(self.policy_sides):
            interface = GripperInterface.from_collection(self.config, side)
            if interface.control.mode != 'continuous':
                raise ValueError('Canonical gripper policies require execution_config before constructing the environment')
            targets[:, i * 7 + 6] = interface.command_from_opening(targets[:, i * 7 + 6])
        return targets

    def _build_idle_action(self) -> np.ndarray:
        """Env action that moves nothing, used for arms the policy ignores."""
        idle = np.zeros((self.action_dim,), dtype=np.float32)
        env_cfg = self.config.get("env") or {}
        for index, side in enumerate(("left", "right")):
            offset = index * POLICY_BLOCK_DIM
            if side in self.policy_sides or offset + POLICY_BLOCK_DIM > self.action_dim:
                continue
            gripper_control = (env_cfg.get(side) or {}).get("gripper_control")
            idle[offset + 6] = compute_noop_gripper_action(gripper_control)
        return idle

    def as_policy_chunk(self, actions) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim not in (1, 2) or actions.size == 0:
            raise ValueError(f"Expected a nonempty [T,A] or [A] action array, got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy actions contain NaN or infinity")
        if actions.ndim == 1:
            if actions.shape[0] != self.policy_dim:
                raise ValueError(f"Unexpected 1D action shape {actions.shape}")
            return actions[None, :]
        if (
            actions.ndim == 2
            and actions.shape[-1] != self.policy_dim
            and actions.shape[0] == self.policy_dim
        ):
            actions = actions.T
        if actions.shape[-1] != self.policy_dim:
            raise ValueError(
                f"Unexpected action shape {actions.shape}; expected a chunk of "
                f"{self.policy_dim}-dim actions for side(s) {self.policy_sides}."
            )
        return actions

    def to_env_chunk(self, actions) -> np.ndarray:
        """Place a policy chunk into the env's action layout.
        """
        actions = self.as_policy_chunk(actions)
        if actions.shape[-1] == self.action_dim:
            return actions

        chunk = np.repeat(self.idle_action[None], actions.shape[0], axis=0)
        for index, side in enumerate(self.policy_sides):
            source = actions[:, index * POLICY_BLOCK_DIM : (index + 1) * POLICY_BLOCK_DIM]
            offset = 0 if side == "left" else POLICY_BLOCK_DIM
            if self.action_dim == POLICY_BLOCK_DIM:
                offset = 0
            chunk[:, offset : offset + POLICY_BLOCK_DIM] = source
        return chunk


def episode_result(reward, terminated, truncated, info):
    """Describe the previous episode without treating an unlabeled end as failure."""
    event = info.get("keyboard", {}).get("event")
    if event in ("success", "failure", "skip"):
        outcome = event
    elif info.get("succeed", False) or reward > 0:
        outcome = "success"
    else:
        outcome = "unknown"
    return {
        "outcome": outcome,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }




def response_to_chunk(response: dict, rotation_unit: str = "rad") -> np.ndarray:
    if not isinstance(response, dict):
        raise TypeError(f"Response must be dict, got {type(response)!r}")
    if response.get("ok") is False:
        raise RuntimeError(f"Policy server rejected query: {response.get('error', response)}")

    if "actions" not in response:
        raise KeyError(
            f"Missing 'actions' in response. Response keys: {list(response.keys())}"
        )

    actions = response["actions"]
    if isinstance(actions, dict):
        raise TypeError(
            "response['actions'] is still a dict. "
            "Expected a numeric actions array from the NPZ response. "
            f"actions keys={list(actions.keys())}"
        )

    # Physical pose offsets are not normalized [-1,1] actions. Clipping a
    # rotation vector here changes both its angle and its axis.
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim not in (1, 2) or not actions.size or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite nonempty actions [T,A] or [A], got {actions.shape}")
    if rotation_unit not in ("rad", "deg"):
        raise ValueError(f"Unsupported rotation unit: {rotation_unit!r}")
    if rotation_unit == "deg":
        if actions.shape[-1] not in (7, 14):
            raise ValueError("Degree conversion requires action-major [T,7] or [T,14]")
        actions = actions.copy()
        for offset in range(0, actions.shape[-1], POLICY_BLOCK_DIM):
            actions[..., offset + 3:offset + 6] *= np.pi / 180.0
    return actions


def resolve_server_contract(config, metadata, rotation_unit=None):
    train = config.get("train", {})
    mode = resolve_action_chunk_mode(config)
    frame = train.get("server_action_frame", metadata.get("action_frame", "flange" if mode == 'delta_flange' else "tcp"))
    unit = rotation_unit or train.get("server_rotation_unit", "rad")
    declared = metadata.get("action_frame")
    if declared and frame != declared:
        print(f"[Env Client] Server action_frame={declared!r}; using configured {frame!r}.")
    if mode == "relative_tcp" and frame not in ("tcp", "tcp_local"):
        raise ValueError(f"relative_tcp needs TCP-local actions; server frame is {frame!r}.")
    if mode == 'delta_flange' and frame not in ('flange', 'flange_local'):
        raise ValueError(f'delta_flange needs per-step flange-local actions; server frame is {frame!r}.')
    if unit not in ("rad", "deg"):
        raise ValueError(f"Unsupported server rotation unit {unit!r}")
    return frame, unit


def compensate_chunk_delay(
    chunk, delay_steps: int, *, drop_exhausted=False
):
    """Drop actions made stale while an asynchronous prediction was in flight."""
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    if chunk.paths:
        return chunk, 0
    if drop_exhausted and delay_steps >= len(chunk):
        return None, len(chunk)
    dropped_steps = min(delay_steps, len(chunk) - 1)
    return chunk.drop(dropped_steps), dropped_steps
