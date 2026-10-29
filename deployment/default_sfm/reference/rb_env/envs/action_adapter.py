"""Action Adapter: turn a desired Cartesian state delta into a robot command.

A policy trained on *state deltas* predicts what should happen in physical
space, not what to send to a particular controller.  No robot tracks a command
perfectly, so replaying the predicted displacement verbatim leaves a
systematic error.  The Action Adapter closes that gap with one linear map per
arm::

    u_hat = W @ delta_p + b

``delta_p`` and ``u_hat`` are both **world-frame** 6-vectors
``[dx, dy, dz, rx, ry, rz]`` (meters and a rotation vector in radians), the
exact convention `Rb3Env.step` consumes: the position half is added in the
world frame and the rotation half left-multiplies the current orientation
(``R_delta * R_curr``, i.e. ``dR = R_next @ R_curr.T``).  The gripper channel
is not part of the adapter.

The module holds the whole life cycle of that map:

* :class:`LinearActionAdapter` -- inference (:meth:`~LinearActionAdapter.predict`),
  online adaptation (:meth:`~LinearActionAdapter.update`), and persistence.
* :class:`MultiSideActionAdapter` -- one adapter per arm in a single ``.npz``.
* :class:`ScriptedRandomPolicy` / :func:`collect_calibration_data` /
  :func:`fit_action_adapter` -- the ~1 minute offline calibration.

Calibration CLI::

    python -m rb_env.envs.action_adapter \
        --env_name default_integration --sides left right \
        --trajectories 10 --horizon 50 --seed 0 --out weights.npz

The environment builder is imported lazily inside ``main()``: importing it at
module scope would close the cycle
``builder_integration -> wrappers -> action_adapter``.
"""
from __future__ import annotations

if __name__ == "__main__":  # pragma: no cover - executable entry point
    # Same contract as the other entry points: load the ROS overlays and
    # re-exec before anything imports rclpy (the builder does, lazily).
    from Real_Robo.ros_environment import ensure_ros_environment

    ensure_ros_environment()

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rb_env.envs.action_utils import POLICY_BLOCK_DIM
from rb_env.utils.scipy_spatial_utils import (
    apply_global_pose_offset_rotvec,
    calculate_global_pose_offset_rotvec,
)

# [dx, dy, dz, rx, ry, rz] in the world frame; the gripper channel is separate.
ADAPTER_DIM = 6

FORMAT_VERSION = 1

SIDES = ("left", "right")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineAdaptationSpec:
    """How aggressively the adapter re-fits itself while the robot runs."""

    enable: bool = True
    mu: float = 0.01
    normalize: bool = True
    bias_scale: float = 0.01
    deadband: float = 1e-4
    max_weight_drift: float = 0.5
    reset_each_episode: bool = True

    def __post_init__(self):
        if not np.isfinite(self.mu) or self.mu < 0.0:
            raise ValueError(f"mu must be finite and non-negative, got {self.mu}")
        if not np.isfinite(self.bias_scale) or self.bias_scale <= 0.0:
            raise ValueError(
                f"bias_scale must be finite and positive, got {self.bias_scale}"
            )
        if not np.isfinite(self.deadband) or self.deadband < 0.0:
            raise ValueError(
                f"deadband must be finite and non-negative, got {self.deadband}"
            )
        if self.max_weight_drift is not None and not np.isfinite(
            self.max_weight_drift
        ):
            raise ValueError("max_weight_drift must be finite or None")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "OnlineAdaptationSpec":
        config = dict(config or {})
        unknown = set(config) - {field for field in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(
                f"Unknown online_adaptation key(s): {sorted(unknown)}; "
                f"expected {sorted(cls.__dataclass_fields__)}"
            )
        return cls(**config)


class LinearActionAdapter:
    """``u_hat = W @ delta_p + b`` for one arm, in the world frame."""

    def __init__(
        self,
        W: np.ndarray,
        b: np.ndarray,
        *,
        online: OnlineAdaptationSpec | None = None,
        zero_eps: float = 1e-9,
    ):
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be a square matrix, got shape {W.shape}")
        if b.shape != (W.shape[0],):
            raise ValueError(
                f"b must have shape ({W.shape[0]},), got {b.shape}"
            )
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(b)):
            raise ValueError("W and b must be finite")

        self.W = W.copy()
        self.b = b.copy()
        self.online = online or OnlineAdaptationSpec()
        self.zero_eps = float(zero_eps)
        self._W0 = self.W.copy()
        self._b0 = self.b.copy()

    # -- construction ----------------------------------------------------

    @classmethod
    def identity(
        cls, dim: int = ADAPTER_DIM, *, online: OnlineAdaptationSpec | None = None
    ) -> "LinearActionAdapter":
        """The harmless adapter: commands pass through untouched."""
        return cls(np.eye(int(dim)), np.zeros(int(dim)), online=online)

    @property
    def dim(self) -> int:
        return int(self.W.shape[0])

    @property
    def calibration(self) -> tuple[np.ndarray, np.ndarray]:
        """The (W, b) the adapter was created with, before any online update."""
        return self._W0.copy(), self._b0.copy()

    def with_online(self, online: OnlineAdaptationSpec) -> "LinearActionAdapter":
        self.online = online
        return self

    def copy(self) -> "LinearActionAdapter":
        clone = LinearActionAdapter(
            self.W, self.b, online=self.online, zero_eps=self.zero_eps
        )
        clone._W0 = self._W0.copy()
        clone._b0 = self._b0.copy()
        return clone

    # -- inference -------------------------------------------------------

    def predict(self, delta: np.ndarray) -> np.ndarray:
        """Command that should produce the desired world-frame ``delta``.

        A request to stay put maps to *exactly* zero rather than to ``b``:
        a hold action (the idle block the rollout client sends while a human
        intervenes, or a policy that predicts no motion) must not creep.
        """
        delta = self._as_vector(delta, "delta")
        if float(np.linalg.norm(delta)) < self.zero_eps:
            return np.zeros(self.dim)
        return self.W @ delta + self.b

    __call__ = predict

    # -- online adaptation -----------------------------------------------

    def update(
        self, observed_delta: np.ndarray, command_sent: np.ndarray
    ) -> np.ndarray | None:
        """One LMS step from what the robot actually did.

        ``e = W @ delta_obs + b - u_hat`` is the adapter's own prediction
        error: the command it *would* ask for to reproduce the motion that was
        just observed, minus the command that was actually sent.  Then
        ``W <- W - mu e delta_obs^T`` and ``b <- b - mu e``, which is what
        plain LMS (``normalize=False``) does -- there ``mu`` is in units of
        one over squared meters, so it has to be tuned to the size of the
        deltas the policy asks for.

        ``normalize=True`` runs the same update as NLMS on the regressor
        ``[delta_obs, bias_scale]``, which makes ``mu`` the *fraction of the
        residual* corrected per step (stable for ``mu < 2``) regardless of how
        big the deltas are.  ``bias_scale`` is the delta magnitude at which
        the intercept carries as much of the update as the deltas do; leaving
        it at the typical step size keeps the intercept from soaking up
        essentially the whole update, which is what an unscaled ``1`` in that
        regressor would do.

        Returns the error, or ``None`` when the step was skipped.
        """
        if not self.online.enable:
            return None

        observed = self._as_vector(observed_delta, "observed_delta")
        command = self._as_vector(command_sent, "command_sent")
        if float(np.linalg.norm(observed)) < self.online.deadband:
            # Sensor noise around a standstill carries no information about
            # the command-to-motion map, and would drag ``b`` around.
            return None

        error = self.W @ observed + self.b - command
        bias_regressor = self.online.bias_scale**2
        step = self.online.mu
        if self.online.normalize:
            step /= 1e-24 + float(observed @ observed) + bias_regressor
            self.b -= step * bias_regressor * error
        else:
            self.b -= step * error
        self.W -= step * np.outer(error, observed)
        self._clip_drift()
        return error

    def reset_online(self) -> None:
        """Forget online adaptation and return to the calibrated map."""
        self.W = self._W0.copy()
        self.b = self._b0.copy()

    def _clip_drift(self) -> None:
        """Keep the online map near its calibration, whatever the data does."""
        limit = self.online.max_weight_drift
        if limit is None or limit <= 0.0:
            return
        for current, anchor in ((self.W, self._W0), (self.b, self._b0)):
            deviation = current - anchor
            norm = float(np.linalg.norm(deviation))
            if norm > limit:
                current[...] = anchor + deviation * (limit / norm)

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> Path:
        return MultiSideActionAdapter({"default": self}).save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        side: str = "default",
        *,
        online: OnlineAdaptationSpec | None = None,
    ) -> "LinearActionAdapter":
        return MultiSideActionAdapter.load(path, sides=(side,), online=online)[side]

    def _as_vector(self, value, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"{name} must have shape ({self.dim},), got "
                f"{np.shape(value)}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return vector

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LinearActionAdapter(dim={self.dim}, "
            f"W_drift={np.linalg.norm(self.W - self._W0):.4g}, "
            f"b={np.array2string(self.b, precision=4)})"
        )


class MultiSideActionAdapter:
    """One :class:`LinearActionAdapter` per arm, stored in a single file."""

    def __init__(self, adapters: Mapping[str, LinearActionAdapter]):
        if not adapters:
            raise ValueError("at least one side is required")
        self._adapters = dict(adapters)

    def __getitem__(self, side: str) -> LinearActionAdapter:
        try:
            return self._adapters[side]
        except KeyError:
            raise KeyError(
                f"no adapter for side {side!r}; available: {sorted(self._adapters)}"
            ) from None

    def __contains__(self, side: object) -> bool:
        return side in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def items(self):
        return self._adapters.items()

    def reset_online(self) -> None:
        for adapter in self._adapters.values():
            adapter.reset_online()

    def with_online(self, online: OnlineAdaptationSpec) -> "MultiSideActionAdapter":
        for adapter in self._adapters.values():
            adapter.with_online(online)
        return self

    @classmethod
    def identity(
        cls,
        sides: Iterable[str],
        dim: int = ADAPTER_DIM,
        *,
        online: OnlineAdaptationSpec | None = None,
    ) -> "MultiSideActionAdapter":
        return cls(
            {
                side: LinearActionAdapter.identity(dim, online=online)
                for side in sides
            }
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "format_version": np.asarray(FORMAT_VERSION),
            "sides": np.asarray(self.sides, dtype="U32"),
            "dim": np.asarray(next(iter(self._adapters.values())).dim),
            "created_at": np.asarray(
                time.strftime("%Y-%m-%dT%H:%M:%S"), dtype="U32"
            ),
        }
        for side, adapter in self._adapters.items():
            payload[f"{side}__W"] = adapter.W
            payload[f"{side}__b"] = adapter.b
        np.savez(path, **payload)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        sides: Sequence[str] | None = None,
        online: OnlineAdaptationSpec | None = None,
        allow_missing: bool = False,
    ) -> "MultiSideActionAdapter":
        """Load weights, optionally restricted to ``sides``.

        ``allow_missing`` falls back to an identity adapter for a side the
        file does not carry.  It is off by default: an enabled adapter that
        silently degrades to a no-op is the kind of thing that gets found out
        on the robot.
        """
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"action adapter weights not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            stored = [str(side) for side in data["sides"]] if "sides" in data else []
            dim = int(data["dim"]) if "dim" in data else ADAPTER_DIM
            wanted = list(sides) if sides is not None else stored
            if not wanted:
                raise ValueError(f"{path} contains no adapters")

            adapters = {}
            missing = []
            for side in wanted:
                if f"{side}__W" in data and f"{side}__b" in data:
                    adapters[side] = LinearActionAdapter(
                        data[f"{side}__W"], data[f"{side}__b"], online=online
                    )
                else:
                    missing.append(side)
                    adapters[side] = LinearActionAdapter.identity(dim, online=online)

        if missing and not allow_missing:
            raise KeyError(
                f"{path} has no weights for side(s) {missing}; stored sides: "
                f"{stored}"
            )
        return cls(adapters)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSpec:
    """The scripted random policy and the fit that follow it.

    Defaults reproduce the reference protocol: ``M = 10`` trajectories of
    ``K = 50`` steps (up to 500 pairs, about a minute at 10 Hz), a step size
    drawn uniformly from [0.002, 0.01] m and [0.005, 0.02] rad, and a
    direction vector refreshed every 10 steps by mixing 70% of the previous
    direction with 30% Gaussian noise.
    """

    trajectories: int = 10
    horizon: int = 50
    position_step_range: tuple[float, float] = (0.002, 0.01)
    rotation_step_range: tuple[float, float] = (0.005, 0.02)
    direction_refresh_steps: int = 10
    direction_momentum: float = 0.7
    seed: int | None = None
    ridge: float = 0.0
    fit_bias: bool = True
    eval_trajectories: int = 2
    eval_seed: int | None = None
    # Steps between a command and the motion it is paired with in the fit
    # (see `lagged_pairs`); 0 pairs each command with its own step.
    lag: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "position_step_range", _step_range(self.position_step_range, "position")
        )
        object.__setattr__(
            self, "rotation_step_range", _step_range(self.rotation_step_range, "rotation")
        )
        if self.trajectories < 1:
            raise ValueError("trajectories must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.direction_refresh_steps < 1:
            raise ValueError("direction_refresh_steps must be >= 1")
        if not 0.0 <= self.direction_momentum <= 1.0:
            raise ValueError(
                f"direction_momentum must be in [0, 1], got {self.direction_momentum}"
            )
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.eval_trajectories < 0:
            raise ValueError("eval_trajectories must be >= 0")
        if self.lag < 0 or self.lag >= self.horizon:
            raise ValueError(f"lag must be in [0, horizon), got {self.lag}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "CalibrationSpec":
        config = dict(config or {})
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown calibration key(s): {sorted(unknown)}; "
                f"expected {sorted(cls.__dataclass_fields__)}"
            )
        return cls(**config)

    def held_out(self) -> "CalibrationSpec | None":
        """The spec for the separate evaluation pass, or ``None`` if disabled.

        Every fitted pair is training data -- the fit uses all of them -- so
        the only honest way to read the numbers is on trajectories the fit
        never saw.  A different seed makes those genuinely different
        trajectories rather than a replay of the calibration ones.
        """
        if self.eval_trajectories <= 0:
            return None
        seed = self.eval_seed
        if seed is None and self.seed is not None:
            seed = self.seed + 10_000
        return replace(
            self, trajectories=self.eval_trajectories, seed=seed, eval_trajectories=0
        )

    def merged(self, **overrides) -> "CalibrationSpec":
        """Apply the overrides that were actually given (``None`` = keep)."""
        given = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **given) if given else self


def _step_range(value, name: str) -> tuple[float, float]:
    low, high = (float(v) for v in value)
    if not (0.0 <= low <= high) or not np.isfinite(high):
        raise ValueError(
            f"{name}_step_range must be a finite, non-negative, ordered pair, "
            f"got {value}"
        )
    return (low, high)


class ScriptedRandomPolicy:
    """``pi_rand``: a smoothly wandering source of world-frame deltas.

    Calibration needs commands that excite every axis without whipping the
    arm around, so the direction is held for ``direction_refresh_steps`` and
    then only partially replaced, while the magnitude is redrawn every step.
    Position and rotation directions are normalized separately, which makes
    the sampled step sizes literal: meters for the position half, radians for
    the rotation half.
    """

    def __init__(self, spec: CalibrationSpec, rng: np.random.Generator | None = None):
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng(spec.seed)
        self.reset()

    def reset(self) -> None:
        self._direction = self._normalize(self.rng.standard_normal(ADAPTER_DIM))
        self._steps = 0

    @property
    def direction(self) -> np.ndarray:
        return self._direction.copy()

    def sample(self) -> np.ndarray:
        if self._steps and self._steps % self.spec.direction_refresh_steps == 0:
            momentum = self.spec.direction_momentum
            self._direction = self._normalize(
                momentum * self._direction
                + (1.0 - momentum) * self.rng.standard_normal(ADAPTER_DIM)
            )

        position_step = self.rng.uniform(*self.spec.position_step_range)
        rotation_step = self.rng.uniform(*self.spec.rotation_step_range)
        self._steps += 1
        return np.concatenate(
            [
                position_step * self._direction[:3],
                rotation_step * self._direction[3:],
            ]
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).copy()
        for block in (slice(0, 3), slice(3, 6)):
            norm = float(np.linalg.norm(vector[block]))
            vector[block] = (
                vector[block] / norm if norm > 1e-12 else np.zeros(3)
            )
        return vector


@dataclass
class CalibrationPairs:
    """``(delta_p, u)`` pairs for one arm, plus what was thrown away.

    ``log`` keeps every step that was run -- skipped ones included -- with its
    poses and timing, so a surprising fit can be diagnosed after the fact
    (see :func:`save_calibration_log`).
    """

    deltas: np.ndarray
    commands: np.ndarray
    skipped_clipped: int = 0
    skipped_intervened: int = 0
    skipped_rate_limited: int = 0
    log: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        self.deltas = np.asarray(self.deltas, dtype=float).reshape(-1, ADAPTER_DIM)
        self.commands = np.asarray(self.commands, dtype=float).reshape(
            -1, ADAPTER_DIM
        )
        if len(self.deltas) != len(self.commands):
            raise ValueError("deltas and commands must have the same length")

    def __len__(self) -> int:
        return len(self.deltas)

    @classmethod
    def empty(cls) -> "CalibrationPairs":
        return cls(np.zeros((0, ADAPTER_DIM)), np.zeros((0, ADAPTER_DIM)))


def fit_action_adapter(
    deltas: np.ndarray,
    commands: np.ndarray,
    *,
    ridge: float = 0.0,
    fit_bias: bool = True,
    online: OnlineAdaptationSpec | None = None,
) -> LinearActionAdapter:
    """Ordinary least squares over the calibration pairs.

    Solves ``min_{W, b} sum ||W delta_p + b - u||^2`` in closed form.  Note
    the direction: the *observed displacement* is the input and the *command
    that produced it* is the target, which is exactly how
    :meth:`LinearActionAdapter.predict` is then used at run time -- it is fed
    a desired displacement and returns the command to send.

    ``ridge`` adds an L2 penalty on ``W`` only; the bias is never penalized.
    """
    deltas = np.asarray(deltas, dtype=float).reshape(-1, ADAPTER_DIM)
    commands = np.asarray(commands, dtype=float).reshape(-1, ADAPTER_DIM)
    if len(deltas) != len(commands):
        raise ValueError(
            f"deltas ({len(deltas)}) and commands ({len(commands)}) must match"
        )
    if not np.all(np.isfinite(deltas)) or not np.all(np.isfinite(commands)):
        raise ValueError("calibration pairs must be finite")

    columns = ADAPTER_DIM + (1 if fit_bias else 0)
    if len(deltas) < columns:
        raise ValueError(
            f"need at least {columns} calibration pairs to fit a "
            f"{ADAPTER_DIM}x{ADAPTER_DIM} map, got {len(deltas)}"
        )

    design = (
        np.hstack([deltas, np.ones((len(deltas), 1))]) if fit_bias else deltas
    )
    if ridge > 0.0:
        penalty = np.eye(columns) * float(ridge)
        if fit_bias:
            penalty[-1, -1] = 0.0  # never shrink the bias
        theta = np.linalg.solve(
            design.T @ design + penalty, design.T @ commands
        )
    else:
        theta, *_ = np.linalg.lstsq(design, commands, rcond=None)

    W = theta[:ADAPTER_DIM].T
    b = theta[ADAPTER_DIM] if fit_bias else np.zeros(ADAPTER_DIM)
    return LinearActionAdapter(W, b, online=online)


def evaluate_adapter(
    adapter: LinearActionAdapter, deltas: np.ndarray, commands: np.ndarray
) -> dict[str, Any]:
    """Per-axis fit quality of ``W delta_p + b`` against the sent commands."""
    deltas = np.asarray(deltas, dtype=float).reshape(-1, ADAPTER_DIM)
    commands = np.asarray(commands, dtype=float).reshape(-1, ADAPTER_DIM)
    if len(deltas) == 0:
        raise ValueError("cannot evaluate an adapter on zero pairs")

    predicted = deltas @ adapter.W.T + adapter.b
    residual = predicted - commands
    rmse_per_axis = np.sqrt(np.mean(residual**2, axis=0))
    variance = np.var(commands, axis=0)
    r2_per_axis = np.where(
        variance > 1e-24, 1.0 - np.mean(residual**2, axis=0) / np.maximum(variance, 1e-24), np.nan
    )
    return {
        "pairs": int(len(deltas)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rmse_per_axis": rmse_per_axis,
        "r2_per_axis": r2_per_axis,
    }


def collect_calibration_data(
    env,
    sides: Sequence[str],
    spec: CalibrationSpec | None = None,
    *,
    gripper_actions: Mapping[str, float] | None = None,
    verbose: bool = True,
) -> dict[str, CalibrationPairs]:
    """Drive ``pi_rand`` on the real environment and record ``(delta_p, u)``.

    Each arm gets its own map, so ``sides`` selects which arms this run
    calibrates.  Every other arm the environment builds is *held*: its
    Cartesian command block stays zero and its gripper channel carries that
    arm's no-op value from ``gripper_actions`` (zero is not a no-op for a
    ``continuous`` gripper -- see `compute_noop_gripper_action`).

    The commands go out as plain world-frame delta actions, so the recorded
    pairs describe the controller as it is -- which is why the Action Adapter
    wrapper must be *disabled* while this runs (an enabled one is rejected
    outright, since fitting on already-corrected commands would double-count
    the correction).
    """
    spec = spec or CalibrationSpec()
    sides = tuple(sides)
    assert_adapter_disabled(env)

    base_env = env.unwrapped
    env_sides = tuple(
        side for side in SIDES if getattr(base_env, f"env_{side}", None) is not None
    )
    missing = [side for side in sides if side not in env_sides]
    if missing:
        raise ValueError(
            f"side(s) {missing} are not available in the environment "
            f"(built: {list(env_sides)})"
        )
    side_envs = {side: getattr(base_env, f"env_{side}") for side in sides}

    action_dim = int(env.action_space.shape[0])
    slices = {side: _side_slices(action_dim, side) for side in sides}
    # Held arms need their gripper slot filled too, so lay out every block.
    held_slices = {
        side: _side_slices(action_dim, side)
        for side in env_sides
        if side not in sides and action_dim > POLICY_BLOCK_DIM
    }
    gripper_actions = dict(gripper_actions or {})
    for side in env_sides:
        if side not in gripper_actions:
            gripper_actions[side] = 0.0

    rng = np.random.default_rng(spec.seed)
    policies = {side: ScriptedRandomPolicy(spec, rng) for side in sides}
    collected = {
        side: {
            "deltas": [], "commands": [], "skipped": 0, "intervened": 0,
            "rate_limited": 0, "log": {key: [] for key in _LOG_KEYS},
        }
        for side in sides
    }
    started = time.monotonic()

    for trajectory in range(spec.trajectories):
        env.reset()
        for policy in policies.values():
            policy.reset()

        for step in range(spec.horizon):
            action = np.zeros(action_dim, dtype=np.float32)
            for side, (_, _, gripper_index) in (*slices.items(), *held_slices.items()):
                action[gripper_index] = float(gripper_actions[side])

            commands = {}
            poses_before = {}
            clipped = {}
            rate_limited = {}
            for side in sides:
                command = policies[side].sample()
                pos_slice, rot_slice, _ = slices[side]
                action[pos_slice] = command[:3]
                action[rot_slice] = command[3:]
                commands[side] = command
                poses_before[side] = np.asarray(
                    side_envs[side].currpos, dtype=float
                ).copy()
                clipped[side] = _target_is_clipped(
                    side_envs[side], poses_before[side], command
                )
                rate_limited[side] = _command_is_rate_limited(
                    side_envs[side], command
                )

            from rb_env.envs.action_utils import world_delta_to_action
            _, _, _, _, info = env.step(world_delta_to_action(env, action))
            step_time = time.monotonic() - started
            # A SpaceMouse nudge replaces the command that was sent, so the
            # motion that follows says nothing about the scripted command.
            intervened = isinstance(info, dict) and (
                "intervene_action" in info or "action_intervene" in info
            )

            for side in sides:
                pose_after = np.asarray(side_envs[side].currpos, dtype=float).copy()
                delta = calculate_global_pose_offset_rotvec(
                    poses_before[side], pose_after
                )
                used = not (intervened or clipped[side] or rate_limited[side])
                log = collected[side]["log"]
                for key, value in (
                    ("trajectory", trajectory), ("step", step), ("time", step_time),
                    ("command", commands[side]), ("delta", delta),
                    ("pose_before", poses_before[side]), ("pose_after", pose_after),
                    ("clipped", clipped[side]), ("rate_limited", rate_limited[side]),
                    ("intervened", intervened), ("used", used),
                ):
                    log[key].append(value)

                if intervened:
                    collected[side]["intervened"] += 1
                    continue
                if clipped[side]:
                    # The safety box, not the controller, decided this motion.
                    collected[side]["skipped"] += 1
                    continue
                if rate_limited[side]:
                    # So did the per-step speed bound: the arm moved the
                    # bound's worth, whatever the command asked for.
                    collected[side]["rate_limited"] += 1
                    continue
                collected[side]["deltas"].append(delta)
                collected[side]["commands"].append(commands[side])

        if verbose:
            counts = ", ".join(
                f"{side}={len(collected[side]['deltas'])}" for side in sides
            )
            print(
                f"[action-adapter] trajectory {trajectory + 1}/"
                f"{spec.trajectories} done; pairs: {counts}"
            )

    return {
        side: CalibrationPairs(
            np.asarray(data["deltas"] or np.zeros((0, ADAPTER_DIM))),
            np.asarray(data["commands"] or np.zeros((0, ADAPTER_DIM))),
            skipped_clipped=data["skipped"],
            skipped_intervened=data["intervened"],
            skipped_rate_limited=data["rate_limited"],
            log={key: np.asarray(values) for key, values in data["log"].items()},
        )
        for side, data in collected.items()
    }


_LOG_KEYS = (
    "trajectory", "step", "time", "command", "delta", "pose_before",
    "pose_after", "clipped", "rate_limited", "intervened", "used",
)


def save_calibration_log(
    path: str | Path,
    logs: Mapping[str, Mapping[str, "CalibrationPairs"]],
    *,
    spec: CalibrationSpec | None = None,
    limits: Mapping[str, Mapping[str, float]] | None = None,
) -> Path:
    """Write every calibration step to one ``.npz``.

    Keys are ``<split>__<side>__<field>`` (``split`` is ``train`` or
    ``eval``), so a split/side's log reads back with a prefix filter.
    ``limits`` records the per-step speed bounds each arm ran with.
    """
    import json

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "created_at": np.asarray(time.strftime("%Y-%m-%dT%H:%M:%S"), dtype="U32"),
        "meta": np.asarray(json.dumps({
            "spec": None if spec is None else {
                key: getattr(spec, key) for key in spec.__dataclass_fields__
            },
            "limits": dict(limits or {}),
        })),
    }
    for split, by_side in logs.items():
        for side, pairs in by_side.items():
            for key, value in (pairs.log or {}).items():
                payload[f"{split}__{side}__{key}"] = value
    np.savez(path, **payload)
    return path


def calibrate(
    env,
    sides: Sequence[str],
    spec: CalibrationSpec | None = None,
    *,
    out_path: str | Path | None = None,
    gripper_actions: Mapping[str, float] | None = None,
    online: OnlineAdaptationSpec | None = None,
    merge_existing: bool = False,
    log_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[MultiSideActionAdapter, dict[str, Any]]:
    """Collect, fit, report and (optionally) save -- the whole check-up.

    One arm at a time is a normal way to work, so ``merge_existing`` keeps the
    sides already stored at ``out_path`` and overwrites only the ones just
    calibrated.  Without it the file is replaced outright.  ``log_path``
    additionally keeps every step of both passes (see
    :func:`save_calibration_log`).
    """
    spec = spec or CalibrationSpec()
    if verbose:
        print(f"[action-adapter] collecting training trajectories for {tuple(sides)}")
    pairs = collect_calibration_data(
        env, sides, spec, gripper_actions=gripper_actions, verbose=verbose
    )

    eval_spec = spec.held_out()
    eval_pairs = {}
    if eval_spec is not None:
        if verbose:
            print(
                "[action-adapter] collecting held-out evaluation trajectories "
                f"({eval_spec.trajectories} x {eval_spec.horizon} steps)"
            )
        eval_pairs = collect_calibration_data(
            env, sides, eval_spec, gripper_actions=gripper_actions, verbose=verbose
        )

    adapters, report = _fit_and_report(pairs, eval_pairs, spec, online)

    if log_path is not None:
        base_env = env.unwrapped
        limits = {
            side: {
                "translation_m": _step_limits(getattr(base_env, f"env_{side}"))[0],
                "rotation_rad": _step_limits(getattr(base_env, f"env_{side}"))[1],
            }
            for side in sides
        }
        report["log_path"] = str(save_calibration_log(
            log_path, {"train": pairs, "eval": eval_pairs}, spec=spec, limits=limits
        ))

    multi = _save_adapters(
        adapters, report, out_path, merge_existing=merge_existing, online=online,
        verbose=verbose,
    )
    return multi, report


def lagged_pairs(pairs: CalibrationPairs, lag: int) -> CalibrationPairs:
    """Pair each command with the motion measured ``lag`` steps later.

    The arm answers a step's command mostly during the *next* control period,
    so pairing a command with its own step's motion mixes in the previous
    command's response and biases ``W`` low.  Both steps must have been kept
    (``used``) and belong to the same trajectory.  ``lag = 0`` is the
    original pairing.
    """
    if lag == 0:
        return pairs
    log = pairs.log
    if log is None or len(log.get("used", ())) == 0:
        raise ValueError("lagged pairing needs the per-step log")
    trajectory = np.asarray(log["trajectory"])
    used = np.asarray(log["used"], dtype=bool)
    keep = (trajectory[:-lag] == trajectory[lag:]) & used[:-lag] & used[lag:]
    return replace(
        pairs,
        deltas=np.asarray(log["delta"])[lag:][keep],
        commands=np.asarray(log["command"])[:-lag][keep],
    )


def _fit_and_report(
    pairs: Mapping[str, CalibrationPairs],
    eval_pairs: Mapping[str, CalibrationPairs],
    spec: CalibrationSpec,
    online: OnlineAdaptationSpec | None,
) -> tuple[dict[str, LinearActionAdapter], dict[str, Any]]:
    adapters = {}
    report: dict[str, Any] = {}
    for side, side_pairs in pairs.items():
        # Every collected pair goes into the fit; evaluation uses the
        # separately collected trajectories instead of a split.
        train = lagged_pairs(side_pairs, spec.lag)
        adapter = fit_action_adapter(
            train.deltas,
            train.commands,
            ridge=spec.ridge,
            fit_bias=spec.fit_bias,
            online=online,
        )
        adapters[side] = adapter
        held_out = eval_pairs.get(side)
        if held_out is not None and len(held_out) > 0:
            held_out = lagged_pairs(held_out, spec.lag)
        report[side] = {
            # A controller that tracks its commands gives a diagonal near 1.
            "W_diag": np.diag(adapter.W).copy(),
            "lag": spec.lag,
            "train": _pair_metrics(adapter, train),
            "eval": (
                _pair_metrics(adapter, held_out)
                if held_out is not None and len(held_out) > 0
                else None
            ),
        }
    return adapters, report


def _save_adapters(
    adapters: Mapping[str, LinearActionAdapter],
    report: dict[str, Any],
    out_path: str | Path | None,
    *,
    merge_existing: bool,
    online: OnlineAdaptationSpec | None,
    verbose: bool,
) -> MultiSideActionAdapter:
    multi = MultiSideActionAdapter(adapters)
    if out_path is not None:
        stored = _existing_sides(out_path)
        dropped = [side for side in stored if side not in adapters]
        if dropped:
            if merge_existing:
                previous = MultiSideActionAdapter.load(out_path, online=online)
                multi = MultiSideActionAdapter(
                    {**dict(previous.items()), **adapters}
                )
                report["kept_sides"] = dropped
            elif verbose:
                print(
                    f"[action-adapter] warning: {out_path} already holds "
                    f"{dropped}; those weights are being replaced. Pass "
                    "--merge to keep them."
                )
        report["path"] = str(multi.save(out_path))
        report["saved_sides"] = list(multi.sides)
    return multi


def load_calibration_log(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, CalibrationPairs]]]:
    """Read a :func:`save_calibration_log` file back into pairs.

    Returns ``(meta, {split: {side: CalibrationPairs}})``; each
    ``CalibrationPairs`` carries the full step log and, as its pairs, the
    steps calibration kept.
    """
    import json

    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        logs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for key in data.files:
            parts = key.split("__")
            if len(parts) == 3:
                split, side, field = parts
                logs.setdefault(split, {}).setdefault(side, {})[field] = data[key]

    splits: dict[str, dict[str, CalibrationPairs]] = {}
    for split, by_side in logs.items():
        for side, log in by_side.items():
            if len(log.get("used", ())) == 0:
                continue
            used = np.asarray(log["used"], dtype=bool)
            intervened = np.asarray(log["intervened"], dtype=bool)
            clipped = np.asarray(log["clipped"], dtype=bool) & ~intervened
            limited = np.asarray(log["rate_limited"], dtype=bool) & ~intervened & ~clipped
            splits.setdefault(split, {})[side] = CalibrationPairs(
                np.asarray(log["delta"])[used],
                np.asarray(log["command"])[used],
                skipped_clipped=int(clipped.sum()),
                skipped_intervened=int(intervened.sum()),
                skipped_rate_limited=int(limited.sum()),
                log=log,
            )
    return meta, splits


def refit_from_log(
    log_path: str | Path,
    spec: CalibrationSpec,
    *,
    sides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    online: OnlineAdaptationSpec | None = None,
    merge_existing: bool = False,
    verbose: bool = True,
) -> tuple[MultiSideActionAdapter, dict[str, Any]]:
    """Fit again from a saved step log -- no robot, no new data.

    Only the fit settings of ``spec`` matter here (``lag``, ``ridge``,
    ``fit_bias``); the data is whatever the log recorded.
    """
    _, splits = load_calibration_log(log_path)
    train = splits.get("train") or {}
    wanted = tuple(sides) if sides else tuple(train)
    missing = [side for side in wanted if side not in train]
    if missing:
        raise KeyError(f"{log_path} has no training log for {missing}; has {sorted(train)}")
    pairs = {side: train[side] for side in wanted}
    eval_pairs = {
        side: value for side, value in (splits.get("eval") or {}).items() if side in wanted
    }
    adapters, report = _fit_and_report(pairs, eval_pairs, spec, online)
    report["refit_from"] = str(Path(log_path).expanduser())
    multi = _save_adapters(
        adapters, report, out_path, merge_existing=merge_existing, online=online,
        verbose=verbose,
    )
    return multi, report


def _existing_sides(path: str | Path) -> list[str]:
    path = Path(path).expanduser()
    if not path.is_file():
        return []
    try:
        with np.load(path, allow_pickle=False) as data:
            return [str(side) for side in data.get("sides", [])]
    except Exception:  # pragma: no cover - unreadable file is not our problem
        return []


def _pair_metrics(
    adapter: LinearActionAdapter, pairs: "CalibrationPairs"
) -> dict[str, Any]:
    return {
        **evaluate_adapter(adapter, pairs.deltas, pairs.commands),
        "skipped_clipped": pairs.skipped_clipped,
        "skipped_intervened": pairs.skipped_intervened,
        "skipped_rate_limited": pairs.skipped_rate_limited,
    }


def format_report(report: Mapping[str, Any]) -> str:
    lines = []
    for side, metrics in report.items():
        if side in ("path", "kept_sides", "saved_sides", "log_path", "refit_from"):
            continue
        for split in ("train", "eval"):
            split_metrics = metrics.get(split)
            if split_metrics is None:
                lines.append(f"[{side}] {split}: not collected")
                continue
            lines.append(
                f"[{side}] {split}: pairs={split_metrics['pairs']} "
                f"skipped_clipped={split_metrics['skipped_clipped']} "
                f"skipped_intervened={split_metrics['skipped_intervened']} "
                f"skipped_rate_limited={split_metrics.get('skipped_rate_limited', 0)} "
                f"rmse={split_metrics['rmse']:.6f}"
            )
            lines.append(
                "  rmse/axis: "
                + np.array2string(split_metrics["rmse_per_axis"], precision=6)
            )
            lines.append(
                "  r2/axis:   "
                + np.array2string(split_metrics["r2_per_axis"], precision=4)
            )
        adapter_diag = metrics.get("W_diag")
        if adapter_diag is not None:
            lines.append(
                f"[{side}] W diag:   " + np.array2string(np.asarray(adapter_diag), precision=3)
                + f"   (lag {metrics.get('lag', 0)})"
            )
    if "refit_from" in report:
        lines.append(f"refit from step log {report['refit_from']}")
    if "log_path" in report:
        lines.append(f"step log -> {report['log_path']}")
    if "kept_sides" in report:
        lines.append(f"kept existing weights for: {report['kept_sides']}")
    if "path" in report:
        lines.append(f"saved {report['saved_sides']} -> {report['path']}")
    return "\n".join(lines)


def assert_adapter_disabled(env) -> None:
    """Refuse to calibrate through an active Action Adapter wrapper."""
    current = env
    seen = 0
    while current is not None and seen < 64:
        if type(current).__name__ == "ActionAdapterWrapper" and getattr(
            current, "enabled", False
        ):
            raise RuntimeError(
                "The environment already applies an ActionAdapterWrapper. "
                "Calibrate with action_adapter.enable = false, otherwise the "
                "fit would be made on already-corrected commands."
            )
        current = getattr(current, "env", None)
        seen += 1


def _side_slices(action_dim: int, side: str):
    from rb_env.envs.action_utils import resolve_side_slices

    return resolve_side_slices(action_dim, side)


def _target_is_clipped(side_env, pose_before: np.ndarray, command: np.ndarray) -> bool:
    """Would `Rb3Env.clip_safety_box` move the target this command asks for?"""
    clip = getattr(side_env, "clip_safety_box", None)
    if clip is None:
        return False
    target = apply_global_pose_offset_rotvec(pose_before, command)
    try:
        clipped = clip(target.copy())
    except Exception:  # pragma: no cover - a stub without a bounding box
        return False
    return not np.allclose(np.asarray(clipped, dtype=float), target, atol=1e-12)


def _step_limits(side_env) -> tuple[float | None, float | None]:
    """Per-step (translation m, rotation rad) bounds `Rb3Env` sends, if known."""
    hz = getattr(side_env, "hz", None)
    if not hz:
        return None, None
    translation = getattr(side_env, "max_translation_speed_mps", None)
    rotation = getattr(side_env, "max_rotation_speed_radps", None)
    return (
        None if translation is None else float(translation) / float(hz),
        None if rotation is None else float(rotation) / float(hz),
    )


def _command_is_rate_limited(side_env, command: np.ndarray) -> bool:
    """Would the per-step speed bound shorten this command?

    `Rb3Env._send_pos_command` scales the translation and rotation halves of
    every step down to ``max_*_speed / hz``; a pair recorded past that bound
    measures the bound, not the controller's response to the command.
    """
    translation, rotation = _step_limits(side_env)
    command = np.asarray(command, dtype=float)
    return bool(
        (translation is not None and np.linalg.norm(command[:3]) > translation + 1e-12)
        or (rotation is not None and np.linalg.norm(command[3:]) > rotation + 1e-12)
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate the Action Adapter on the real robot."
    )
    parser.add_argument("--env_path", type=str, default=None)
    parser.add_argument("--env_name", type=str, default="default_integration")
    parser.add_argument("--sides", nargs="+", default=None, choices=list(SIDES))
    parser.add_argument("--trajectories", type=int, default=None, help="M")
    parser.add_argument("--horizon", type=int, default=None, help="K")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument(
        "--eval_trajectories",
        type=int,
        default=None,
        help="Extra trajectories collected after the fit, used only to report "
        "held-out error. 0 skips the evaluation pass.",
    )
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument(
        "--no_bias", action="store_true", help="fit W only, forcing b = 0"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to write the .npz; defaults to action_adapter.weights_path.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Keep the weights already stored for arms this run does not "
        "calibrate, instead of replacing the whole file.",
    )
    parser.add_argument(
        "--position_step_range", type=float, nargs=2, default=None,
        metavar=("LOW", "HIGH"), help="per-step translation in meters",
    )
    parser.add_argument(
        "--rotation_step_range", type=float, nargs=2, default=None,
        metavar=("LOW", "HIGH"), help="per-step rotation in radians",
    )
    parser.add_argument(
        "--log_out",
        type=str,
        default=None,
        help="Where to write the per-step log (.npz); defaults to "
        "<weights dir>/logs/<weights name>_<timestamp>.npz.",
    )
    parser.add_argument(
        "--lag", type=int, default=None,
        help="pair each command with the motion this many steps later "
        "(default: action_adapter.calibration.lag, else 0)",
    )
    parser.add_argument(
        "--from_log", type=str, default=None,
        help="Refit from a saved step log instead of moving the robot; "
        "writes <weights dir>/<log name>_lag<N>.npz unless --out is given.",
    )
    args = parser.parse_args(argv)

    # Lazy: builder_integration imports wrappers, which imports this module.
    from Real_Robo.resource_paths import config_dir
    from rb_env.envs.builder_integration import (
        build_environment,
        load_config,
        resolve_action_adapter_config,
    )
    from rb_env.gripper_drivers import compute_noop_gripper_action

    env_dir = Path(args.env_path) if args.env_path else config_dir()
    config = load_config(env_dir / f"{args.env_name}.json")
    adapter_cfg = resolve_action_adapter_config(config, require_enabled=False)

    # Collection must never run through an adapter.
    config.setdefault("action_adapter", {})["enable"] = False

    spec = adapter_cfg["calibration"].merged(
        trajectories=args.trajectories,
        horizon=args.horizon,
        seed=args.seed,
        ridge=args.ridge,
        fit_bias=False if args.no_bias else None,
        eval_trajectories=args.eval_trajectories,
        eval_seed=args.eval_seed,
        position_step_range=args.position_step_range,
        rotation_step_range=args.rotation_step_range,
        lag=args.lag,
    )

    if args.from_log:
        # Offline: no environment is built and nothing moves.
        log_file = Path(args.from_log).expanduser()
        refit_out = Path(args.out).expanduser() if args.out else (
            Path(adapter_cfg["weights_path"]).parent / f"{log_file.stem}_lag{spec.lag}.npz"
        )
        print(f"[action-adapter] refitting from {log_file} with lag={spec.lag} -> {refit_out}")
        _, report = refit_from_log(
            log_file, spec, sides=args.sides, out_path=refit_out,
            merge_existing=args.merge, verbose=True,
        )
        print(format_report(report))
        return 0

    arm_type = config["train"]["arm_type"]
    active_sides = SIDES if arm_type == "dual" else (arm_type,)
    sides = tuple(args.sides) if args.sides else tuple(
        side for side in adapter_cfg["sides"] if side in active_sides
    )
    unavailable = [side for side in sides if side not in active_sides]
    if unavailable:
        raise SystemExit(
            f"side(s) {unavailable} are not built by train.arm_type={arm_type!r}"
        )
    out_path = Path(args.out).expanduser() if args.out else adapter_cfg["weights_path"]
    log_path = Path(args.log_out).expanduser() if args.log_out else (
        Path(out_path).parent / "logs"
        / f"{Path(out_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    )

    env = build_environment(config)
    try:
        # Every built arm needs its no-op gripper value, including the ones
        # this run only holds still.
        gripper_actions = {
            side: compute_noop_gripper_action(
                (config["env"].get(side) or {}).get("gripper_control")
            )
            for side in active_sides
        }
        held = [side for side in active_sides if side not in sides]
        print(
            f"[action-adapter] calibrating sides={sides} "
            + (f"(holding {held}) " if held else "")
            + f"M={spec.trajectories} K={spec.horizon} "
            + f"pos={spec.position_step_range} rot={spec.rotation_step_range} "
            + f"-> {out_path}"
        )
        for side in sides:
            translation, rotation = _step_limits(getattr(env.unwrapped, f"env_{side}"))
            if (translation is not None and spec.position_step_range[1] > translation) or (
                rotation is not None and spec.rotation_step_range[1] > rotation
            ):
                print(
                    f"[action-adapter] warning: [{side}] step ranges exceed the "
                    f"per-step speed bound (translation {translation}, rotation "
                    f"{rotation}); those steps are dropped from the fit."
                )
        _, report = calibrate(
            env,
            sides,
            spec,
            out_path=out_path,
            gripper_actions=gripper_actions,
            merge_existing=args.merge,
            log_path=log_path,
            verbose=True,
        )
    finally:
        env.close()

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
