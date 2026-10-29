"""Internal policy plans. Environment actions are plain absolute-RPY arrays.

These containers never cross the environment step boundary.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class PreparedChunk:
    actions: np.ndarray
    paths: dict[str, np.ndarray] = field(default_factory=dict)
    action_period_s: float = 0.1
    policy_horizon: int | None = None
    # Paths with one fixed absolute target per control cycle; no resampling.
    fixed_control_path: bool = False

    def __post_init__(self):
        actions = np.asarray(self.actions, dtype=float).copy()
        if actions.ndim != 2 or not len(actions) or not np.all(np.isfinite(actions)):
            raise ValueError("actions must be a nonempty finite [T, A] array")
        period = float(self.action_period_s)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("action_period_s must be finite and positive")
        horizon = len(actions) if self.policy_horizon is None else self.policy_horizon
        if type(horizon) is not int or horizon < 1:
            raise ValueError("policy_horizon must be a positive integer")
        object.__setattr__(self, "policy_horizon", horizon)
        paths = {}
        for side, value in self.paths.items():
            path = np.asarray(value, dtype=float).copy()
            if path.shape != (len(actions) + 1, 7) or not np.all(np.isfinite(path)):
                raise ValueError("paths must contain finite [T+1, 7] quaternion poses")
            norms = np.linalg.norm(path[:, 3:], axis=1)
            if np.any(norms < 1e-12):
                raise ValueError("path quaternions must be nonzero")
            path[:, 3:] /= norms[:, None]
            path.setflags(write=False)
            paths[side] = path
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "paths", paths)
        if self.fixed_control_path and not paths:
            raise ValueError("fixed control paths require absolute poses")
        object.__setattr__(self, "action_period_s", period)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.actions[index]

    def drop(self, count):
        if self.paths:
            raise ValueError("Progress-tracked paths must not be dropped by elapsed steps")
        return PreparedChunk(self.actions[min(max(0, count), len(self)-1):],
                             action_period_s=self.action_period_s, policy_horizon=self.policy_horizon)

    def prefix(self, count):
        count = min(max(1, int(count)), len(self))
        return PreparedChunk(self.actions[:count],
                             {s: p[:count+1] for s, p in self.paths.items()},
                             self.action_period_s, self.policy_horizon, self.fixed_control_path)

    @classmethod
    def from_actions(cls, actions):
        return cls(actions)
