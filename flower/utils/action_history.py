"""Action-space constants derived from SFM's fitted normalization statistics."""
import numpy as np


def normalized_zero_motion_action(normalization):
    """Normalize physical zero translation/rotation, with startup gripper +1.

    Only synthetic history padding uses this value. Recorded rel_actions are
    already normalized and must pass through unchanged. This matches Quest's
    normalize_delta_actions in float64 before the final float32 cast.
    """
    stats = normalization["action_pose"]
    if stats["method"] != "symlog_minmax":
        raise ValueError("Skill history padding requires SFM symlog_minmax action statistics")
    offset = np.asarray(stats["offset"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)
    temperature = float(stats["temperature"])
    if (offset.shape != (6,) or scale.shape != (6,) or not np.isfinite(offset).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Invalid action normalization statistics for Skill history padding")
    unit = -offset / scale
    action = np.ones(7, dtype=np.float64)
    action[:6] = np.sign(unit) * np.log1p(temperature * np.abs(unit)) / np.log1p(temperature)
    return action.astype(np.float32)
