"""Stored policy label layouts and their conversion to executable pose labels."""
import numpy as np

OBSERVED = 'relative_observed'
RESIDUAL = 'relative_observed_residual'
RELATIVE_MODES = ('relative', OBSERVED, RESIDUAL)
MODES = ('delta', 'delta_flange', *RELATIVE_MODES)


def arm_width(mode):
    return 13 if mode == RESIDUAL else 7


def columns(mode, sides):
    pose = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    if mode == RESIDUAL:
        names = ['observed/' + key for key in pose] + ['residual/' + key for key in pose]
    elif mode == OBSERVED:
        names = ['observed/' + key for key in pose]
    else:
        names = pose
    return [side + '/' + key for side in sides for key in [*names, 'gripper']]


def execution_labels(values, mode):
    """Compose residuals only after denormalization; retain one gripper per arm."""
    values = np.asarray(values, dtype=float)
    width = arm_width(mode)
    if values.ndim < 1 or not values.shape[-1] or values.shape[-1] % width or not np.isfinite(values).all():
        raise ValueError(f'Expected finite action labels with {width} channels per arm')
    if mode != RESIDUAL:
        return values.copy()
    blocks = values.reshape(*values.shape[:-1], -1, width)
    result = np.concatenate((blocks[..., :6] + blocks[..., 6:12], blocks[..., 12:13]), axis=-1)
    return result.reshape(*values.shape[:-1], -1)
