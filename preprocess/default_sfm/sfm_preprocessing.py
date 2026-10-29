"""Shared offline/inference transforms for the confirmed default_sfm contract.

Math follows the archived csilab crop, angle, Delta and normalization functions.
Held labels follow FAFLOW's pinned command-latch implementation, not the newer
csilab next-observed-state labels. No robot drivers or model imports are needed.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

STATE_INDICES = (0, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18)
STATE_COLUMNS = ("gripper", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz")
CAMERAS = {"image_head": "head_cam_d455", "image_wrist_right": "wrist_cam_d405_a"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    value = np.asarray(value)
    require(value.dtype.kind in "biuf" and np.isfinite(value).all(), f"{name}: nonfinite or nonnumeric")
    return value


def sample(value, shape, name):
    value = finite(value, name)
    if value.shape == (1, *shape):
        value = value[0]
    require(value.shape == shape, f"{name}: expected {shape}, got {value.shape}")
    return value


def state_physical(state):
    state = finite(state, "raw state")
    require(state.shape[-1] == 19, "Expected raw state19")
    result = state[..., STATE_INDICES].astype(np.float64)
    upper = float(np.nextafter(np.float32(2 * np.pi), np.float32(0)))
    for index in (4, 6):
        result[..., index] = np.minimum(np.remainder(result[..., index], 2 * np.pi), upper)
    # csilab casts selected observations to float32 before fitting statistics.
    return result.astype(np.float32)


def initial_start(state, config):
    state = finite(state, "raw state").astype(np.float64)
    require(state.ndim == 2 and state.shape[1] == 19 and len(state), "Expected nonempty [T,19]")
    require(config["enabled"] and config["pose_keys"] == ["right/tcp_pose"]
            and config["gripper_keys"] == ["right/gripper_pose"], "Unsupported start-detection inputs")
    rotations = Rotation.from_euler("xyz", state[:, 7:10])
    moving = ((np.linalg.norm(state[:, 4:7] - state[0, 4:7], axis=1) > config["translation_threshold_m"])
              | ((rotations[0].inv() * rotations).magnitude() > config["rotation_threshold_rad"])
              | (np.abs(state[:, 0] - state[0, 0]) > config["gripper_threshold"]))
    run = 0
    for index, value in enumerate(moving):
        run = run + 1 if value else 0
        if run >= config["consecutive_frames"]:
            first = index - run + 1
            return max(0, first - config["pre_roll_frames"]), first
    if config["no_motion"] == "drop":
        return None, None
    if config["no_motion"] == "keep":
        return 0, None
    raise ValueError("No sustained motion")


def preprocess_image(value, camera, vision):
    image = np.asarray(value)
    if image.shape == (1, 480, 640, 3):
        image = image[0]
    require(image.shape == (480, 640, 3) and image.dtype == np.uint8, "Expected native 640x480 uint8 RGB")
    top, left, height, width = vision["crop_tlhw"][camera]
    require(top >= 0 and left >= 0 and top + height <= 480 and left + width <= 640, "Crop outside native image")
    cropped = image[top:top + height, left:left + width]
    out_h, out_w = vision["final_image_size"]
    scale = min(out_h / height, out_w / width)
    h, w = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y, x = (out_h - h) // 2, (out_w - w) // 2
    result[y:y + h, x:x + w] = resized
    return result


def held_targets(rows, state, control):
    """Reconstruct complete-episode commanded targets before any frame removal."""
    require(np.all((state[:, 0] >= -1e-6) & (state[:, 0] <= 1 + 1e-6)), "Unexpected gripper observation range")
    previous = float(state[0, 0]) * 2 - 1
    close = control.get("binary_close_threshold_m")
    open_ = control.get("binary_open_threshold_m")
    if close is None:
        close, open_ = -.5, .5
    require(open_ is not None and np.isfinite([close, open_]).all() and close < open_, "Bad gripper thresholds")
    labels = np.empty(len(rows), np.float64)
    percents = np.full(len(rows), np.nan, np.float64)
    present = np.zeros(len(rows), bool)
    fallback = 0
    for i, row in enumerate(rows):
        info = row.get("infos") or {}
        receipt = (info.get("command_receipts") or {}).get("right", info)
        percent = receipt.get("gripper_target_percent")
        if percent is not None:
            require(np.isfinite(percent) and 0 <= percent <= 100, "Invalid gripper target receipt")
            percents[i], present[i] = float(percent), True
            previous = -1 + float(percent) / 100 * 2
        else:
            require(control.get("binary_closed_width_m") is None, "Custom-width gripper requires receipts")
            raw = np.float32(row["actions"][6])
            if raw <= np.float32(close):
                previous = -1.
            elif raw >= np.float32(open_):
                previous = 1.
            fallback += 1
        labels[i] = previous
    # No quantization: unexpected intermediate targets require inspection.
    require(np.isin(labels, [-1., 1.]).all(), "Nonbinary held target: inspect receipt; do not round it")
    return labels, percents, present, fallback


def delta_actions(state, targets, held):
    state, targets = finite(state, "state"), finite(targets, "targets")
    require(state.shape == (len(targets), 19) and targets.shape[1:] == (7,), "Unexpected pose/action layout")
    result = np.empty((len(targets), 7), np.float64)
    result[:, :3] = targets[:, :3].astype(np.float64) - state[:, 4:7].astype(np.float64)
    result[:, 3:6] = (Rotation.from_euler("xyz", targets[:, 3:6])
                       * Rotation.from_euler("xyz", state[:, 7:10]).inv()).as_rotvec()
    result[:, 6] = held
    return result


def chunk_actions(actions, length=5):
    actions = finite(actions, "actions")
    indices = np.arange(len(actions))[:, None] + np.arange(length)
    mask = indices < len(actions)
    result = np.zeros((len(actions), length, actions.shape[-1]), np.float64)
    result[mask] = actions[indices[mask]]
    return result, mask


class Moments:
    """Population moments merged in float64, equivalent to csilab normalization."""
    def __init__(self):
        self.count = 0

    def add(self, values):
        x = finite(values, "normalization input").astype(np.float64).reshape(-1, values.shape[-1])
        if not len(x):
            return
        mean = x.mean(axis=0)
        m2 = ((x - mean) ** 2).sum(axis=0)
        low, high = x.min(axis=0), x.max(axis=0)
        if self.count == 0:
            self.mean, self.m2, self.low, self.high = mean, m2, low, high
        else:
            total = self.count + len(x)
            difference = mean - self.mean
            self.m2 += m2 + difference ** 2 * self.count * len(x) / total
            self.mean += difference * len(x) / total
            self.low, self.high = np.minimum(self.low, low), np.maximum(self.high, high)
        self.count += len(x)

    def result(self, method, epsilon, temperature=20.):
        require(self.count > 0 and epsilon > 0, "No normalization samples or invalid epsilon")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.))
        if method == "standard":
            offset, scale = self.mean, np.maximum(std, epsilon)
        elif method == "symlog_minmax":
            offset, scale = (self.low + self.high) / 2, np.maximum((self.high - self.low) / 2, epsilon)
        else:
            raise ValueError(method)
        return dict(method=method, epsilon=epsilon, temperature=temperature, count=self.count,
                    mean=self.mean.tolist(), std=std.tolist(), min=self.low.tolist(), max=self.high.tolist(),
                    offset=offset.tolist(), scale=scale.tolist())


def normalize(values, stats, inverse=False):
    x = finite(values, "normalization values").astype(np.float64)
    offset, scale = np.asarray(stats["offset"]), np.asarray(stats["scale"])
    require(x.shape[-1] == len(offset) and np.isfinite(scale).all() and np.all(scale > 0), "Invalid normalization shape/scale")
    if stats["method"] == "standard":
        result = x * scale + offset if inverse else (x - offset) / scale
    elif stats["method"] == "symlog_minmax":
        temperature = stats["temperature"]
        if inverse:
            result = np.sign(x) * np.expm1(np.abs(x) * np.log1p(temperature)) / temperature
            result = result * scale + offset
        else:
            x = (x - offset) / scale
            result = np.sign(x) * np.log1p(np.abs(x) * temperature) / np.log1p(temperature)
    else:
        raise ValueError(stats["method"])
    return finite(result, "normalization output")


def normalized_actions(values, mask, stats):
    result = np.asarray(values, dtype=np.float64).copy()
    result[..., :6] = normalize(result[..., :6], stats)
    result[~mask] = 0.
    return result.astype(np.float32)


def variant_spec(plan, variant_id):
    variant = next(v for v in plan["variants"] if v["id"] == variant_id)
    suffix = ("xy_vxy_remove" if variant.get("remove_linear_velocity_xy") else
              "xy_remove" if variant["remove_state_xy"] else "xy_keep")
    condition = next(c for c in plan["state"]["conditions"] if c["id"] == suffix)
    indices = [STATE_INDICES.index(i) for i in condition["retained_raw_indices"]]
    return dict(variant, state_condition=suffix, state_columns=indices,
                state_dim=len(indices), model_state_dim=len(indices) + (300 if variant["ft_method"] == "state_embedding" else 0),
                ft_already_normalized=True)


def select_variant(state, ft_history, specification):
    state = np.asarray(state)[..., specification["state_columns"]]
    ft_history = np.asarray(ft_history)
    require(ft_history.shape[-2:] == (50, 6), "Expected FT50x6")
    if specification["ft_method"] == "state_embedding":
        state = np.concatenate((state, ft_history.reshape(*ft_history.shape[:-2], 300)), axis=-1)
    return state.astype(np.float32)


def preprocess_observation(observation, plan, stats, variant_id):
    """Raw observation -> model inputs; also used for offline/inference parity."""
    raw_state = sample(observation["state"], (19,), "state")
    raw_ft = sample(observation["right_force_history"], (50, 6), "FT")
    state = normalize(state_physical(raw_state), stats["state"]).astype(np.float32)
    ft = normalize(raw_ft, stats["ft"]).astype(np.float32)
    spec = variant_spec(plan, variant_id)
    result = {"state": select_variant(state, ft, spec),
              **{key: preprocess_image(observation[camera], camera, plan["vision"]) for key, camera in CAMERAS.items()}}
    if spec["ft_method"] != "state_embedding":
        result["ft_history"] = ft
    return result
