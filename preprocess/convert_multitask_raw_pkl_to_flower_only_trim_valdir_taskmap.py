"""
Convert a folder-organized multi-task raw robot pickle dataset into a
FLOWER / CALVIN-style disk dataset.

Expected input layout:

    INPUT_ROOT/
      multi_usb/
        2026-04-15/
          *.pkl
      multi_lan/
        2026-04-15/
          *.pkl
      ...

Optional separate validation input layout:

    VAL_INPUT_ROOT/
      multi_usb/
        2026-04-20/
          *.pkl
      multi_val_usb_p1/
        2026-05-18/
          *.pkl
      multi_val_usb_p2/
        2026-05-15/
          *.pkl
      multi_val_usb_unseen/
        *.pkl
      ...

Validation folder names such as multi_val_usb_p1, multi_val_usb_p2, and
multi_val_usb_unseen are automatically mapped to the canonical task name
multi_usb for language annotations and per-task summaries.

When --val_input / --validation_input is provided, --input is written to
training/ and --val_input is written to validation/. In that mode --val_ratio
is ignored.

The generated dataset layout is:

    OUTPUT_ROOT/
      training/
        episode_0000000.npz
        ...
        ep_start_end_ids.npy
        lang_clip_resnet50/
          auto_lang_ann.npy
      validation/
        episode_0000000.npz
        ...
        ep_start_end_ids.npy
        lang_clip_resnet50/
          auto_lang_ann.npy

Each episode_XXXXXXX.npz stores a single timestep with keys expected by the
FLOWER disk loader:
    - rgb_static
    - rgb_gripper
    - robot_obs
    - rel_actions
    - scene_obs
    - right_force_history (optional; [T, 6] force/torque history)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


LOGGER = logging.getLogger("convert_multitask_raw_pkl_to_flower")

# [MOD] Action relabeling is disabled.
# Preserve every raw action value exactly as it appears in the pickle files.
# The previous gripper relabeling constants/helpers were removed because they
# overwrote the raw gripper command from the observed gripper state.


TASK_PROMPTS = {
    "multi_usb": "Insert USB connector into USB port",
    "multi_rotary_switch": "Grasp rotary switch and rotate clockwise",
    "multi_lan": "Insert LAN cable into Ethernet port",
    "multi_outlet": "Insert plug into outlet",
    "multi_knob": "Grasp knob and rotate clockwise",
    "multi_bnc": "Insert BNC connector and rotate clockwise",
    "multi_hdmi": "Insert HDMI connector into HDMI port",
    "multi_bar_latch": "Grasp bar latch and rotate clockwise",
    "multi_audiojack": "Insert audio jack into audio port",
    "multi_key_lock": "Insert key and rotate clockwise",
}


def canonicalize_task_folder_name(folder_name: str, task_prompts: Optional[Dict[str, str]] = None) -> str:
    """Map validation-condition folder names to canonical FLOWER task names.

    Examples:
        multi_val_usb_p1 -> multi_usb
        multi_val_usb_p2 -> multi_usb
        multi_val_usb_unseen -> multi_usb
        multi_val_bar_latch_p1 -> multi_bar_latch
        multi_usb -> multi_usb

    The function is conservative: if a folder cannot be mapped to a known task,
    the original folder name is returned so existing behavior is preserved.
    """
    known_tasks = task_prompts or TASK_PROMPTS
    if folder_name in known_tasks:
        return folder_name

    # Remove validation split prefix used by the provided directory layout.
    candidate = folder_name
    if candidate.startswith("multi_val_"):
        candidate = "multi_" + candidate[len("multi_val_"):]

    # Remove condition suffixes such as _p1, _p2, ... and _unseen.
    candidate = re.sub(r"_(?:p\d+|unseen)$", "", candidate)
    if candidate in known_tasks:
        return candidate

    # Fallback: find the longest known task that prefixes the candidate.
    # This handles names such as multi_usb_p1 even without the multi_val_ prefix.
    for task_name in sorted(known_tasks.keys(), key=len, reverse=True):
        if candidate == task_name or candidate.startswith(task_name + "_"):
            return task_name

    return folder_name


@dataclass
class EpisodeRecord:
    steps: List[Dict[str, Any]]
    source_name: str
    source_episode_index: int
    task_name: str
    raw_task_folder: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Root directory containing task folders with pickle files.",
    )
    parser.add_argument(
        "--val_input",
        "--validation_input",
        dest="val_input",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing validation task folders. "
            "If provided, --input is used only for training and --val_ratio is ignored."
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=Path,
        help="Output dataset root. It will contain training/ and validation/.",
    )
    parser.add_argument(
        "--task_names",
        nargs="*",
        default=None,
        help="Optional subset of task folder names to include.",
    )
    parser.add_argument(
        "--pkl_glob",
        default="**/*.pkl",
        help="Glob pattern used inside each task folder.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio at episode level, applied per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/validation split.",
    )
    parser.add_argument(
        "--min_episode_steps",
        type=int,
        default=11,
        help="Episodes shorter than this are skipped.",
    )
    parser.add_argument(
        "--zero_action_trailing_keep",
        type=int,
        default=5,
        help="Number of trailing steps to keep after the final non-zero action when trimming zero-action padding.",
    )
    parser.add_argument(
        "--task_name",
        default="custom_task",
        help="Fallback task label stored when a folder name is not found in TASK_PROMPTS.",
    )
    parser.add_argument(
        "--default_instruction",
        default=None,
        help="Fallback instruction used when a task-specific instruction is not found.",
    )
    parser.add_argument(
        "--instructions_json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with per-episode language annotations. "
            "Supported keys: task_name, source_name, source_name#episode_idx, or global episode index string."
        ),
    )
    parser.add_argument(
        "--task_prompts_json",
        type=Path,
        default=None,
        help="Optional JSON file overriding or extending TASK_PROMPTS.",
    )
    parser.add_argument(
        "--lang_folder",
        default="lang_clip_resnet50",
        help="Language annotation folder name expected by the FLOWER config.",
    )
    parser.add_argument(
        "--obs_state_key",
        default="state",
        help="Observation key containing low-dimensional state.",
    )
    parser.add_argument(
        "--static_cam_key",
        default="front_cam",
        help="Observation key used as rgb_static.",
    )
    parser.add_argument(
        "--gripper_cam_key",
        default="right_wrist_cam",
        help="Observation key used as rgb_gripper.",
    )
    parser.add_argument(
        "--action_source",
        choices=["raw", "state_delta"],
        default="raw",
        help=(
            "How to build rel_actions. "
            "'raw' keeps the sliced right-arm action, 'state_delta' derives xyz/rpy deltas from obs->next_obs."
        ),
    )
    parser.add_argument(
        "--prefer_intervene_action",
        action="store_true",
        help="Use infos['intervene_action'] when available instead of the raw action.",
    )
    parser.add_argument(
        "--action_slice",
        default="7:14",
        help="Slice selecting the right-arm action from the raw action vector.",
    )
    parser.add_argument(
        "--state_pose_slice",
        default="23:29",
        help="Slice selecting right-arm tcp pose from the state vector.",
    )
    parser.add_argument(
        "--state_gripper_index",
        type=int,
        default=19,
        help="Index selecting right gripper pose from the state vector.",
    )
    parser.add_argument(
        "--include_right_ft",
        action="store_true",
        help="Append right-arm force(3) and torque(3) from the state vector to robot_obs.",
    )
    parser.add_argument(
        "--state_force_slice",
        default="20:23",
        help="Slice selecting right-arm tcp force from the state vector.",
    )
    parser.add_argument(
        "--state_torque_slice",
        default="29:32",
        help="Slice selecting right-arm tcp torque from the state vector.",
    )
    parser.add_argument(
        "--include_right_force_history",
        action="store_true",
        help=(
            "Store observations[right_force_history] into each timestep npz as "
            "'right_force_history'. Expected raw shape is [1,T,6] or [T,6]."
        ),
    )
    parser.add_argument(
        "--right_force_history_key",
        default="right_force_history",
        help="Observation key containing right-arm force/torque history.",
    )
    parser.add_argument(
        "--scene_obs_dim",
        type=int,
        default=1,
        help="Length of the dummy scene_obs vector stored per timestep.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def parse_slice(text: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected slice in start:end form, got: {text}")
    return slice(int(parts[0]), int(parts[1]))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def to_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def squeeze_leading_singletons(array: np.ndarray) -> np.ndarray:
    while array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


# [MOD] Gripper/action relabeling helpers were intentionally removed.
# From this point onward, conversion may trim zero-action padding, but it must
# not rewrite step["actions"] based on robot state.


def trim_zero_action_episode_steps(
    episode: Sequence[Dict[str, Any]],
    trailing_keep: int = 5,
) -> Optional[List[Dict[str, Any]]]:
    if not episode:
        return None

    actions = [to_numpy(step.get("actions")) for step in episode]
    total_steps = len(actions)

    start_idx = 0
    while start_idx < total_steps and np.all(actions[start_idx] == 0):
        start_idx += 1

    if start_idx == total_steps:
        return None

    end_idx = total_steps - 1
    while end_idx >= 0 and np.all(actions[end_idx] == 0):
        end_idx -= 1

    # [MOD] Keep only zero-action padding trimming.
    # Action relabeling is disabled, so returned steps keep their original action values.
    trim_end = min(total_steps, end_idx + 1 + trailing_keep)
    return list(episode[start_idx:trim_end])


def normalize_image(image_like: Any) -> np.ndarray:
    image = squeeze_leading_singletons(to_numpy(image_like))
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeeze, got shape {image.shape}")

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def resolve_obs_key(obs_dict: Dict[str, Any], requested_key: str, fallback_keys: Sequence[str]) -> str:
    if requested_key in obs_dict:
        return requested_key

    for key in fallback_keys:
        if key in obs_dict:
            LOGGER.warning("Observation key '%s' not found. Falling back to '%s'.", requested_key, key)
            return key

    available = ", ".join(sorted(obs_dict.keys()))
    raise KeyError(f"Observation key '{requested_key}' not found. Available keys: {available}")


def recursive_index(data: Any, index: int) -> Any:
    if isinstance(data, dict):
        return {key: recursive_index(value, index) for key, value in data.items()}
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return data.item()
        return data[index]
    if isinstance(data, (list, tuple)):
        return data[index]
    return data


def is_transition_dict(data: Any) -> bool:
    return isinstance(data, dict) and {
        "observations",
        "actions",
        "next_observations",
        "dones",
    }.issubset(data.keys())


def is_single_step_transition(data: Any) -> bool:
    if not is_transition_dict(data):
        return False
    action = to_numpy(data["actions"])
    done = to_numpy(data["dones"])
    return action.ndim == 1 and done.ndim == 0


def is_sequence_transition(data: Any) -> bool:
    if not is_transition_dict(data):
        return False
    action = to_numpy(data["actions"])
    if action.ndim < 1:
        return False
    if action.ndim == 1:
        done = to_numpy(data["dones"])
        return done.ndim > 0 and action.shape[0] != 14
    return True


def group_step_transitions(step_transitions: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for step in step_transitions:
        current.append(step)
        if bool(np.asarray(step["dones"]).item()):
            episodes.append(current)
            current = []

    if current:
        LOGGER.warning("Final episode stream did not terminate with done=True. Keeping the trailing partial episode.")
        episodes.append(current)

    return episodes


def split_sequence_transition(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    actions = to_numpy(data["actions"])
    num_steps = int(actions.shape[0])
    dones = to_numpy(data["dones"]).astype(bool).reshape(-1)
    if dones.shape[0] != num_steps:
        raise ValueError(f"dones length {dones.shape[0]} does not match actions length {num_steps}")

    end_indices = [idx + 1 for idx, done in enumerate(dones) if done]
    if not end_indices or end_indices[-1] != num_steps:
        end_indices.append(num_steps)

    episodes: List[List[Dict[str, Any]]] = []
    start = 0
    for end in end_indices:
        if end <= start:
            continue
        episodes.append([recursive_index(data, idx) for idx in range(start, end)])
        start = end
    return episodes


def extract_episodes_from_object(data: Any) -> List[List[Dict[str, Any]]]:
    if is_single_step_transition(data):
        return [[data]]

    if is_sequence_transition(data):
        return split_sequence_transition(data)

    if isinstance(data, dict) and "episodes" in data:
        return extract_episodes_from_object(data["episodes"])

    if isinstance(data, (list, tuple)):
        if data and all(is_single_step_transition(item) for item in data):
            return group_step_transitions(list(data))

        episodes: List[List[Dict[str, Any]]] = []
        for item in data:
            episodes.extend(extract_episodes_from_object(item))
        return episodes

    raise ValueError(f"Unsupported pickle payload type: {type(data)}")


def load_task_prompts(path: Path | None) -> Dict[str, str]:
    prompts = dict(TASK_PROMPTS)
    if path is None:
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    prompts.update(loaded)
    return prompts


def load_episode_records(
    input_path: Path,
    pkl_glob: str,
    task_names: Sequence[str] | None,
    state_key: str,
    zero_action_trailing_keep: int,
    task_prompts: Optional[Dict[str, str]] = None,
) -> List[EpisodeRecord]:
    if input_path.is_file():
        raise ValueError("Multi-task conversion expects --input to be a root directory containing task folders.")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    all_task_dirs = sorted([path for path in input_path.iterdir() if path.is_dir()])
    if task_names is None:
        task_dirs = all_task_dirs
    else:
        requested = set(task_names)
        task_dirs = [
            path
            for path in all_task_dirs
            if path.name in requested
            or canonicalize_task_folder_name(path.name, task_prompts) in requested
        ]
        missing = sorted(requested - {path.name for path in task_dirs} - {canonicalize_task_folder_name(path.name, task_prompts) for path in task_dirs})
        if missing:
            raise FileNotFoundError(
                f"Task folders not found in {input_path}: {missing}. "
                "For validation layouts like multi_val_usb_p1, you may request the canonical task name, e.g. multi_usb."
            )

    if not task_dirs:
        raise FileNotFoundError(f"No task folders found in {input_path}")

    records: List[EpisodeRecord] = []

    # [MOD] state_key is kept in the function signature for call-site compatibility,
    # but action relabeling from state is disabled and this function no longer mutates actions.

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task folder not found: {task_dir}")

        raw_task_folder = task_dir.name
        task_name = canonicalize_task_folder_name(raw_task_folder, task_prompts)
        if task_name != raw_task_folder:
            LOGGER.info("Mapped task folder '%s' -> canonical task '%s'.", raw_task_folder, task_name)
        pkl_files = sorted(task_dir.glob(pkl_glob))
        if not pkl_files:
            LOGGER.warning("No pickle files matched %s in %s", pkl_glob, task_dir)
            continue

        first_payload = load_pickle(pkl_files[0])
        if is_single_step_transition(first_payload):
            LOGGER.info("Detected step-level pickle stream for task %s.", task_name)
            step_stream = [first_payload]
            for path in pkl_files[1:]:
                step_stream.append(load_pickle(path))
            episodes = group_step_transitions(step_stream)
            trimmed_episodes: List[List[Dict[str, Any]]] = []
            dropped_episode_count = 0
            trimmed_step_count = 0

            for steps in episodes:
                # [MOD] Trim zero-action padding only. Do not relabel or overwrite action values.
                trimmed_steps = trim_zero_action_episode_steps(steps, trailing_keep=zero_action_trailing_keep)
                if trimmed_steps is None:
                    dropped_episode_count += 1
                    continue
                trimmed_step_count += len(steps) - len(trimmed_steps)
                trimmed_episodes.append(trimmed_steps)

            episodes = trimmed_episodes
            LOGGER.info(
                "Trimmed %d zero-action steps and dropped %d all-zero episodes for task %s. Action relabeling disabled.",
                trimmed_step_count,
                dropped_episode_count,
                task_name,
            )
            records.extend(
                EpisodeRecord(
                    steps=steps,
                    source_name=f"{raw_task_folder}::step_stream",
                    source_episode_index=idx,
                    task_name=task_name,
                    raw_task_folder=raw_task_folder,
                )
                for idx, steps in enumerate(episodes)
            )
            continue

        task_trimmed_step_count = 0
        task_dropped_episode_count = 0
        for path in tqdm(pkl_files, desc=f"Loading {task_name} pickle files"):
            payload = load_pickle(path)
            episodes = extract_episodes_from_object(payload)
            for idx, steps in enumerate(episodes):
                # [MOD] Trim zero-action padding only. Do not relabel or overwrite action values.
                trimmed_steps = trim_zero_action_episode_steps(steps, trailing_keep=zero_action_trailing_keep)
                if trimmed_steps is None:
                    task_dropped_episode_count += 1
                    continue
                task_trimmed_step_count += len(steps) - len(trimmed_steps)
                records.append(
                    EpisodeRecord(
                        steps=trimmed_steps,
                        source_name=path.stem,
                        source_episode_index=idx,
                        task_name=task_name,
                        raw_task_folder=raw_task_folder,
                    )
                )
        LOGGER.info(
            "Trimmed %d zero-action steps and dropped %d all-zero episodes for task %s. Action relabeling disabled.",
            task_trimmed_step_count,
            task_dropped_episode_count,
            task_name,
        )

    return records


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def build_robot_obs(
    step: Dict[str, Any],
    state_key: str,
    pose_slice: slice,
    gripper_index: int,
    *,
    include_right_ft: bool,
    force_slice: slice,
    torque_slice: slice,
) -> np.ndarray:
    state = squeeze_leading_singletons(to_numpy(step["observations"][state_key], dtype=np.float32))
    pose = state[pose_slice]
    gripper = np.asarray([state[gripper_index]], dtype=np.float32)
    pieces = [pose, gripper]
    if include_right_ft:
        force = state[force_slice]
        torque = state[torque_slice]
        pieces.extend([force, torque])

    robot_obs = np.concatenate(pieces, axis=0)
    expected_dim = 13 if include_right_ft else 7
    if robot_obs.shape != (expected_dim,):
        raise ValueError(f"Expected robot_obs with shape ({expected_dim},), got {robot_obs.shape}")
    return robot_obs


def select_action_vector(step: Dict[str, Any], action_slice: slice, prefer_intervene_action: bool) -> np.ndarray:
    action_source = step["actions"]
    info = step.get("infos", {})
    if prefer_intervene_action and isinstance(info, dict) and "intervene_action" in info:
        action_source = info["intervene_action"]

    action = squeeze_leading_singletons(to_numpy(action_source, dtype=np.float32))
    action = action[action_slice]
    if action.shape != (7,):
        raise ValueError(f"Expected right-arm action with shape (7,), got {action.shape}")
    return action


def build_rel_action(
    step: Dict[str, Any],
    *,
    action_source: str,
    state_key: str,
    pose_slice: slice,
    action_slice: slice,
    prefer_intervene_action: bool,
) -> np.ndarray:
    raw_action = select_action_vector(step, action_slice, prefer_intervene_action)
    if action_source == "raw":
        return raw_action

    current_state = squeeze_leading_singletons(to_numpy(step["observations"][state_key], dtype=np.float32))
    next_state = squeeze_leading_singletons(to_numpy(step["next_observations"][state_key], dtype=np.float32))

    current_pose = current_state[pose_slice]
    next_pose = next_state[pose_slice]

    delta_pos = next_pose[:3] - current_pose[:3]
    delta_rot = wrap_to_pi(next_pose[3:6] - current_pose[3:6])
    gripper = raw_action[-1:]
    rel_action = np.concatenate([delta_pos, delta_rot, gripper], axis=0).astype(np.float32)
    if rel_action.shape != (7,):
        raise ValueError(f"Expected rel_action with shape (7,), got {rel_action.shape}")
    return rel_action


def build_right_force_history(step: Dict[str, Any], history_key: str) -> np.ndarray:
    """Extract per-timestep right force/torque history as [T, 6].

    The raw robot pickle stores 100 Hz F/T history under observations with a
    leading singleton batch dimension, typically [1, 10, 6]. The FLOWER disk
    dataset should save one timestep's history as [T, 6]; ExtendedDiskDataset
    later stacks obs frames into [obs_seq_len, T, 6].
    """
    observations = step["observations"]
    resolved_key = resolve_obs_key(
        observations,
        history_key,
        ("right_force_history", "right_ft_history", "force_history"),
    )
    history = squeeze_leading_singletons(to_numpy(observations[resolved_key], dtype=np.float32))

    if history.ndim == 1:
        if history.size % 6 != 0:
            raise ValueError(
                f"{resolved_key} flat history length must be divisible by 6, got {history.shape}"
            )
        history = history.reshape(-1, 6)

    if history.ndim != 2 or history.shape[-1] != 6:
        raise ValueError(
            f"{resolved_key} should have shape [T,6] after squeeze, got {history.shape}"
        )

    return history.astype(np.float32)


def load_instruction_mapping(path: Path | None) -> Any:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_instruction(
    record: EpisodeRecord,
    global_episode_index: int,
    mapping: Any,
    default_instruction: str,
    default_task_name: str,
    task_prompts: Dict[str, str],
) -> Tuple[str, str]:
    task_name = record.task_name
    task_instruction = task_prompts.get(task_name, default_instruction)
    task_label = task_name if task_name in task_prompts else default_task_name

    if mapping is None:
        return task_instruction, task_label

    if isinstance(mapping, list):
        if global_episode_index < len(mapping):
            value = mapping[global_episode_index]
        else:
            return task_instruction, task_label
    elif isinstance(mapping, dict):
        candidates = [
            f"{record.source_name}#{record.source_episode_index}",
            record.source_name,
            task_name,
            str(global_episode_index),
            "default",
        ]
        value = None
        for key in candidates:
            if key in mapping:
                value = mapping[key]
                break
        if value is None:
            return task_instruction, task_label
    else:
        raise ValueError("instructions_json must contain either a JSON list or dict.")

    if isinstance(value, str):
        return value, task_label

    if isinstance(value, dict):
        instruction = value.get("instruction", task_instruction)
        resolved_task_name = value.get("task", task_label)
        return instruction, resolved_task_name

    raise ValueError(f"Unsupported instruction mapping entry: {value!r}")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if any(output_root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output root {output_root} is not empty. Use --overwrite to allow writing into it."
            )
    output_root.mkdir(parents=True, exist_ok=True)


def save_timestep_npz(
    destination: Path,
    step: Dict[str, Any],
    *,
    state_key: str,
    static_cam_key: str,
    gripper_cam_key: str,
    pose_slice: slice,
    gripper_index: int,
    include_right_ft: bool,
    force_slice: slice,
    torque_slice: slice,
    include_right_force_history: bool,
    right_force_history_key: str,
    action_source: str,
    action_slice: slice,
    prefer_intervene_action: bool,
    scene_obs_dim: int,
) -> None:
    observations = step["observations"]
    resolved_static_cam_key = resolve_obs_key(
        observations,
        static_cam_key,
        ("front_cam", "rgb_static", "head_cam", "static_cam", "left/head_cam"),
    )
    resolved_gripper_cam_key = resolve_obs_key(
        observations,
        gripper_cam_key,
        ("right_wrist_cam", "rgb_gripper", "wrist_cam", "right/wrist_cam"),
    )

    rgb_static = normalize_image(observations[resolved_static_cam_key])
    rgb_gripper = normalize_image(observations[resolved_gripper_cam_key])
    robot_obs = build_robot_obs(
        step,
        state_key,
        pose_slice,
        gripper_index,
        include_right_ft=include_right_ft,
        force_slice=force_slice,
        torque_slice=torque_slice,
    )
    rel_actions = build_rel_action(
        step,
        action_source=action_source,
        state_key=state_key,
        pose_slice=pose_slice,
        action_slice=action_slice,
        prefer_intervene_action=prefer_intervene_action,
    )
    scene_obs = np.zeros((scene_obs_dim,), dtype=np.float32)

    payload = {
        "rgb_static": rgb_static,
        "rgb_gripper": rgb_gripper,
        "robot_obs": robot_obs.astype(np.float32),
        "rel_actions": rel_actions.astype(np.float32),
        "scene_obs": scene_obs,
    }
    if include_right_force_history:
        payload["right_force_history"] = build_right_force_history(step, right_force_history_key)

    np.savez_compressed(destination, **payload)


def write_split(
    split_name: str,
    records: Sequence[Tuple[int, EpisodeRecord]],
    output_root: Path,
    *,
    lang_folder: str,
    default_instruction: str,
    default_task_name: str,
    instruction_mapping: Any,
    task_prompts: Dict[str, str],
    state_key: str,
    static_cam_key: str,
    gripper_cam_key: str,
    pose_slice: slice,
    gripper_index: int,
    include_right_ft: bool,
    force_slice: slice,
    torque_slice: slice,
    include_right_force_history: bool,
    right_force_history_key: str,
    action_source: str,
    action_slice: slice,
    prefer_intervene_action: bool,
    scene_obs_dim: int,
    min_episode_steps: int,
) -> Dict[str, Any]:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    lang_dir = split_dir / lang_folder
    lang_dir.mkdir(parents=True, exist_ok=True)

    ep_start_end_ids: List[Tuple[int, int]] = []
    lang_ann: List[str] = []
    lang_tasks: List[str] = []
    lang_indx: List[Tuple[int, int]] = []
    metadata: List[Dict[str, Any]] = []
    per_task_episode_count: Dict[str, int] = {}

    global_step_index = 0
    saved_episode_count = 0
    skipped_episode_count = 0

    for global_episode_index, record in tqdm(records, desc=f"Writing {split_name} split"):
        if len(record.steps) < min_episode_steps:
            skipped_episode_count += 1
            continue

        start = global_step_index
        end = start + len(record.steps)

        instruction, task_name = resolve_instruction(
            record,
            global_episode_index,
            instruction_mapping,
            default_instruction,
            default_task_name,
            task_prompts,
        )

        for step in record.steps:
            timestep_path = split_dir / f"episode_{global_step_index:07d}.npz"
            save_timestep_npz(
                timestep_path,
                step,
                state_key=state_key,
                static_cam_key=static_cam_key,
                gripper_cam_key=gripper_cam_key,
                pose_slice=pose_slice,
                gripper_index=gripper_index,
                include_right_ft=include_right_ft,
                force_slice=force_slice,
                torque_slice=torque_slice,
                include_right_force_history=include_right_force_history,
                right_force_history_key=right_force_history_key,
                action_source=action_source,
                action_slice=action_slice,
                prefer_intervene_action=prefer_intervene_action,
                scene_obs_dim=scene_obs_dim,
            )
            global_step_index += 1

        ep_start_end_ids.append((start, end))
        lang_indx.append((start, end))
        lang_ann.append(instruction)
        lang_tasks.append(task_name)
        per_task_episode_count[task_name] = per_task_episode_count.get(task_name, 0) + 1
        metadata.append(
            {
                "generated_episode_index": saved_episode_count,
                "global_episode_index": global_episode_index,
                "source_name": record.source_name,
                "source_episode_index": record.source_episode_index,
                "task_name": task_name,
                "raw_task_folder": record.raw_task_folder,
                "num_steps": len(record.steps),
                "instruction": instruction,
                "start_step_index": start,
                "end_step_index": end,
            }
        )
        saved_episode_count += 1

    ep_start_end_array = np.asarray(ep_start_end_ids, dtype=np.int64).reshape(-1, 2)
    np.save(split_dir / "ep_start_end_ids.npy", ep_start_end_array)

    auto_lang_ann = {
        "language": {
            "ann": lang_ann,
            "task": lang_tasks,
            "emb": np.zeros((len(lang_ann), 1, 1), dtype=np.float32),
        },
        "info": {
            "episodes": [entry["generated_episode_index"] for entry in metadata],
            "indx": lang_indx,
        },
    }
    np.save(lang_dir / "auto_lang_ann.npy", auto_lang_ann)

    summary = {
        "split": split_name,
        "saved_episode_count": saved_episode_count,
        "skipped_episode_count": skipped_episode_count,
        "saved_step_count": global_step_index,
        "lang_folder": lang_folder,
        "robot_obs_dim": 13 if include_right_ft else 7,
        "include_right_force_history": include_right_force_history,
        "right_force_history_key": right_force_history_key if include_right_force_history else None,
        "per_task_episode_count": per_task_episode_count,
        "metadata": metadata,
    }
    with (split_dir / "conversion_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def filter_records_by_min_steps(
    records: Sequence[EpisodeRecord],
    min_episode_steps: int,
    split_label: str,
) -> Tuple[List[EpisodeRecord], int, Dict[str, int]]:
    """Filter short episodes and report per-task counts for one source split."""
    short_episode_count = sum(1 for record in records if len(record.steps) < min_episode_steps)
    filtered_records = [record for record in records if len(record.steps) >= min_episode_steps]

    LOGGER.info(
        "[%s] Keeping %d episodes with at least %d steps. Skipped %d short episodes.",
        split_label,
        len(filtered_records),
        min_episode_steps,
        short_episode_count,
    )

    per_task_counts: Dict[str, int] = {}
    for record in filtered_records:
        per_task_counts[record.task_name] = per_task_counts.get(record.task_name, 0) + 1
    LOGGER.info("[%s] Task episode counts after filtering: %s", split_label, per_task_counts)

    return filtered_records, short_episode_count, per_task_counts


def merge_count_dicts(*count_dicts: Dict[str, int]) -> Dict[str, int]:
    """Merge per-task count dictionaries for conversion_summary.json."""
    merged: Dict[str, int] = {}
    for count_dict in count_dicts:
        for key, value in count_dict.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def split_records(
    records: Sequence[EpisodeRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, EpisodeRecord]], List[Tuple[int, EpisodeRecord]]]:
    if not records:
        return [], []

    grouped: Dict[str, List[Tuple[int, EpisodeRecord]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record.task_name, []).append((index, record))

    rng = random.Random(seed)
    train_records: List[Tuple[int, EpisodeRecord]] = []
    val_records: List[Tuple[int, EpisodeRecord]] = []

    for task_name in sorted(grouped.keys()):
        task_records = grouped[task_name][:]
        rng.shuffle(task_records)
        val_count = int(round(len(task_records) * val_ratio))
        val_count = max(0, min(len(task_records), val_count))

        val_records.extend(task_records[:val_count])
        train_records.extend(task_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val_ratio must satisfy 0.0 <= val_ratio < 1.0")
    if args.zero_action_trailing_keep < 0:
        raise ValueError("--zero_action_trailing_keep must be >= 0")

    prepare_output_root(args.output_root, args.overwrite)

    pose_slice = parse_slice(args.state_pose_slice)
    force_slice = parse_slice(args.state_force_slice)
    torque_slice = parse_slice(args.state_torque_slice)
    action_slice = parse_slice(args.action_slice)
    default_instruction = args.default_instruction or args.task_name.replace("_", " ")
    instruction_mapping = load_instruction_mapping(args.instructions_json)
    task_prompts = load_task_prompts(args.task_prompts_json)

    raw_train_records = load_episode_records(
        args.input,
        args.pkl_glob,
        args.task_names,
        args.obs_state_key,
        args.zero_action_trailing_keep,
        task_prompts,
    )
    LOGGER.info("Loaded %d raw training episodes before filtering from %s.", len(raw_train_records), args.input)

    train_source_records, train_short_episode_count, train_per_task_raw_counts = filter_records_by_min_steps(
        raw_train_records,
        args.min_episode_steps,
        "training input",
    )

    if len(train_source_records) < 1:
        raise ValueError("Need at least 1 training episode after filtering.")

    raw_val_records: List[EpisodeRecord] = []
    val_source_records: List[EpisodeRecord] = []
    val_short_episode_count = 0
    val_per_task_raw_counts: Dict[str, int] = {}

    if args.val_input is not None:
        if args.val_input.resolve() == args.input.resolve():
            raise ValueError("--val_input must be different from --input when using a separate validation directory.")

        LOGGER.info(
            "Using separate validation input directory %s. --val_ratio=%s will be ignored.",
            args.val_input,
            args.val_ratio,
        )
        raw_val_records = load_episode_records(
            args.val_input,
            args.pkl_glob,
            args.task_names,
            args.obs_state_key,
            args.zero_action_trailing_keep,
            task_prompts,
        )
        LOGGER.info("Loaded %d raw validation episodes before filtering from %s.", len(raw_val_records), args.val_input)

        val_source_records, val_short_episode_count, val_per_task_raw_counts = filter_records_by_min_steps(
            raw_val_records,
            args.min_episode_steps,
            "validation input",
        )

        if len(val_source_records) < 1:
            raise ValueError("Need at least 1 validation episode after filtering.")

        train_records = list(enumerate(train_source_records))
        val_records = list(enumerate(val_source_records))
        split_mode = "separate_validation_input"
    else:
        train_records, val_records = split_records(train_source_records, args.val_ratio, args.seed)
        split_mode = "val_ratio"
        LOGGER.info("Episode split by val_ratio=%s: %d train / %d validation", args.val_ratio, len(train_records), len(val_records))

    per_task_raw_counts = merge_count_dicts(train_per_task_raw_counts, val_per_task_raw_counts)
    LOGGER.info("Final conversion split mode: %s. Episode split: %d train / %d validation", split_mode, len(train_records), len(val_records))

    train_summary = write_split(
        "training",
        train_records,
        args.output_root,
        lang_folder=args.lang_folder,
        default_instruction=default_instruction,
        default_task_name=args.task_name,
        instruction_mapping=instruction_mapping,
        task_prompts=task_prompts,
        state_key=args.obs_state_key,
        static_cam_key=args.static_cam_key,
        gripper_cam_key=args.gripper_cam_key,
        pose_slice=pose_slice,
        gripper_index=args.state_gripper_index,
        include_right_ft=args.include_right_ft,
        force_slice=force_slice,
        torque_slice=torque_slice,
        include_right_force_history=args.include_right_force_history,
        right_force_history_key=args.right_force_history_key,
        action_source=args.action_source,
        action_slice=action_slice,
        prefer_intervene_action=args.prefer_intervene_action,
        scene_obs_dim=args.scene_obs_dim,
        min_episode_steps=args.min_episode_steps,
    )
    val_summary = write_split(
        "validation",
        val_records,
        args.output_root,
        lang_folder=args.lang_folder,
        default_instruction=default_instruction,
        default_task_name=args.task_name,
        instruction_mapping=instruction_mapping,
        task_prompts=task_prompts,
        state_key=args.obs_state_key,
        static_cam_key=args.static_cam_key,
        gripper_cam_key=args.gripper_cam_key,
        pose_slice=pose_slice,
        gripper_index=args.state_gripper_index,
        include_right_ft=args.include_right_ft,
        force_slice=force_slice,
        torque_slice=torque_slice,
        include_right_force_history=args.include_right_force_history,
        right_force_history_key=args.right_force_history_key,
        action_source=args.action_source,
        action_slice=action_slice,
        prefer_intervene_action=args.prefer_intervene_action,
        scene_obs_dim=args.scene_obs_dim,
        min_episode_steps=args.min_episode_steps,
    )

    overall_summary = {
        "input": str(args.input),
        "val_input": str(args.val_input) if args.val_input is not None else None,
        "split_mode": split_mode,
        "output_root": str(args.output_root),
        "task_name": args.task_name,
        "default_instruction": default_instruction,
        "robot_obs_dim": 13 if args.include_right_ft else 7,
        "include_right_ft": args.include_right_ft,
        "include_right_force_history": args.include_right_force_history,
        "right_force_history_key": args.right_force_history_key if args.include_right_force_history else None,
        "zero_action_trailing_keep": args.zero_action_trailing_keep,
        "num_raw_episodes": len(raw_train_records) + len(raw_val_records),
        "num_used_episodes": len(train_source_records) + len(val_source_records),
        "num_skipped_short_episodes": train_short_episode_count + val_short_episode_count,
        "task_prompts": task_prompts,
        "task_episode_count": per_task_raw_counts,
        "train_source": {
            "input": str(args.input),
            "raw_episode_count": len(raw_train_records),
            "used_episode_count": len(train_source_records),
            "skipped_short_episode_count": train_short_episode_count,
            "per_task_episode_count": train_per_task_raw_counts,
        },
        "validation_source": {
            "input": str(args.val_input) if args.val_input is not None else None,
            "raw_episode_count": len(raw_val_records),
            "used_episode_count": len(val_source_records),
            "skipped_short_episode_count": val_short_episode_count,
            "per_task_episode_count": val_per_task_raw_counts,
        },
        "train_split": {
            "episode_count": train_summary["saved_episode_count"],
            "step_count": train_summary["saved_step_count"],
            "skipped_episode_count": train_summary["skipped_episode_count"],
            "per_task_episode_count": train_summary["per_task_episode_count"],
        },
        "validation_split": {
            "episode_count": val_summary["saved_episode_count"],
            "step_count": val_summary["saved_step_count"],
            "skipped_episode_count": val_summary["skipped_episode_count"],
            "per_task_episode_count": val_summary["per_task_episode_count"],
        },
    }
    with (args.output_root / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Finished dataset conversion. Summary written to %s", args.output_root / "conversion_summary.json")


if __name__ == "__main__":
    main()
