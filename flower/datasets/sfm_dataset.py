"""FLOWER adapter for the verified default_sfm crop-v2 disk dataset."""
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import ColorJitter, Compose, Normalize

from flower.datasets.disk_dataset import ExtendedDiskDataset
from flower.datasets.sfm_proprioception import load_physical_proprio, validate_proprio_source
from flower.utils.action_history import normalized_zero_motion_action
from flower.utils.transforms import ScaleImageTensor


class SfmDiskDataset(ExtendedDiskDataset):
    """Use inclusive demo boundaries and share color jitter across both views.

    Numeric arrays and 224x224 crops are already processed on the data host.
    The training config selects the action window length from stored per-step
    actions. Only complete windows are sampled because FLOWER has no padding
    loss mask. The final observation/action stays available in the last window.
    """

    def __init__(self, *args, shared_color_jitter=True, allow_smoke_test=False,
                 proprio_source="normalized", load_skill_history=False,
                 load_quest_observations=False, **kwargs):
        self.load_skill_history = bool(load_skill_history)
        self.load_quest_observations = bool(load_quest_observations)
        dataset_dir = Path(kwargs["datasets_dir"])
        root = dataset_dir.parent
        self.contract = json.loads((root / "data_contract.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest["complete"] or (manifest["smoke_test"] and not allow_smoke_test):
            raise ValueError("A complete production SFM dataset is required")
        if self.contract["schema"] != "flower_calvin_npz_sfm_v1":
            raise ValueError("Unsupported SFM data contract")
        if self.load_skill_history:
            normalization = json.loads((root / self.contract["normalization_file"]).read_text())
            self.skill_history_padding_action = normalized_zero_motion_action(normalization)
        self.proprio_source = validate_proprio_source(proprio_source)
        self.physical_proprio = load_physical_proprio(root) if self.proprio_source == "physical" else None
        if self.contract.get("prohibit_additional_crop", False):
            # Cropping/letterboxing is already complete. Restrict the loader
            # transforms to pixel scaling/normalization, including nested Compose.
            # Shared color jitter is applied separately below on the stored crops.
            def check_image_transform(transform):
                if isinstance(transform, Compose):
                    for child in transform.transforms:
                        check_image_transform(child)
                elif not isinstance(transform, (ScaleImageTensor, Normalize)):
                    raise ValueError(
                        "SFM images are already cropped: only ScaleImageTensor and "
                        f"Normalize are allowed after loading, got {type(transform).__name__}"
                    )

            for key in self.contract["rgb_keys"]:
                transform = (kwargs.get("transforms") or {}).get(key)
                if transform is not None:
                    check_image_transform(transform)
        for name in ("obs_seq_len", "action_seq_len"):
            value = kwargs[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.required_window = int(kwargs["obs_seq_len"]) + int(kwargs["action_seq_len"]) - 1
        if not (kwargs["min_window_size"] == kwargs["max_window_size"] == self.required_window):
            raise ValueError("SFM uses fixed windows of obs_seq_len + action_seq_len - 1")
        if kwargs.get("pad", False):
            raise ValueError("SFM action padding requires a loss mask; use complete windows")
        # Legacy contracts record action_horizon=5 from the conversion recipe.
        # NPZs/extracted arrays contain every individual action, so that value
        # does not limit the training horizon selected by action_seq_len.
        super().__init__(*args, **kwargs)
        self.color_jitter = None
        if shared_color_jitter and not self.validation:
            config = self.contract["image_preprocessing"]["augmentation"]["color_jitter"]
            self.color_jitter = ColorJitter(**config)

    def _build_file_indices_lang(self, abs_datasets_dir):
        annotations = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        boundaries = np.asarray(annotations["info"]["indx"], dtype=np.int64)
        stored = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        if not np.array_equal(boundaries, stored):
            raise ValueError("Language and episode boundaries differ")
        if len(boundaries) == 0 or np.any(boundaries[:, 1] < boundaries[:, 0]):
            raise ValueError("Invalid episode boundaries")
        if np.any(boundaries[1:, 0] != boundaries[:-1, 1] + 1):
            raise ValueError("SFM demos must be contiguous and nonoverlapping")
        lookup, lang_lookup = [], []
        for episode, (start, end) in enumerate(boundaries):
            starts = range(int(start), int(end) - self.required_window + 2, self.skip_frames)
            if len(starts) == 0:
                raise ValueError(f"Episode {episode} is shorter than the action window")
            lookup.extend(starts)
            lang_lookup.extend([episode] * len(starts))
        return (np.asarray(lookup, dtype=np.int64), lang_lookup,
                annotations["language"]["emb"], annotations["language"]["ann"])

    def _load_episode(self, idx, window_size):
        episode = super()._load_episode(idx, window_size)
        start = int(self.episode_lookup[idx])
        if self.load_skill_history:
            episode.update(self._build_skill_prev_history(start + self.obs_seq_len - 1))
        if self.load_quest_observations:
            # Quest's prior was trained on the stored crops and normalized full
            # state. Preserve these BEFORE color augmentation or physical/XY
            # proprio ablations used by FLOWER.
            for key in ("rgb_static", "rgb_gripper"):
                episode[f"quest_{key}"] = episode[key].transpose(0, 3, 1, 2).copy()
            episode["quest_robot_obs"] = episode["robot_obs"].copy()
        if self.physical_proprio is not None:
            episode["robot_obs"] = np.array(self.physical_proprio[start:start + self.obs_seq_len], copy=True)
        if self.color_jitter is None:
            return episode
        keys = ("rgb_static", "rgb_gripper")
        images = np.concatenate([episode[key] for key in keys], axis=0)
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        images = self.color_jitter(tensor).permute(0, 2, 3, 1).numpy()
        vision = self.contract["image_preprocessing"]
        out_h, out_w = vision["final_image_size"]
        for view, (key, camera) in enumerate(zip(keys, ("head_cam_d455", "wrist_cam_d405_a"))):
            image = images[view * self.obs_seq_len:(view + 1) * self.obs_seq_len].copy()
            _, _, height, width = vision["crop_tlhw"][camera]
            scale = min(out_h / height, out_w / width)
            h, w = round(height * scale), round(width * scale)
            top, left = (out_h - h) // 2, (out_w - w) // 2
            image[:, :top] = 0
            image[:, top + h:] = 0
            image[:, :, :left] = 0
            image[:, :, left + w:] = 0
            episode[key] = image
        return episode

    def _build_skill_prev_history(self, current_t):
        """[t-32,t) within this demo; t is the first target action, never history."""
        episode_id = np.searchsorted(self.ep_start_end_ids[:, 0], current_t, side="right") - 1
        if episode_id < 0 or current_t > self.ep_start_end_ids[episode_id, 1]:
            raise ValueError(f"Frame {current_t} is outside the inclusive demo boundaries")
        start = max(int(self.ep_start_end_ids[episode_id, 0]), current_t - 32)
        # Physical zero motion is not zero in the asymmetric normalized action
        # space. Keep the existing +1 (open) startup gripper convention.
        history = np.broadcast_to(self.skill_history_padding_action, (32, 7)).copy()
        valid = np.zeros(32, dtype=np.float32)
        count = current_t - start
        if count:
            if self.use_extracted_rel_actions:
                indices = [self.extracted_ep_npz_name_to_npy_idx[t] for t in range(start, current_t)]
                actions = self.extracted_ep_rel_actions[indices]
            else:
                actions = []
                for t in range(start, current_t):
                    frame = self.load_file(self._get_episode_name(t))
                    try:
                        actions.append(np.asarray(frame["rel_actions"], dtype=np.float32).copy())
                    finally:
                        if hasattr(frame, "close"):
                            frame.close()
                actions = np.stack(actions)
            if actions.shape != (count, 7) or not np.isfinite(actions).all():
                raise ValueError("Quest history requires finite normalized 7-D rel_actions")
            # SFM already stores held gripper targets; do not apply old relabeling.
            history[-count:] = actions
            valid[-count:] = 1
        return dict(skill_prev_actions=history, skill_prev_valid_mask=valid)
