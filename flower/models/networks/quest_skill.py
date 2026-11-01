"""Frozen ID2430 skill extraction using the original Quest_SFM handoff code.

The action encoder and observation-conditioned prior both produce [B,8,4].
Only the downstream FLOWER projection is trainable; no AE decoder is called.
"""
import importlib.util
from functools import partial
from pathlib import Path

import torch
from torch import nn

from flower.utils.action_history import normalized_zero_motion_action


def quest_partial(target, *args, **kwargs):
    """Equivalent to Hydra 1.2+ _partial_, also usable by FLOWER's Hydra 1.1."""
    from hydra.utils import get_method
    return partial(get_method(target), *args, **kwargs)


def _compatible_factories(value):
    """Translate factory syntax only; preserve all trained architecture values."""
    if isinstance(value, list):
        return [_compatible_factories(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _compatible_factories(item) for key, item in value.items() if key != "_partial_"}
    if value.get("_partial_", False):
        result["target"] = result["_target_"]
        result["_target_"] = "flower.models.networks.quest_skill.quest_partial"
    return result


def prepare_action_history(history, batch_size, device, dtype, *, padding_action=None, valid_mask=None):
    """Keep 32 past actions; fill missing slots with normalized physical zero motion.

    Missing history is an error; an explicitly empty history means episode start.
    padding_action must come from the fitted action statistics, with startup
    gripper +1. Valid stored actions are never normalized again. The optional
    mask identifies left padding for replacement, not attention masking inside
    the frozen Quest encoder (which consumes the complete 32-step sequence).
    """
    if history is None:
        raise KeyError("history skill conditioning requires 'skill_prev_actions'")
    history = torch.as_tensor(history, device=device, dtype=dtype)
    if history.ndim == 4 and history.shape[1] == 1:
        history = history[:, 0]
    if history.ndim == 2:
        history = history.unsqueeze(0)
    if history.ndim != 3 or history.shape[0] != batch_size or history.shape[-1] != 7:
        raise ValueError(f"skill_prev_actions must be [{batch_size},T,7], got {tuple(history.shape)}")
    if not torch.isfinite(history).all():
        raise ValueError("skill_prev_actions contains nonfinite values")
    if history.shape[1] < 32 or valid_mask is not None:
        if padding_action is None:
            raise ValueError("Short/masked history requires a normalized physical-zero padding_action")
        padding_action = torch.as_tensor(padding_action, device=device, dtype=dtype)
        if padding_action.shape != (7,) or not torch.isfinite(padding_action).all():
            raise ValueError("padding_action must be a finite normalized 7-D action")
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=device)
        if valid.ndim == 3 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != history.shape[:2] or not torch.all((valid == 0) | (valid == 1)):
            raise ValueError("skill_prev_valid_mask must be binary [B,T] matching skill_prev_actions")
        valid = valid.bool()
        if torch.any(valid[:, :-1] & ~valid[:, 1:]):
            raise ValueError("skill_prev_valid_mask must describe left padding followed by valid actions")
        history = torch.where(valid[..., None], history, padding_action)
    history = history[:, -32:]
    if history.shape[1] < 32:
        padding = padding_action.expand(batch_size, 32 - history.shape[1], 7)
        history = torch.cat((padding, history), dim=1)
    return history.contiguous()


class FrozenQuestSkill(nn.Module):
    """Load the native completed checkpoint, pinned code and (for prior) CLIP."""

    def __init__(self, source, project_root, checkpoint_path, clip_snapshot=None):
        super().__init__()
        if source not in ("current", "history", "both"):
            raise ValueError("Quest skill source must be 'current', 'history' or 'both'")
        if not project_root or not checkpoint_path:
            raise ValueError("Quest conditioning requires quest_project_root and quest_checkpoint_path")
        project = Path(project_root).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        entry = project / "reproduction/flower_id2430/inference.py"
        if not entry.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"Quest handoff/checkpoint missing: {entry}, {checkpoint}")
        spec = importlib.util.spec_from_file_location("flower_quest_id2430_handoff", entry)
        handoff = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(handoff)
        # The handoff uses Hydra's newer _partial_ syntax. Adapt the private
        # loader instance, avoiding a global patch or a shared Hydra upgrade.
        native_read = handoff.read_checkpoint

        def read_compatible(*args, **kwargs):
            saved, info = native_read(*args, **kwargs)
            saved["config"] = _compatible_factories(saved["config"])
            return saved, info

        handoff.read_checkpoint = read_compatible
        self.source = source
        metadata_dir = entry.parent / "metadata"
        self.metadata = handoff.metadata(metadata_dir)
        # Derived from pinned metadata, not a learned tensor or a new checkpoint
        # key. The dataset computes the same vector from its matching statistics.
        self.register_buffer("history_padding_action", torch.from_numpy(
            normalized_zero_motion_action(self.metadata["normalization"])), persistent=False)
        if source == "history":
            self.model, self.provenance = handoff.load_autoencoder(
                checkpoint, device="cpu", project=project,
            )
        else:
            self.model, self.provenance = handoff.load_policy(
                checkpoint, device="cpu", project=project,
                metadata_dir=metadata_dir, clip_snapshot=clip_snapshot,
            )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent FLOWER.train() must not enable Quest dropout or update BN stats.
        return super().train(False)

    def _pack_current(self, batch, batch_size, device):
        """Separate untouched Quest inputs from Florence transforms/proprio ablations."""
        obs = {}
        for source, target in (("quest_rgb_static", "agentview_rgb"),
                               ("quest_rgb_gripper", "eye_in_hand_rgb")):
            if source not in batch:
                raise KeyError(f"current skill conditioning requires '{source}' (uint8 RGB)")
            value = torch.as_tensor(batch[source], device=device)
            if value.ndim == 4:
                value = value[:, None]
            if (value.dtype != torch.uint8 or value.ndim != 5 or value.shape[0] != batch_size
                    or value.shape[1] < 1 or value.shape[2:] != (3, 224, 224)):
                raise ValueError(f"{source} must be uint8 [B,T,3,224,224]")
            obs[target] = value[:, -1:].clone()
        if "quest_robot_obs" not in batch:
            raise KeyError("current skill conditioning requires normalized 13-D 'quest_robot_obs'")
        state = torch.as_tensor(batch["quest_robot_obs"], device=device, dtype=torch.float32)
        if state.ndim == 2:
            state = state[:, None]
        if (state.ndim != 3 or state.shape[0] != batch_size or state.shape[1] < 1
                or state.shape[-1] != 13 or not torch.isfinite(state).all()):
            raise ValueError("quest_robot_obs must be finite, normalized [B,T,13]")
        for key, columns in self.metadata["handoff"]["proprio_columns"].items():
            obs[key] = state[:, -1:, columns].clone()
        names = self.metadata["handoff"]["task_names"]
        mapping = self.metadata["data_contract"]["instruction_mapping"]
        task_ids = {text: i for i, name in enumerate(names) for text in (name, mapping[name])}
        texts = batch.get("lang_text", [])
        if isinstance(texts, str) or len(texts) != batch_size:
            raise ValueError("lang_text must contain one Quest task name or exact instruction per sample")
        unknown = [text for text in texts if text not in task_ids]
        if unknown:
            raise ValueError(f"Unknown Quest task instruction(s): {unknown}; expected {mapping}")
        return dict(obs=obs, task_id=torch.tensor([task_ids[text] for text in texts], device=device))

    @torch.no_grad()
    def forward(self, batch, batch_size):
        self.eval()
        device = next(self.model.parameters()).device
        # FLOWER uses mixed precision. Keep the pretrained Quest inference path
        # in float32, matching the handoff, without changing FLOWER's autocast.
        with torch.autocast(device_type=device.type, enabled=False):
            skills = {}
            if self.source in ("history", "both"):
                history = prepare_action_history(batch.get("skill_prev_actions"), batch_size,
                                                 device, torch.float32,
                                                 padding_action=self.history_padding_action,
                                                 valid_mask=batch.get("skill_prev_valid_mask"))
                encoder = self.model if self.source == "history" else self.model.autoencoder
                skills["history"] = encoder.get_skill(history)
            if self.source in ("current", "both"):
                data = self._pack_current(batch, batch_size, device)
                _, context = self.model.conditioned_context(data, train_mode=False)
                skills["current"] = self.model.predicted_skill(context)
        for name, skill in skills.items():
            if skill.shape != (batch_size, 8, 4) or not torch.isfinite(skill).all():
                raise ValueError(f"Quest {name} must return finite skills [B,8,4], got {tuple(skill.shape)}")
            skills[name] = skill.detach()
        # Combined experiments share one frozen Stage-II checkpoint, including
        # its Stage-I AE, while keeping both conditioning signals distinct.
        return skills if self.source == "both" else skills[self.source]
