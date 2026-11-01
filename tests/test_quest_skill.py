"""Causality, frozen weights, input contracts and AdaLN integration without downloads."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hydra
import numpy as np
import pytest
import torch
from torch import nn

from flower.datasets.sfm_dataset import SfmDiskDataset
from flower.models.networks.quest_skill import FrozenQuestSkill, prepare_action_history
from flower.models.networks.transformers import FlowerAttention, stateless_norm
from flower.utils.action_history import normalized_zero_motion_action
from test_force_history import make_batch, make_model


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Insert USB connector into USB port"
PADDING_NORMALIZATION = {"action_pose": {
    "method": "symlog_minmax", "temperature": 20.0,
    "offset": [0, 0.02, 0, 1e-7, -3e-7, 0],
    "scale": [0.05, 0.03, 0.05, 1e-6, 1e-6, 0.1],
}}
HISTORY_PADDING = normalized_zero_motion_action(PADDING_NORMALIZATION)


class TinyQuest(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_projection = nn.Linear(7, 4)
        self.state_projection = nn.Linear(13, 4)
        self.dropout = nn.Dropout(0.5)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long), persistent=False)

    @property
    def autoencoder(self):
        return self

    def get_skill(self, actions):
        self.calls += 1
        return self.dropout(self.action_projection(actions.reshape(-1, 8, 4, 7).mean(2)))

    def conditioned_context(self, data, train_mode):
        assert train_mode is False
        obs = data["obs"]
        state = torch.cat([obs[key] for key in ("gripper_state", "tcp_pose", "tcp_velocity")], -1)
        context = self.state_projection(state)
        return data, context

    def predicted_skill(self, context):
        self.calls += 1
        return self.dropout(context.expand(-1, 8, -1))

    def decode_actions(self, skill):
        raise AssertionError("FLOWER must never call the Quest action decoder")


def setup_tiny_quest(self, source, *args):
    nn.Module.__init__(self)
    self.source = source
    self.model = TinyQuest()
    self.register_buffer("history_padding_action", torch.from_numpy(HISTORY_PADDING.copy()), persistent=False)
    self.metadata = {
        "handoff": {"task_names": ["usb"], "proprio_columns": {
            "tcp_pose": [1, 2, 3, 4, 5, 6], "tcp_velocity": [7, 8, 9, 10, 11, 12], "gripper_state": [0]}},
        "data_contract": {"instruction_mapping": {"usb": INSTRUCTION}},
    }
    self.requires_grad_(False)
    self.eval()


def skill_model(source, **kwargs):
    with patch.object(FrozenQuestSkill, "__init__", setup_tiny_quest):
        return make_model(skill_source=source, **kwargs)


def skill_batch(size=2):
    batch = make_batch(size)
    batch["skill_prev_actions"] = torch.randn(size, 32, 7)
    batch["quest_robot_obs"] = torch.randn(size, 1, 13)
    for key in ("quest_rgb_static", "quest_rgb_gripper"):
        batch[key] = torch.randint(256, (size, 1, 3, 224, 224), dtype=torch.uint8)
    return batch


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(2)
    torch.manual_seed(19)


@pytest.mark.parametrize("source", ["current", "history"])
@pytest.mark.parametrize("fusion", ["sum", "concat_mlp"])
def test_freeze_optimizer_and_both_adaln_paths(source, fusion):
    model = skill_model(source, adaln_conditioning_type=fusion)
    model.train()
    assert model.training and model.skill_proj.training
    assert all(not module.training for module in model.quest_skill.modules())
    assert all(not p.requires_grad for p in model.quest_skill.parameters())
    model.optimizer_config = SimpleNamespace(transformer_weight_decay=0.01)
    groups = model._get_param_groups()
    optimized = {id(p) for group in groups for p in group["params"]}
    assert all(id(p) not in optimized for p in model.quest_skill.parameters())
    assert all(id(p) in optimized for p in model.skill_proj.parameters())
    before = {k: v.clone() for k, v in model.quest_skill.state_dict().items()}
    batch = skill_batch()
    batch["skill_prev_actions"].requires_grad_()
    batch["quest_robot_obs"].requires_grad_()
    loss, _ = model.rf_loss(model.encode_observations(batch), batch["actions"])
    loss.backward()
    for parameter in (model.skill_proj[-1].weight, model.force_history_encoder.gru.weight_ih_l0,
                      model.adaln["eef_delta"].modCX[-1].weight,
                      model.dit[-1].adaLN_modulation[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert batch["skill_prev_actions"].grad is None
    assert batch["quest_robot_obs"].grad is None
    torch.optim.AdamW(groups, lr=1e-3).step()
    for key, value in model.quest_skill.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("source", ["current", "history"])
def test_skill_added_like_upstream_and_changes_actions(source):
    model = skill_model(source)
    cond = model.encode_observations(skill_batch())
    zero_cond = dict(cond, skill_embeds=torch.zeros_like(cond["skill_embeds"]))
    shared, blocks = [], []
    hooks = [model.adaln["eef_delta"].register_forward_pre_hook(lambda _, args: shared.append(args[0].clone())),
             model.dit[0].register_forward_pre_hook(lambda _, args: blocks.append(args[1].clone()))]
    try:
        z, t = torch.randn(2, 5, 7), torch.full((2,), 0.5)
        original = model.dit_forward(z, t, cond)
        changed = model.dit_forward(z, t, zero_cond)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(shared[0] - shared[1], stateless_norm(cond["skill_embeds"]))
    torch.testing.assert_close(shared[0], blocks[0])
    assert not torch.allclose(original, changed)


@pytest.mark.parametrize("source", ["current", "history"])
def test_one_skill_per_observation_future_targets_do_not_leak(source):
    model = skill_model(source)
    batch = skill_batch(1)
    expected = model.encode_observations(batch)["skill_embeds"]
    batch["actions"].fill_(999)
    torch.testing.assert_close(model.encode_observations(batch)["skill_embeds"], expected)
    model.quest_skill.model.calls.zero_()
    with torch.no_grad():
        output = model(batch, {"lang_text": INSTRUCTION})
    assert model.quest_skill.model.calls == 1
    assert output.shape == (1, 5, 7) and torch.isfinite(output).all()


def test_current_inputs_are_exact_latest_values_and_not_mutated():
    model = skill_model("current")
    batch = skill_batch()
    batch["quest_robot_obs"] = torch.randn(2, 3, 13)
    batch["quest_rgb_static"] = torch.randint(256, (2, 3, 3, 224, 224), dtype=torch.uint8)
    original = {k: v.clone() for k, v in batch.items() if k.startswith("quest_")}
    data = model.quest_skill._pack_current(batch, 2, model.device)
    torch.testing.assert_close(data["obs"]["tcp_pose"], batch["quest_robot_obs"][:, -1:, 1:7])
    torch.testing.assert_close(data["obs"]["agentview_rgb"], batch["quest_rgb_static"][:, -1:])
    data["obs"]["agentview_rgb"].zero_()
    model.encode_observations(batch)
    for key, value in original.items():
        torch.testing.assert_close(batch[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["missing_image", "float_image", "short_state", "unknown_task", "batch_mismatch"])
def test_current_input_errors_are_explicit(bad):
    model, batch = skill_model("current"), skill_batch()
    if bad == "missing_image": del batch["quest_rgb_static"]
    if bad == "float_image": batch["quest_rgb_static"] = batch["quest_rgb_static"].float()
    if bad == "short_state": batch["quest_robot_obs"] = batch["quest_robot_obs"][..., :11]
    if bad == "unknown_task": batch["lang_text"][0] = "unknown"
    if bad == "batch_mismatch": batch["lang_text"] = [INSTRUCTION]
    with pytest.raises((KeyError, ValueError)):
        model.encode_observations(batch)


def test_history_padding_crop_and_no_batch_broadcast():
    actions = torch.randn(2, 40, 7)
    torch.testing.assert_close(prepare_action_history(actions, 2, "cpu", torch.float32), actions[:, -32:])
    short = prepare_action_history(actions[:, :2], 2, "cpu", torch.float32, padding_action=HISTORY_PADDING)
    torch.testing.assert_close(short[:, :30], torch.from_numpy(HISTORY_PADDING).expand(2, 30, 7))
    torch.testing.assert_close(short[:, -2:], actions[:, :2])
    empty = prepare_action_history(actions[:, :0], 2, "cpu", torch.float32, padding_action=HISTORY_PADDING)
    torch.testing.assert_close(empty, torch.from_numpy(HISTORY_PADDING).expand(2, 32, 7))
    with pytest.raises(ValueError, match="padding_action"):
        prepare_action_history(actions[:, :2], 2, "cpu", torch.float32)
    for history in (None, actions[:1], torch.zeros(2, 32, 6), torch.full((2, 32, 7), float("nan"))):
        with pytest.raises((KeyError, ValueError)):
            prepare_action_history(history, 2, "cpu", torch.float32)


def test_padding_normalization_restores_zero_physical_pose_on_every_axis():
    stats = PADDING_NORMALIZATION["action_pose"]
    pose = HISTORY_PADDING[:6].astype(np.float64)
    restored = (np.sign(pose) * np.expm1(np.abs(pose) * np.log1p(20)) / 20
                * np.array(stats["scale"]) + np.array(stats["offset"]))
    np.testing.assert_allclose(restored, 0, rtol=0, atol=2e-9)
    assert HISTORY_PADDING[1] < -0.8 and HISTORY_PADDING[-1] == 1


@pytest.mark.parametrize("source", ["history", "both"])
def test_forward_mask_corrects_padding_before_frozen_encoder_without_changing_real_actions(source):
    model = skill_model(source)
    batch = skill_batch(1)
    batch["skill_prev_actions"][:, :30] = 0  # Legacy caller's incorrect padding.
    before = batch["skill_prev_actions"].clone()
    batch["skill_prev_valid_mask"] = torch.zeros(1, 32)
    batch["skill_prev_valid_mask"][:, -2:] = 1
    with patch.object(model.quest_skill.model, "get_skill", wraps=model.quest_skill.model.get_skill) as encoder:
        with torch.no_grad():
            model(batch, {"lang_text": INSTRUCTION})
    received = encoder.call_args[0][0]
    torch.testing.assert_close(received[:, :30], torch.from_numpy(HISTORY_PADDING).expand(1, 30, 7))
    torch.testing.assert_close(received[:, -2:], before[:, -2:], rtol=0, atol=0)
    torch.testing.assert_close(batch["skill_prev_actions"], before, rtol=0, atol=0)


@pytest.mark.parametrize("mask", [torch.zeros(2, 32), torch.full((1, 32), 0.5),
                                  torch.tensor([[1] * 31 + [0]]), torch.full((1, 32), float("nan"))])
def test_history_rejects_misaligned_or_noncausal_padding_mask(mask):
    with pytest.raises(ValueError, match="skill_prev_valid_mask"):
        prepare_action_history(torch.zeros(1, 32, 7), 1, "cpu", torch.float32,
                               padding_action=HISTORY_PADDING, valid_mask=mask)


def test_training_dropout_attention_supports_cpu_and_retains_causality():
    # Production training has attention dropout; the math SDPA backend rejects
    # the previous simultaneous explicit mask + is_causal=True combination.
    attention = FlowerAttention(32, 4, attn_pdrop=0.1, resid_pdrop=0.0).train()
    x = torch.randn(2, 10, 32, requires_grad=True)
    result = attention(x, is_causal=True)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    attention.eval()
    with torch.no_grad():
        original = attention(x, is_causal=True)
        changed = x.detach().clone()
        changed[:, 5:] += 10
        altered = attention(changed, is_causal=True)
    torch.testing.assert_close(original[:, :5], altered[:, :5], rtol=0, atol=0)
    assert not torch.allclose(original[:, 5:], altered[:, 5:])


@pytest.mark.parametrize("extracted", [False, True])
def test_history_stays_in_demo_and_excludes_current_target(extracted):
    dataset = object.__new__(SfmDiskDataset)
    dataset.skill_history_padding_action = HISTORY_PADDING
    dataset.ep_start_end_ids = np.array([[0, 39], [40, 79]])
    dataset.use_extracted_rel_actions = extracted
    dataset.extracted_ep_rel_actions = np.arange(80 * 7, dtype=np.float32).reshape(80, 7)
    dataset.extracted_ep_npz_name_to_npy_idx = {i: i for i in range(80)}
    dataset._get_episode_name = lambda i: i
    dataset.load_file = lambda i: {"rel_actions": dataset.extracted_ep_rel_actions[i]}
    for current in (0, 1, 31, 32, 39, 40, 41, 79):
        result = dataset._build_skill_prev_history(current)
        start = max(0 if current < 40 else 40, current - 32)
        count = current - start
        assert result["skill_prev_valid_mask"].sum() == count
        if count:
            np.testing.assert_array_equal(result["skill_prev_actions"][-count:], dataset.extracted_ep_rel_actions[start:current])
        np.testing.assert_array_equal(result["skill_prev_actions"][:32-count],
                                      np.broadcast_to(HISTORY_PADDING, (32-count, 7)))


def test_rollout_tracks_only_returned_actions_and_reset_clears_history():
    model = skill_model("history")
    batch = skill_batch(1)
    del batch["skill_prev_actions"]
    with torch.no_grad():
        for _ in range(7):
            action = model.step(batch, {"lang_text": INSTRUCTION})
            torch.testing.assert_close(model.skill_action_history[0, -1], action)
            torch.testing.assert_close(model.skill_action_history[0, 0], torch.from_numpy(HISTORY_PADDING))
    assert model.quest_skill.model.calls == 2
    model.reset()
    assert model.skill_action_history is None and model.rollout_step_counter == 0
    with pytest.raises(KeyError, match="skill_prev_actions"):
        model(batch, {"lang_text": INSTRUCTION})


def test_step_consumes_short_explicit_mask_before_padding_and_reset():
    model = skill_model("history")
    batch = skill_batch(1)
    batch["skill_prev_actions"] = batch["skill_prev_actions"][:, -2:]
    batch["skill_prev_valid_mask"] = torch.tensor([[0, 1]])
    valid_action = batch["skill_prev_actions"][0, -1].clone()
    with torch.no_grad():
        model.step(batch, {"lang_text": INSTRUCTION})
    torch.testing.assert_close(model.skill_action_history[0, :30], torch.from_numpy(HISTORY_PADDING).expand(30, 7))
    torch.testing.assert_close(model.skill_action_history[0, -2], valid_action)
    assert batch["skill_prev_valid_mask"].shape == (1, 2)
    model.reset()
    assert model.skill_action_history is None


@pytest.mark.parametrize("source", ["current", "history"])
def test_checkpoint_roundtrip_and_legacy_no_extra_parameters(source):
    model = skill_model(source)
    restored = skill_model(source)
    restored.load_state_dict(model.state_dict(), strict=True)
    batch = skill_batch()
    torch.testing.assert_close(model.encode_observations(batch)["skill_embeds"],
                               restored.encode_observations(batch)["skill_embeds"])
    legacy = make_model()
    assert not any(key.startswith(("quest_skill.", "skill_proj.")) for key in legacy.state_dict())


@pytest.mark.parametrize("fusion", ["sum", "concat_mlp"])
def test_combined_skills_have_separate_gradients_and_both_condition_adaln(fusion):
    model = skill_model("both", adaln_conditioning_type=fusion)
    batch = skill_batch()
    cond = model.encode_observations(batch)
    assert cond["skill_embeds"].shape == cond["history_skill_embeds"].shape == (2, 32)
    loss, _ = model.rf_loss(cond, batch["actions"])
    loss.backward()
    for projection in (model.skill_proj, model.history_skill_proj):
        assert torch.isfinite(projection[-1].weight.grad).all()
        assert projection[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.quest_skill.parameters())
    z, t = torch.randn(2, 5, 7), torch.full((2,), 0.5)
    expected = model.dit_forward(z, t, cond)
    for key in ("skill_embeds", "history_skill_embeds"):
        modified = dict(cond, **{key: torch.zeros_like(cond[key])})
        assert not torch.allclose(expected, model.dit_forward(z, t, modified))
    restored = skill_model("both", adaln_conditioning_type=fusion)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored.dit_forward(z, t, restored.encode_observations(batch)), expected)


def test_combined_generator_and_encoder_run_once_each_per_action_chunk():
    model = skill_model("both")
    batch = skill_batch(1)
    with patch.object(model.quest_skill.model, "get_skill", wraps=model.quest_skill.model.get_skill) as encoder, \
         patch.object(model.quest_skill.model, "predicted_skill", wraps=model.quest_skill.model.predicted_skill) as generator:
        with torch.no_grad():
            output = model(batch, {"lang_text": INSTRUCTION})
    assert encoder.call_count == generator.call_count == 1
    assert output.shape == (1, 5, 7)
    model.reset()
    assert model.skill_action_history is None


@pytest.mark.parametrize("source", ["current", "history"])
def test_config_has_separate_logs_and_only_required_inputs(source):
    with hydra.initialize_config_dir(config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name=f"config_default_sfm_skill_{source}")
    assert cfg.model.skill_source == source
    assert Path(cfg.model.quest_checkpoint_path).name == "final.pt"
    assert cfg.datamodule.datasets.lang_dataset.load_skill_history == (source == "history")
    assert cfg.datamodule.datasets.lang_dataset.load_quest_observations == (source == "current")
    assert cfg.model.use_force_history_adaln and cfg.act_seq_len == 10
    assert cfg.log_dir.endswith(f"skill_{source}")


@pytest.mark.skipif(not (ROOT / "datasets/default_sfm_crop_v2/manifest.json").is_file(),
                    reason="Production SFM dataset is not installed")
@pytest.mark.parametrize("proprio_source", ["normalized", "physical"])
def test_real_loader_preserves_quest_inputs_across_flower_augmentation_and_state_ablation(proprio_source):
    data_root = ROOT / "datasets/default_sfm_crop_v2"
    if proprio_source == "physical" and not (data_root / "unnormalized_proprio/manifest.json").is_file():
        pytest.skip("Physical state export is not installed")
    with hydra.initialize_config_dir(config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name="config_default_sfm_skill_current", overrides=[
            "proprio_dims=11", "datamodule.proprioception_dims.keep_indices=[[0,1],[3,13]]",
            "datamodule.datasets.lang_dataset.load_skill_history=true",
            f"datamodule.datasets.lang_dataset.proprio_source={proprio_source}",
        ])
    module = hydra.utils.instantiate(cfg.datamodule)
    module.setup("fit")
    dataset = module.train_datasets["lang"]
    # Keep training color jitter enabled: it must not alter Quest's inputs.
    for frame in (0, 1, 32, int(dataset.ep_start_end_ids[1, 0])):
        row = int(np.searchsorted(dataset.episode_lookup, frame))
        sample = dataset[row]
        with np.load(data_root / "training" / f"episode_{frame:07d}.npz") as original:
            np.testing.assert_array_equal(sample["quest_robot_obs"], original["robot_obs"][None])
            assert sample["robot_obs"].shape == (1, 11)
            np.testing.assert_array_equal(sample["actions"][0], original["rel_actions"])
            for key in ("rgb_static", "rgb_gripper"):
                assert sample[f"quest_{key}"].dtype == torch.uint8
                np.testing.assert_array_equal(sample[f"quest_{key}"][0], original[key].transpose(2, 0, 1))
        expected = dataset._build_skill_prev_history(frame)
        np.testing.assert_array_equal(sample["skill_prev_actions"], expected["skill_prev_actions"])
        np.testing.assert_array_equal(sample["skill_prev_valid_mask"], expected["skill_prev_valid_mask"])
        # Known native Quest normalization of physical zero XYZ/rotvec. In
        # particular, normalized delta-y=0 would instead mean +2.22 cm.
        zero_motion = np.array([-1.2364959900423855e-7, -0.9301646947860718,
                                -3.248862157079202e-8, -7.656934940314386e-6,
                                -2.1586030925391242e-5, 8.672435214407415e-9, 1], dtype=np.float32)
        invalid = np.asarray(sample["skill_prev_valid_mask"]) == 0
        np.testing.assert_array_equal(sample["skill_prev_actions"][invalid],
                                      np.broadcast_to(zero_motion, (int(invalid.sum()), 7)))
