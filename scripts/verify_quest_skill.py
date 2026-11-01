#!/usr/bin/env python3
"""Offline integration check with actual SFM samples, Quest and FLOWER weights."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
from torch.utils.data import default_collate


def weights_digest(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check(source, args):
    with hydra.initialize_config_dir(config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name=args.config_name or f"config_default_sfm_skill_{source}", overrides=args.override)
    module = hydra.utils.instantiate(cfg.datamodule)
    module.setup("fit")
    dataset = module.train_datasets["lang"]
    # A complete history and an episode-start sample exercise both input cases.
    rows = [min(40, len(dataset) - 1), 0][:args.batch_size]
    batch = default_collate([dataset[row] for row in rows])
    for key in ("quest_rgb_static", "quest_rgb_gripper"):
        if source in ("current", "both"):
            assert batch[key].dtype == torch.uint8
    quest_normalization = Path(cfg.model.quest_project_root) / "reproduction/flower_id2430/metadata/normalization.json"
    assert hashlib.sha256(quest_normalization.read_bytes()).digest() == hashlib.sha256(
        (Path(cfg.root_data_dir) / "normalization.json").read_bytes()).digest(), "Different data normalization"
    model = hydra.utils.instantiate(cfg.model).to(args.device)

    def move(value):
        if isinstance(value, dict): return {key: move(item) for key, item in value.items()}
        return value.to(args.device) if torch.is_tensor(value) else value

    batch = move(batch)
    model.train()
    assert all(not p.requires_grad for p in model.quest_skill.parameters())
    assert all(not child.training for child in model.quest_skill.modules())
    initial_digest = weights_digest(model.quest_skill)
    padding_check = None
    if source in ("history", "both"):
        from reproduction.flower_id2430.inference import normalize_delta_actions, denormalize_actions
        neutral = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        native_padding = normalize_delta_actions(neutral, metadata_dir=quest_normalization.parent)
        actual_padding = model.quest_skill.history_padding_action.cpu().numpy()
        np.testing.assert_array_equal(actual_padding, native_padding)
        recovered = denormalize_actions(actual_padding, metadata_dir=quest_normalization.parent)
        np.testing.assert_allclose(recovered[:6], 0, rtol=0, atol=2e-9)
        assert recovered[6] == 1
        invalid = ~batch["skill_prev_valid_mask"].bool()
        torch.testing.assert_close(batch["skill_prev_actions"][invalid],
                                   model.quest_skill.history_padding_action.expand(int(invalid.sum()), 7),
                                   rtol=0, atol=0)
        padding_check = dict(normalized_physical_zero=actual_padding.tolist(),
                             inverse_physical_action=recovered.tolist(),
                             inverse_pose_max_abs_error=float(np.abs(recovered[:6]).max()),
                             padded_slots_checked=int(invalid.sum()), native_normalization_exact_match=True,
                             mask_semantics="Replace missing-history slots before encoding; no attention masking")

    # Compare against the native model methods with the same random state.
    torch.manual_seed(123)
    with torch.no_grad():
        actual = model.quest_skill(batch, len(rows))
        torch.manual_seed(123)
        expected = {}
        if source in ("history", "both"):
            encoder = model.quest_skill.model if source == "history" else model.quest_skill.model.autoencoder
            expected["history"] = encoder.get_skill(batch["skill_prev_actions"])
        if source in ("current", "both"):
            from reproduction.flower_id2430.inference import pack_input
            examples = [pack_input(
                batch["quest_rgb_static"][i, -1].permute(1, 2, 0).cpu().numpy(),
                batch["quest_rgb_gripper"][i, -1].permute(1, 2, 0).cpu().numpy(),
                batch["quest_robot_obs"][i, -1].cpu().numpy(),
                next(name for name, text in model.quest_skill.metadata["data_contract"]["instruction_mapping"].items()
                     if text == batch["lang_text"][i]),
                device=args.device,
                metadata_dir=quest_normalization.parent,
            ) for i in range(len(rows))]
            native_batch = dict(obs={key: torch.cat([e["obs"][key] for e in examples]) for key in examples[0]["obs"]},
                                task_id=torch.cat([e["task_id"] for e in examples]))
            _, context = model.quest_skill.model.conditioned_context(native_batch, train_mode=False)
            expected["current"] = model.quest_skill.model.predicted_skill(context)
        if source != "both":
            expected = expected[source]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    loss = None
    if args.backward:
        cond = model.encode_observations(batch)
        loss, _ = model.rf_loss(cond, batch["actions"])
        loss.backward()
        assert torch.isfinite(loss)
        gradient = model.skill_proj[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        if source == "both":
            history_gradient = model.history_skill_proj[-1].weight.grad
            assert history_gradient is not None and torch.isfinite(history_gradient).all() and history_gradient.abs().sum() > 0
        assert all(p.grad is None for p in model.quest_skill.parameters())
        del cond
        model.zero_grad(set_to_none=True)

    model.eval()
    calls = []
    hook = model.quest_skill.register_forward_hook(lambda *unused: calls.append(1))
    with torch.no_grad():
        cond = model.encode_observations(batch)
        output = model.sample_actions(torch.randn_like(batch["actions"]), cond, inference=True)
    hook.remove()
    assert len(calls) == 1 and torch.isfinite(output).all()
    assert weights_digest(model.quest_skill) == initial_digest
    result = dict(source=source, device=args.device, config=args.config_name or f"config_default_sfm_skill_{source}",
                  checkpoint=model.quest_skill.provenance, samples=rows,
                  skill_shape={k: list(v.shape) for k, v in actual.items()} if source == "both" else list(actual.shape),
                  action_shape=list(output.shape),
                  native_skill_exact_match=True, frozen_weights_unchanged=True,
                  skill_calls_per_flow_sample=len(calls), backward_checked=args.backward,
                  loss=float(loss.detach()) if loss is not None else None,
                  history_padding=padding_check,
                  quest_weights_sha256=initial_digest)
    del model, module, dataset, batch, cond, output, actual, expected, loss
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["current", "history", "both"], default="both")
    parser.add_argument("--config-name", help="Verify one composed experiment config, including combined skills")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--override", action="append", default=[], help="Hydra override, repeatable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    sources = ["current", "history"] if args.source == "both" else [args.source]
    if args.config_name:
        with hydra.initialize_config_dir(config_dir=str(ROOT / "conf")):
            cfg = hydra.compose(config_name=args.config_name, overrides=args.override)
        sources = [cfg.model.skill_source]
    report = dict(results=[check(source, args) for source in sources])
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
