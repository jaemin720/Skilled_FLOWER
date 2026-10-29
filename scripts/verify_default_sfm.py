#!/usr/bin/env python3
"""Verify transferred SFM files and exercise the actual FLOWER data loader."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/default_sfm_crop_v2")
    parser.add_argument("--allow-smoke-test", action="store_true")
    parser.add_argument("--skip-file-hashes", action="store_true", help="Only after a previous complete file verification")
    parser.add_argument("--action-seq-len", type=int, help="Override the training config's action chunk length")
    parser.add_argument("--report", type=Path, help="Save a separate report instead of replacing flower_verification.json")
    args = parser.parse_args()
    if args.action_seq_len is not None and args.action_seq_len < 1:
        parser.error("--action-seq-len must be positive")
    torch.set_num_threads(2)
    root = args.dataset.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contract = json.loads((root / "data_contract.json").read_text())
    assert manifest["complete"] and (not manifest["smoke_test"] or args.allow_smoke_test)
    assert sha256(root / "normalization.json") == contract["normalization_sha256"]
    records = manifest["episodes_detail"]
    files = sorted((root / "training").glob("episode_*.npz"))
    assert len(files) == manifest["frames"]
    assert [p.name for p in files] == [f"episode_{i:07d}.npz" for i in range(manifest["frames"])]
    task_counts = dict(Counter(row["task"] for row in records))
    assert task_counts == manifest["task_demo_counts"]
    if not manifest["smoke_test"]:
        assert manifest["episodes"] == 450 and len(task_counts) == 9 and set(task_counts.values()) == {50}
        positions = Counter((r["task"], r["position"]) for r in records)
        assert len(positions) == 45 and set(positions.values()) == {10}
    for i, row in enumerate(records):
        assert row["source"].split("/")[0] == f"{row['task']}_p{row['position']}"
        assert "bnc_old" not in row["source"]
        assert row["instruction"] == contract["instruction_mapping"][row["task"]]
        if not args.skip_file_hashes:
            digest = hashlib.sha256()
            for index in range(row["start_frame"], row["end_frame"] + 1):
                path = files[index]
                digest.update(path.name.encode() + b"\0" + bytes.fromhex(sha256(path)))
            assert digest.hexdigest() == row["frame_files_sha256"], f"Transferred frames differ: {row['source']}"
        if (i + 1) % 50 == 0:
            phase = "cohort_checked" if args.skip_file_hashes else "files_verified"
            print(json.dumps({"phase": phase, "episodes": i + 1, "total": len(records)}), flush=True)

    overrides = [f"root_data_dir={root}"]
    if args.action_seq_len is not None:
        overrides.append(f"act_seq_len={args.action_seq_len}")
    with hydra.initialize_config_dir(config_dir=str(REPO / "conf")):
        config = hydra.compose(config_name="config_default_sfm", overrides=overrides)
    horizon = config.act_seq_len
    assert config.model.act_window_size == horizon
    if args.allow_smoke_test:
        OmegaConf.set_struct(config, False)
        config.datamodule.datasets.lang_dataset.allow_smoke_test = True
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup("fit")
    dataset = datamodule.train_datasets["lang"]
    expected_windows = sum(r["frames"] - config.act_seq_len + 1 for r in records)
    assert len(dataset) == expected_windows
    assert datamodule.val_dataloader() is None
    jitter = dataset.color_jitter
    dataset.color_jitter = None
    for row in records:
        for index in (row["start_frame"], row["end_frame"] - config.act_seq_len + 1):
            sample_index = int(np.searchsorted(dataset.episode_lookup, index))
            assert dataset.episode_lookup[sample_index] == index
            sample = dataset[sample_index]
            assert sample["lang_text"] == row["instruction"]
            assert sample["robot_obs"].shape == (1, 13)
            assert sample["right_force_history"].shape == (1, 50, 6)
            assert sample["actions"].shape == (horizon, 7)
            assert torch.isfinite(sample["actions"]).all()
            assert torch.all(torch.abs(sample["actions"]) <= 1.00001)
            assert torch.all(torch.abs(sample["actions"][:, 6]) == 1)
            for step in range(config.act_seq_len):
                with np.load(files[index + step], allow_pickle=False) as stored:
                    np.testing.assert_array_equal(sample["actions"][step].numpy(), stored["rel_actions"])
            with np.load(files[index], allow_pickle=False) as stored:
                np.testing.assert_array_equal(sample["robot_obs"][0].numpy(), stored["robot_obs"])
                np.testing.assert_array_equal(sample["right_force_history"][0].numpy(), stored["right_force_history"])
                assert int(stored["source_index"]) == row["start"] + index - row["start_frame"]
                for key in ("rgb_static", "rgb_gripper"):
                    image = stored[key]
                    assert image.shape == (224, 224, 3) and image.dtype == np.uint8
                    expected = datamodule.train_transforms[key](torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0))
                    torch.testing.assert_close(sample["rgb_obs"][key], expected, rtol=0, atol=0)

    # Padding must remain black after the shared jitter, including contrast.
    dataset.color_jitter = jitter
    state_before = torch.random.get_rng_state()
    raw = dataset._load_episode(0, dataset.required_window)
    for key, camera in (("rgb_static", "head_cam_d455"), ("rgb_gripper", "wrist_cam_d405_a")):
        layout = contract["image_preprocessing"]["crop_layout_by_camera"][camera]
        top, bottom, left, right = layout["padding_tblr"]
        image = raw[key]
        assert np.all(image[:, :top] == 0) and np.all(image[:, image.shape[1] - bottom:] == 0)
        if left:
            assert np.all(image[:, :, :left] == 0)
        if right:
            assert np.all(image[:, :, -right:] == 0)
    torch.random.set_rng_state(state_before)
    batches = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)
    batch = next(iter(batches))
    assert batch["actions"].shape == (2, horizon, 7)
    assert batch["robot_obs"].shape == (2, 1, 13)
    assert batch["right_force_history"].shape == (2, 1, 50, 6)
    for image in batch["rgb_obs"].values():
        assert image.shape == (2, 1, 3, 224, 224) and torch.isfinite(image).all()
    report = dict(verified=True, dataset=str(root), smoke_test=manifest["smoke_test"],
                  action_seq_len=horizon,
                  episodes=len(records), frames=len(files), training_windows=len(dataset), task_demo_counts=task_counts,
                  full_frame_file_hashes_verified=not args.skip_file_hashes,
                  first_and_last_window_verified_for_every_demo=True,
                  numeric_values_unchanged_by_loader=True, image_geometry_unchanged_by_loader=True,
                  language_mapping_verified=True, final_actions_reachable=True,
                  batch_shapes=dict(actions=list(batch["actions"].shape), robot_obs=list(batch["robot_obs"].shape),
                                    right_force_history=list(batch["right_force_history"].shape),
                                    rgb_static=list(batch["rgb_obs"]["rgb_static"].shape)),
                  training_started=False)
    report_path = args.report if args.report is not None else root / "flower_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
