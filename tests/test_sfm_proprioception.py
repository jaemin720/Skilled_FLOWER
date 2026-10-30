"""Verify exact physical state exports and CPU-only train/inference adaptation."""
import json

import hydra
import numpy as np
import pytest
import torch

from flower.datasets.sfm_proprioception import (
    config_proprio_source, load_physical_proprio, physical_from_normalized, sha256,
    validate_proprio_source,
)
import server_flower_rpc as rpc

ROOT = rpc.ROOT / "datasets/default_sfm_crop_v2"


@pytest.fixture(scope="module")
def physical():
    if not (ROOT / "unnormalized_proprio/manifest.json").is_file():
        pytest.skip("Kingston state export is not installed")
    return load_physical_proprio(ROOT)


def test_export_transfer_and_frame_alignment(physical):
    folder = ROOT / "unnormalized_proprio"
    metadata = json.loads((folder / "manifest.json").read_text())
    for name, info in metadata["files"].items():
        assert sha256(folder / name) == info["sha256"]
    raw = np.load(folder / "source_state.npy", mmap_mode="r", allow_pickle=False)
    source_indices = np.load(folder / "source_index.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(folder / "timestamp.npy", mmap_mode="r", allow_pickle=False)
    stats = json.loads((ROOT / "normalization.json").read_text())["state"]
    # Unwrapped position/velocity/gripper channels must be exact source bytes.
    for i, j in enumerate(metadata["raw_state_indices"]):
        if metadata["robot_obs_columns"][i] not in ("roll", "yaw"):
            np.testing.assert_array_equal(physical[:, i], raw[:, j])
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for episode in manifest["episodes_detail"]:
        for index in (episode["start_frame"], episode["end_frame"]):
            with np.load(ROOT / "training" / f"episode_{index:07d}.npz", allow_pickle=False) as stored:
                expected = ((physical[index].astype(np.float64) - stats["offset"]) / stats["scale"]).astype(np.float32)
                np.testing.assert_array_equal(stored["robot_obs"], expected)
                assert stored["source_index"] == source_indices[index]
                assert stored["timestamp"] == timestamps[index]
                np.testing.assert_allclose(physical_from_normalized(stored["robot_obs"], stats),
                                           physical[index], rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("keep", [[[0, 13]], [[0, 1], [3, 13]]])
def test_actual_loader_selects_physical_state_only(physical, keep):
    width = sum(b-a for a, b in keep)
    with hydra.initialize_config_dir(config_dir=str(rpc.ROOT / "conf")):
        cfg = hydra.compose(config_name="config_default_sfm_unnormalized_state", overrides=[
            f"proprio_dims={width}", f"datamodule.proprioception_dims.keep_indices={json.dumps(keep)}"])
    assert config_proprio_source(cfg) == "physical"
    module = hydra.utils.instantiate(cfg.datamodule)
    module.setup("fit")
    dataset = module.train_datasets["lang"]
    dataset.color_jitter = None
    indices = [i for a, b in keep for i in range(a, b)]
    records = json.loads((ROOT / "manifest.json").read_text())["episodes_detail"]
    for record in records[::50]:
        for frame in (record["start_frame"], record["end_frame"] - cfg.act_seq_len + 1):
            item_index = int(np.searchsorted(dataset.episode_lookup, frame))
            assert dataset.episode_lookup[item_index] == frame
            sample = dataset[item_index]
            np.testing.assert_array_equal(sample["robot_obs"].numpy(), physical[frame, indices][None])
            with np.load(ROOT / "training" / f"episode_{frame:07d}.npz", allow_pickle=False) as stored:
                np.testing.assert_array_equal(sample["right_force_history"][0], stored["right_force_history"])
                np.testing.assert_array_equal(sample["actions"][0], stored["rel_actions"])
                for key in ("rgb_static", "rgb_gripper"):
                    expected = module.train_transforms[key](torch.from_numpy(stored[key].copy()).permute(2, 0, 1)[None])
                    torch.testing.assert_close(sample["rgb_obs"][key], expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_invalid_proprio_mode_rejected():
    with pytest.raises(ValueError, match="proprio_source"):
        validate_proprio_source("raw19")


def test_offline_replay_uses_physical_inputs_and_keeps_report_decoding(physical):
    from scripts.evaluate_default_sfm_trajectories import read_batch
    from omegaconf import OmegaConf

    transforms = OmegaConf.load(rpc.ROOT / "conf/datamodule/transforms/default_sfm_transforms.yaml")
    cfg = OmegaConf.create(dict(datamodule=dict(transforms=transforms)))
    assert config_proprio_source(cfg) == "normalized"  # Older saved configs lack proprio_source.
    indices = [0, *range(3, 13)]
    batch, report_data = read_batch(ROOT, [0, 1], "usb", indices, rpc.image_transforms(cfg), physical)
    np.testing.assert_array_equal(batch["robot_obs"].numpy(), physical[[0, 1]][:, indices][:, None])
    for i in range(2):
        with np.load(ROOT / "training" / f"episode_{i:07d}.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(report_data["robot_obs"][i], stored["robot_obs"])
