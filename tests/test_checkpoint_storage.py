"""Checkpoint output defaults and evaluation metadata, without GPU training."""

import json
from pathlib import Path
import shutil

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import pytest

from flower.utils.checkpoint_storage import prepare_checkpoint_storage
from server_flower_rpc import find_training_config


def config(tmp_path, *, dirpath=None):
    return OmegaConf.create({
        "checkpoint_root": str(tmp_path / "storage"),
        "log_dir": str(tmp_path / "logs" / "default_sfm"),
        "seed": 42,
        "proprio_dims": 11,
        "model": {"lowdim_obs_dim": "${proprio_dims}"},
        "callbacks": {"checkpoint": {"dirpath": dirpath}},
    })


def test_checkpoint_and_config_remain_usable_after_logs_are_removed(tmp_path):
    cfg = config(tmp_path)
    run = Path(cfg.log_dir) / "runs" / "experiment"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra/overrides.yaml").write_text("- max_epochs=30\n")
    destination = prepare_checkpoint_storage(cfg, run)
    callback = ModelCheckpoint(dirpath=cfg.callbacks.checkpoint.dirpath)
    assert Path(callback.dirpath) == destination
    checkpoint = destination / "epoch004.ckpt"
    checkpoint.write_bytes(b"checkpoint path fixture")
    local = run / "seed_42/saved_models"
    assert local.is_symlink() and (local / checkpoint.name).read_bytes() == checkpoint.read_bytes()
    metadata = json.loads((run / "checkpoint_storage.json").read_text())
    assert metadata["local_path_points_to_storage"]
    shutil.rmtree(run)
    stored = OmegaConf.load(find_training_config(checkpoint))
    assert stored.model.lowdim_obs_dim == 11
    assert stored.callbacks.checkpoint.dirpath == str(destination)
    assert (destination / ".hydra/overrides.yaml").read_text() == "- max_epochs=30\n"


def test_other_rank_resolves_same_path_without_writing(tmp_path):
    first, second = config(tmp_path), config(tmp_path)
    run = Path(first.log_dir) / "resumes" / "experiment"
    predicted = prepare_checkpoint_storage(second, run, write_metadata=False)
    assert not predicted.exists() and not run.exists()
    actual = prepare_checkpoint_storage(first, run)
    assert actual == predicted
    assert second.callbacks.checkpoint.dirpath == first.callbacks.checkpoint.dirpath
    # Re-entering with a resolved config is idempotent.
    assert prepare_checkpoint_storage(first, run) == actual


def test_explicit_destination_preserves_existing_log_checkpoints(tmp_path):
    destination = tmp_path / "custom" / "weights"
    cfg = config(tmp_path, dirpath=str(destination))
    run = Path(cfg.log_dir) / "runs" / "old_run"
    old = run / "seed_42/saved_models/epoch004.ckpt"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"existing checkpoint")
    assert prepare_checkpoint_storage(cfg, run) == destination
    assert old.read_bytes() == b"existing checkpoint" and not old.parent.is_symlink()
    metadata = json.loads((run / "checkpoint_storage.json").read_text())
    assert metadata["explicit_dirpath"] and not metadata["local_path_points_to_storage"]


def test_external_run_directories_with_same_name_do_not_collide(tmp_path):
    paths = [prepare_checkpoint_storage(config(tmp_path), tmp_path / base / "same_name", write_metadata=False)
             for base in ("first", "second")]
    assert paths[0] != paths[1]
    for path in paths:
        path.relative_to(tmp_path / "storage")


def test_configs_without_storage_root_keep_their_original_behavior(tmp_path):
    cfg = config(tmp_path, dirpath="saved_models")
    del cfg.checkpoint_root
    assert prepare_checkpoint_storage(cfg, tmp_path) is None
    assert cfg.callbacks.checkpoint.dirpath == "saved_models"


@pytest.mark.parametrize("name", [path.stem for path in sorted(
    (Path(__file__).resolve().parents[1] / "conf").glob("config_default_sfm*.yaml")
)])
def test_all_sfm_variants_inherit_central_storage(name, tmp_path):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "conf")):
        cfg = compose(config_name=name)
    assert cfg.checkpoint_root == "/data/jack/Skilled_FLOWER/checkpoints"
    assert cfg.callbacks.checkpoint.dirpath is None
    cfg.checkpoint_root = str(tmp_path / "storage")
    cfg.log_dir = str(tmp_path / "logs" / Path(cfg.log_dir).name)
    run = Path(cfg.log_dir) / "runs" / "storage_config_check"
    output = prepare_checkpoint_storage(cfg, run)
    expected = Path(cfg.checkpoint_root) / Path(cfg.log_dir).name / "runs/storage_config_check/seed_42/saved_models"
    assert output == expected
    stored = OmegaConf.load(find_training_config(output / "epoch004.ckpt"))
    assert stored.callbacks.checkpoint.dirpath == str(expected)
    assert stored.model.lowdim_obs_dim == cfg.model.lowdim_obs_dim
    assert "${" not in (output / ".hydra/config.yaml").read_text()
