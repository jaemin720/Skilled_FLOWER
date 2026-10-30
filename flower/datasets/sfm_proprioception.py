"""Select standardized SFM states or their exact pre-standardization export."""
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_proprio_source(value):
    if value not in ("normalized", "physical"):
        raise ValueError("proprio_source must be 'normalized' or 'physical'")
    return value


def config_proprio_source(config):
    dataset = config.datamodule.get("datasets", {}).get("lang_dataset", {})
    return validate_proprio_source(dataset.get("proprio_source", "normalized"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_physical_proprio(root):
    root = Path(root)
    folder = root / "unnormalized_proprio"
    if not (folder / "manifest.json").is_file():
        raise FileNotFoundError(f"Export/download the unnormalized proprioception first: {folder}")
    info = json.loads((folder / "manifest.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (info.get("version") != 1 or not info.get("complete") or
            info["source_manifest_sha256"] != sha256(root / "manifest.json") or
            info["source_normalization_sha256"] != sha256(root / "normalization.json") or
            info["source_data_contract_sha256"] != sha256(root / "data_contract.json")):
        raise ValueError("Physical proprioception belongs to a different dataset")
    path = folder / "robot_obs.npy"
    if sha256(path) != info["files"][path.name]["sha256"]:
        raise ValueError("Physical proprioception file checksum mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (manifest["frames"], 13) or values.dtype != np.float32:
        raise ValueError("Expected float32 physical states with one row per dataset frame")
    np.testing.assert_array_equal(np.load(folder / "ep_start_end_ids.npy", allow_pickle=False),
                                  np.load(root / "training/ep_start_end_ids.npy", allow_pickle=False))
    return values


def physical_from_normalized(values, stats):
    """Restore client-standardized state for a checkpoint trained in physical units."""
    if stats["method"] != "standard":
        raise ValueError("SFM state inversion requires standard normalization")
    values = np.asarray(values, dtype=np.float64)
    offset, scale = np.asarray(stats["offset"]), np.asarray(stats["scale"])
    if values.shape[-1] != 13 or offset.shape != (13,) or scale.shape != (13,) or np.any(scale <= 0):
        raise ValueError("Invalid SFM state normalization dimensions/scale")
    result = (values * scale + offset).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Invalid physical proprioception")
    return result
