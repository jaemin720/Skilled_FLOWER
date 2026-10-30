#!/usr/bin/env python3
"""Export exact pre-normalization states from existing SFM physical HDF5s.

Run on Kingston. This reads numeric HDF5 datasets only, never the RGB arrays or
raw pickles, and leaves the original dataset untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(root, output):
    import h5py

    root, output = Path(root), Path(output)
    dataset = root / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    contract = json.loads((dataset / "data_contract.json").read_text())
    stats = json.loads((dataset / "normalization.json").read_text())["state"]
    if not manifest["complete"] or manifest["smoke_test"]:
        raise ValueError("A complete production dataset is required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    count = manifest["frames"]
    specifications = {"robot_obs.npy": ((count, 13), np.float32),
                      "source_state.npy": ((count, 19), np.float32),
                      "source_index.npy": ((count,), np.int64),
                      "timestamp.npy": ((count,), np.float64)}
    arrays = {name: np.lib.format.open_memmap(output / name, mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in specifications.items()}
    boundaries, offset = [], 0
    for row in manifest["episodes_detail"]:
        start, stop = row["start_frame"], row["end_frame"] + 1
        if start != offset or stop - start != row["frames"]:
            raise ValueError("Noncontiguous frame ordering in manifest")
        with h5py.File(root / row["physical_file"], "r") as f:
            state, raw = f["robot_obs"][:], f["source_state"][:]
            indices, timestamps = f["source_indices"][:], f["timestamps"][:]
        if state.dtype != np.float32 or raw.dtype != np.float32 or state.shape != (stop-start, 13) or raw.shape != (stop-start, 19):
            raise ValueError(f"Unexpected state dtype/shape: {row['source']}")
        if not np.isfinite(state).all() or not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite state: {row['source']}")
        np.testing.assert_array_equal(indices, np.arange(row["start"], row["start"] + row["frames"]))
        if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
            raise ValueError("Invalid timestamps")
        # Independently check that the physical state is the original selected
        # state with the already-agreed roll/yaw wrapping, before standardization.
        expected = raw[:, contract["raw_state_indices"]].astype(np.float64)
        upper = float(np.nextafter(np.float32(2*np.pi), np.float32(0)))
        for name in ("roll", "yaw"):
            i = contract["robot_obs_columns"].index(name)
            expected[:, i] = np.minimum(np.remainder(expected[:, i], 2*np.pi), upper)
        np.testing.assert_array_equal(state, expected.astype(np.float32))
        arrays["robot_obs.npy"][start:stop] = state
        arrays["source_state.npy"][start:stop] = raw
        arrays["source_index.npy"][start:stop] = indices
        arrays["timestamp.npy"][start:stop] = timestamps
        boundaries.append([start, stop-1])
        offset = stop
    if offset != count:
        raise ValueError("Frame count mismatch")
    # Every exported frame must reproduce the recorded fit of state statistics.
    state = arrays["robot_obs.npy"].astype(np.float64)
    np.testing.assert_allclose(state.mean(axis=0), stats["mean"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(state.std(axis=0), stats["std"], rtol=1e-10, atol=1e-10)
    for array in arrays.values():
        array.flush()
    boundaries = np.asarray(boundaries, dtype=np.int64)
    np.testing.assert_array_equal(boundaries, np.load(dataset / "training/ep_start_end_ids.npy", allow_pickle=False))
    np.save(output / "ep_start_end_ids.npy", boundaries, allow_pickle=False)
    report = dict(version=1, complete=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_root=str(root), source_manifest_sha256=sha256(dataset / "manifest.json"),
                  source_normalization_sha256=sha256(dataset / "normalization.json"),
                  source_data_contract_sha256=sha256(dataset / "data_contract.json"),
                  frames=count, episodes=manifest["episodes"],
                  robot_obs_columns=contract["robot_obs_columns"], raw_state_indices=contract["raw_state_indices"],
                  robot_obs=dict(file="robot_obs.npy", normalization="none", wrap_angles=["roll", "yaw"],
                                 shape=[count, 13], dtype="float32", source="physical/*.h5:robot_obs"),
                  source_state=dict(file="source_state.npy", normalization="none", wrap_angles=[],
                                    shape=[count, 19], dtype="float32", source="physical/*.h5:source_state"),
                  alignment="row i corresponds to training/episode_{i:07d}.npz; same initial trimming and inclusive demo boundaries",
                  verification=dict(all_rows_match_source_state_selection=True, state_statistics_match=True),
                  files={p.name:dict(sha256=sha256(p), bytes=p.stat().st_size)
                         for p in sorted(output.iterdir()) if p.suffix == ".npy"})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), frames=count, episodes=manifest["episodes"],
                          bytes=sum(v["bytes"] for v in report["files"].values()), complete=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.preprocessing_root, args.output)
