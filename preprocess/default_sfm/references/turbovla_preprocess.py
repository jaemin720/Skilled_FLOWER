#!/usr/bin/env python3
"""Convert the exact 450-demo cohort; raw inputs remain read-only. Never train."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from turbovla.data.sfm_preprocessing import (
    CAMERAS, STATE_COLUMNS, STATE_INDICES, Moments, chunk_actions, delta_actions,
    held_targets, initial_start, normalize, normalized_actions, preprocess_image,
    require, sample, state_physical, variant_spec,
)

AUDIT = REPO / "experiments/default_sfm/preprocessing_audit"
DEFAULT_PLAN = REPO / "experiments/default_sfm/crop_plans/experiment_plan_crop_v2.yaml"


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class HashingReader:
    """Conventional pickle reader, hashing every original byte including RGB."""
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()
        self.bytes = 0

    def read(self, n=-1):
        value = self.stream.read(n)
        self.digest.update(value)
        self.bytes += len(value)
        return value

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        self.digest.update(memoryview(buffer)[:count])
        self.bytes += count
        return count

    def readline(self, *args):
        value = self.stream.readline(*args)
        self.digest.update(value)
        self.bytes += len(value)
        return value


def references():
    source = AUDIT / "csilab_source_snapshot/preprocess/convert.py"
    reference = json.loads((AUDIT / "csilab_reference_recheck.json").read_text())
    expected = next(r["sha256"] for r in reference["files"] if r["path"] == "preprocess/convert.py")
    require(sha256(source) == expected, "csilab reference hash changed")
    names = {"latest", "numeric", "crop_start", "image_sample"}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names), "Missing reference functions")
    scope = {"np": np, "cv2": cv2, "Rotation": Rotation, "PreprocessError": ValueError}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    held_dir = AUDIT / "faflow_held_reference"
    record = json.loads((held_dir / "source.json").read_text())
    require(sha256(held_dir / "gripper.py") == record["sha256"], "FAFLOW reference hash changed")
    spec = importlib.util.spec_from_file_location("sfm_held_reference", held_dir / "gripper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scope["GripperLabels"] = module.GripperLabels
    return scope


class Decoder:
    def channel(self, observation, key):
        state = sample(observation["state"], (19,), "state")
        return state[{"right/tcp_pose": slice(4, 10), "right/gripper_pose": slice(0, 1)}[key]]


def check_collection(root, records):
    contracts, hashes = [], {}
    for row in records:
        if not row["path"].endswith(".json"):
            continue
        path = root / row["path"]
        hashes[row["path"]] = sha256(path)
        require(hashes[row["path"]] == row["sha256"], f"Collection config hash changed: {path}")
        config = json.loads(path.read_text())
        train = config.get("train", {})
        contract = {"train": {k: train.get(k) for k in ("arm_type", "hz", "proprio_keys")},
                    "env": {"right": {"gripper_control": config.get("env", {}).get("right", {}).get("gripper_control", {})}},
                    "gripper_interfaces": config.get("gripper_interfaces")}
        contracts.append(contract)
    require(len(contracts) == 67 and all(c == contracts[0] for c in contracts), "Collection decoding contracts differ")
    contract = contracts[0]
    require(contract["train"] == {"arm_type": "right", "hz": 10, "proprio_keys": [
        "right/gripper_pose", "right/tcp_force", "right/tcp_pose", "right/tcp_torque", "right/tcp_vel"]}, "Unexpected state layout")
    require(contract["gripper_interfaces"] is None and contract["env"]["right"]["gripper_control"]["mode"] == "binary", "Unexpected gripper contract")
    return contract, hashes


def validate_plan(plan):
    require(plan["dataset"]["demo_count"] == 450 and plan["dataset"]["split"]["held_out_demo_count"] == 0, "Unexpected cohort/split")
    require(plan["force_torque"]["normalization"]["status"] == "confirmed"
            and plan["force_torque"]["normalization"]["method"] == "standard"
            and plan["force_torque"]["normalization"]["std_floor"] == .01, "F/T normalization is not the confirmed standard/.01")
    require(plan["force_torque"]["zero_history_policy"]["include_in_statistics"], "Normal zero histories must be retained")
    require(plan["state"]["rpy"]["wrap_angles"] == ["roll", "yaw"], "Unexpected angle wrapping")
    require(plan["state"]["normalization"]["method"] == "standard", "Unexpected state normalization")
    require(plan["action"]["training_representation"] == "delta" and plan["action"]["chunk_length"] == 5
            and plan["action"]["gripper_label_rule"] == "held_command_target", "Unexpected action contract")
    norm = plan["action"]["normalization"]
    require(norm["method"] == "symlog_minmax" and norm["temperature"] == 20. and norm["epsilon"] == 1e-6, "Unexpected action normalization")
    vision = plan["vision"]
    require(vision["final_image_size"] == [224, 224] and vision["resize_mode"] == "letterbox"
            and vision["interpolation"] == "area" and vision["pad_value"] == 0, "Unexpected image transform")
    require(set(vision["crop_tlhw"]) == set(CAMERAS.values()), "Expected one crop for each native camera")
    for camera, crop in vision["crop_tlhw"].items():
        require(isinstance(crop, (list, tuple)) and len(crop) == 4
                and all(type(value) is int for value in crop), f"{camera}: crop must be integer [top, left, height, width]")
        top, left, height, width = crop
        require(top >= 0 and left >= 0 and height > 0 and width > 0
                and top + height <= 480 and left + width <= 640, f"{camera}: crop outside native 640x480 image")
    require(len(plan["variants"]) == 9 and tuple(plan["state"]["conditions"][0]["retained_raw_indices"]) == STATE_INDICES, "Unexpected variant layout")


def collect_one(job):
    cv2.setNumThreads(1)
    root, output, index, record, expected, plan, collection = job
    root, output = Path(root), Path(output)
    source = root / record["path"]
    target = output / "episodes" / f"episode_{index:04d}.h5"
    evidence = target.with_suffix(".audit.json")
    started = time.monotonic()
    before = source.stat()
    require((before.st_size, before.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), f"Raw metadata changed: {source}")
    with source.open("rb") as stream:
        reader = HashingReader(stream)
        rows = pickle.load(reader)
        require(reader.read(1) == b"", "Unexpected trailing pickle data")
    require(reader.bytes == record["bytes"] and reader.digest.hexdigest() == record["sha256"], f"Raw SHA-256 mismatch: {source}")
    after = source.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Raw input changed during read")
    require(isinstance(rows, list) and len(rows) == expected["original_frames"], "Unexpected episode container/count")
    terminals = np.asarray([bool(r.get("dones", False) or r.get("terminated", False) or r.get("truncated", False)) for r in rows])
    require(terminals[-1] and terminals.sum() == 1, "Each PKL must contain exactly one complete episode")
    require({r.get("infos", {}).get("action_format") for r in rows} == {"absolute_tcp_xyz_rpy_gripper"}, "Unexpected action format")
    state = np.stack([sample(r["observations"]["state"], (19,), "state") for r in rows])
    actions = np.stack([sample(r["actions"], (7,), "actions") for r in rows])
    timestamps = np.asarray([r["time"] for r in rows], np.float64)
    require(np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0), "Nonmonotonic or nonfinite observation times")
    cfg = plan["dataset"]["frame_selection"]["detection_config"]
    start, motion = initial_start(state, cfg)
    require(start is not None and start == expected["start"], "Start differs from confirmed full-cohort audit")
    ref = references()
    ref_start, _ = ref["crop_start"](rows, Decoder(), cfg)
    require(start == ref_start, "Start differs from csilab reference")
    held, receipts, receipt_present, fallback = held_targets(rows, state, collection["env"]["right"]["gripper_control"])
    ref_held = ref["GripperLabels"](rows, collection, Decoder(),
        dict(source_mode="binary", mode="absolute", command_range=[-1., 1.], observation_range=[0., 1.], grasp={"enabled": False}), ["right"]).labels[:, 0]
    require(np.array_equal(held, ref_held), "Held labels differ from pinned FAFLOW reference")
    delta = delta_actions(state, actions, held)
    recovered_xyz = state[:, 4:7] + delta[:, :3]
    recovered_rot = Rotation.from_rotvec(delta[:, 3:6]) * Rotation.from_euler("xyz", state[:, 7:10])
    xyz_error = float(np.max(np.abs(recovered_xyz - actions[:, :3])))
    rotation_error = float(np.max((recovered_rot.inv() * Rotation.from_euler("xyz", actions[:, 3:6])).magnitude()))
    require(xyz_error < 1e-10 and rotation_error < 1e-10, "Delta geometry round-trip failed")
    ft = np.stack([sample(r["observations"]["right_force_history"], (50, 6), "FT") for r in rows[start:]])
    require(ft.dtype == np.float32 and state.dtype == np.float32, "Unexpected raw numeric dtype")
    count = len(rows) - start
    task, position = Path(record["path"]).parts[0].rsplit("_p", 1)
    summary = dict(source=record["path"], source_sha256=record["sha256"], source_bytes=record["bytes"],
        file=str(target.relative_to(output)), task=task, position=int(position),
        instruction=plan["dataset"]["instruction_mapping"][task], original_frames=len(rows), frames=count,
        start=start, detected_motion_frame=motion, removed_initial_frames=start,
        all_zero_ft_windows=int(np.all(ft == 0, axis=(1, 2)).sum()),
        receipt_frames_all=int(receipt_present.sum()), receipt_fallback_frames_all=fallback,
        receipt_frames_retained=int(receipt_present[start:].sum()),
        held_counts={str(v): int((held[start:] == v).sum()) for v in (-1., 1.)},
        delta_geometry_max_error_m=xyz_error, delta_geometry_max_error_rad=rotation_error,
        observed_period_s=dict(min=float(np.diff(timestamps).min()), median=float(np.median(np.diff(timestamps))), max=float(np.diff(timestamps).max())),
        per_sensor_sample_timestamps_available=False,
        reference_initial_start_matches=True, reference_held_labels_match_all_frames=True,
        reference_image_pixels_match_all_retained_frames=True)
    temporary = target.with_suffix(".h5.tmp")
    require(not target.exists() and not temporary.exists(), f"Output already exists: {target}")
    with h5py.File(temporary, "x") as f:
        f.attrs.update(schema_version=1, phase="physical", normalized=False, task=task, position=int(position),
                       instruction=summary["instruction"], source_file=record["path"], source_sha256=record["sha256"],
                       num_steps=count, original_num_steps=len(rows), crop_start=start)
        numeric = {"source_state": state[start:], "source_actions": actions[start:], "ft_history_raw": ft,
                   "delta_actions": delta[start:], "gripper_labels": held[start:].astype(np.float32),
                   "gripper_target_percent": receipts[start:], "gripper_receipt_present": receipt_present[start:],
                   "timestamps": timestamps[start:], "source_indices": np.arange(start, len(rows), dtype=np.int64),
                   "terminals": terminals[start:], "rewards": np.asarray([r.get("rewards", 0.) for r in rows[start:]], np.float32),
                   "masks": np.asarray([r.get("masks", 0. if terminals[i] else 1.) for i, r in enumerate(rows)])[start:]}
        for key, value in numeric.items():
            f.create_dataset(key, data=value, compression="lzf")
        image_hashes = {}
        for key, camera in CAMERAS.items():
            ds = f.create_dataset("observations/" + key, shape=(count, 224, 224, 3), dtype=np.uint8,
                                  chunks=(1, 224, 224, 3), compression="lzf")
            ds.attrs.update(color_order="RGB", layout="THWC", source_key=camera)
            image_digest = hashlib.sha256()
            entry = dict(crop=plan["vision"]["crop_tlhw"][camera], resize=[224, 224], resize_mode="letterbox", interpolation="area", input_color="rgb")
            for offset in range(0, count, 16):
                images = []
                for row in rows[start + offset:start + min(offset + 16, count)]:
                    image = preprocess_image(row["observations"][camera], camera, plan["vision"])
                    require(np.array_equal(image, ref["image_sample"](row["observations"][camera], entry)), "Native image crop differs from csilab")
                    images.append(image)
                images = np.stack(images)
                ds[offset:offset + len(images)] = images
                image_digest.update(images.tobytes())
            image_hashes[key] = image_digest.hexdigest()
        summary["processed_image_pixel_sha256"] = image_hashes
        f.create_dataset("metadata/source_audit", data=json_bytes(summary).decode(), dtype=h5py.string_dtype())
    # Independent HDF5 readback of every transformed pixel and every numeric array.
    with h5py.File(temporary, "r") as f:
        for key, value in numeric.items():
            require(np.array_equal(f[key][:], value, equal_nan=True), f"Numeric HDF5 readback differs: {key}")
        for key, expected_hash in image_hashes.items():
            digest = hashlib.sha256()
            for offset in range(0, count, 16):
                digest.update(f["observations/" + key][offset:offset + 16].tobytes())
            require(digest.hexdigest() == expected_hash, f"Image HDF5 readback differs: {key}")
    temporary.replace(target)
    summary["collection_seconds"] = time.monotonic() - started
    write_json(evidence, summary)
    return summary


def fit_statistics(output, records, plan):
    state, ft, pose = Moments(), Moments(), Moments()
    for row in records:
        with h5py.File(output / row["file"], "r") as f:
            state.add(state_physical(f["source_state"][:]))
            ft.add(f["ft_history_raw"][:])
            actions, mask = chunk_actions(f["delta_actions"][:], 5)
            pose.add(actions[mask][:, :6])
    result = dict(version=1, fit_scope="all retained training observations, all 50 FT samples including intentional zeros; valid H=5 action entries",
                  state=state.result("standard", 1e-6), ft=ft.result("standard", plan["force_torque"]["normalization"]["std_floor"]),
                  action_pose=pose.result("symlog_minmax", 1e-6, 20.), gripper={"method": "none", "closed": -1., "open": 1.},
                  state_columns=list(STATE_COLUMNS), ft_columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                  shared_across_all_nine_variants=True, ft_application_count=1, production_statistics_fitted=True)
    write_json(output / "normalization.json", result)
    return result


def normalize_one(job):
    output, row, stats, stats_hash = job
    path = Path(output) / row["file"]
    with h5py.File(path, "r+") as f:
        require(f.attrs["phase"] == "physical", "Refusing repeated normalization")
        physical = state_physical(f["source_state"][:])
        ft = f["ft_history_raw"][:]
        actions, mask = chunk_actions(f["delta_actions"][:], 5)
        state_z = normalize(physical, stats["state"]).astype(np.float32)
        ft_z = normalize(ft, stats["ft"]).astype(np.float32)
        action_z = normalized_actions(actions, mask, stats["action_pose"])
        for key, value in {"observations/state": state_z, "observations/ft_history": ft_z,
                           "actions": action_z, "action_mask": mask}.items():
            f.create_dataset(key, data=value, compression="lzf")
        f.create_dataset("metadata/normalization", data=json_bytes(stats).decode(), dtype=h5py.string_dtype())
        f.attrs.update(phase="normalized", normalized=True, normalization_sha256=stats_hash)
        f["observations/state"].attrs.update(normalization="standard", columns=json.dumps(STATE_COLUMNS))
        f["observations/ft_history"].attrs.update(normalization="standard", epsilon=.01, already_normalized=True)
        f["actions"].attrs.update(normalization="symlog_minmax_pose_held_gripper", frame="robot_base", rotation="rotvec_rad", horizon=5)
    with h5py.File(path, "r") as f:
        # Reopen after persistence, and reverse the stored float32 model inputs.
        errors = {}
        for key, original, coefficients, name in [("observations/state", physical, stats["state"], "state"),
                                                  ("observations/ft_history", ft, stats["ft"], "ft")]:
            restored = normalize(f[key][:], coefficients, inverse=True)
            np.testing.assert_allclose(restored, original, atol=2e-6, rtol=1e-6)
            errors[name] = float(np.max(np.abs(restored - original)))
        stored = f["actions"][:]
        restored = normalize(stored[..., :6][mask], stats["action_pose"], inverse=True)
        np.testing.assert_allclose(restored, actions[..., :6][mask], atol=2e-6, rtol=1e-6)
        errors["action_pose"] = float(np.max(np.abs(restored - actions[..., :6][mask])))
        require(np.array_equal(stored[..., 6][mask], actions[..., 6][mask]), "Held gripper changed by normalization")
        require(np.all(stored[~mask] == 0) and np.array_equal(f["action_mask"][:], mask), "Action tail padding/mask differs")
        raw = f["source_state"][:]
        step = normalize(stored[:, 0, :6], stats["action_pose"], inverse=True)
        target = f["source_actions"][:]
        pos_error = float(np.max(np.abs(raw[:, 4:7] + step[:, :3] - target[:, :3])))
        rotations = Rotation.from_rotvec(step[:, 3:6]) * Rotation.from_euler("xyz", raw[:, 7:10])
        rot_error = float(np.max((rotations.inv() * Rotation.from_euler("xyz", target[:, 3:6])).magnitude()))
        require(pos_error < 2e-6 and rot_error < 2e-6, "Stored action geometry round-trip failed")
        errors.update(absolute_target_position_m=pos_error, absolute_target_rotation_rad=rot_error)
        require(int(mask.sum()) == 5 * len(raw) - 10, "Unexpected valid chunk count")
    return dict(row, bytes=path.stat().st_size, sha256=sha256(path), valid_action_entries=int(mask.sum()),
                normalization_roundtrip_max_errors=errors, verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO / "data/default_sfm")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN,
                        help="Preprocessing plan (default: %(default)s)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-demos", type=int, help="Smoke test only; never use its statistics for the 450-demo cohort")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    require(args.plan.is_relative_to(REPO), "Use a versioned plan inside this repository")
    cv2.setNumThreads(1)
    require(1 <= args.workers <= 8, "Use 1..8 workers")
    root, output = args.raw_root.resolve(), args.output.resolve()
    require(not output.is_relative_to(root) and not root.is_relative_to(output), "Output must be separate from raw input")
    require(not output.exists(), "Output already exists; select a new version directory")
    plan = yaml.safe_load(args.plan.read_text())
    validate_plan(plan)
    inventory = json.loads((REPO / "experiments/default_sfm/setup_records/copy_manifest.json").read_text())
    records = sorted((r for r in inventory["files"] if r["path"].endswith(".pkl")), key=lambda r: r["path"])
    require(len(records) == 450, "Expected exactly 450 PKLs")
    for r in inventory["files"]:
        st = (root / r["path"]).stat()
        require((st.st_size, st.st_mtime_ns) == (r["bytes"], r["mtime_ns"]), f"Input metadata changed: {r['path']}")
    collection, collection_hashes = check_collection(root, inventory["files"])
    if args.limit_demos is not None:
        require(args.limit_demos > 0, "Smoke limit must be positive")
        records = records[:args.limit_demos]
    starts = {r["path"]: r for r in json.loads((AUDIT / "initial_stationary_audit.json").read_text())["demos"]}
    output.mkdir(parents=True)
    (output / "episodes").mkdir()
    source_files = [Path(__file__), REPO / "turbovla/data/sfm_preprocessing.py", REPO / "turbovla/data/default_sfm.py", args.plan,
                    Path(__file__).with_name("verify_preprocessed.py"), Path(__file__).with_name("preflight_numeric.py"),
                    Path(__file__).with_name("test_preprocessing.py"), Path(__file__).with_name("audit_rpy.py"),
                    Path(__file__).with_name("requirements-reproduction.txt"),
                    AUDIT / "csilab_source_snapshot/preprocess/convert.py",
                    AUDIT / "csilab_source_snapshot/preprocess/normalization.py",
                    AUDIT / "csilab_reference_recheck.json", AUDIT / "initial_stationary_audit.json",
                    AUDIT / "faflow_held_reference/gripper.py", AUDIT / "faflow_held_reference/source.json",
                    REPO / "experiments/default_sfm/setup_records/copy_manifest.json"]
    code_hashes = {str(p.relative_to(REPO)): sha256(p) for p in source_files}
    for path in source_files:
        destination = output / "source_snapshot" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    run = dict(started_at_utc=utc(), raw_root=str(root), output_root=str(output), smoke_test=args.limit_demos is not None,
               code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               code_sha256=code_hashes, workers=args.workers, command=sys.argv, raw_data_modified=False,
               training_started=False, optimizer_steps=0, collection_config_sha256=collection_hashes)
    write_json(output / "run.json", run)
    write_json(output / "preprocessing_config.json", plan)
    write_json(output / "collection_decoding_contract.json", collection)
    shutil.copyfile(REPO / "experiments/default_sfm/setup_records/copy_manifest.json", output / "raw_copy_manifest.json")
    start_time = time.monotonic()
    context = multiprocessing.get_context("spawn")
    try:
        jobs = [(str(root), str(output), i, r, starts[r["path"]], plan, collection) for i, r in enumerate(records)]
        collected = []
        write_json(output / "status.json", dict(phase="collecting", completed=0, total=len(jobs), updated_at_utc=utc()))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for result in pool.map(collect_one, jobs):
                collected.append(result)
                status = dict(phase="collecting", completed=len(collected), total=len(jobs), updated_at_utc=utc(),
                              elapsed_seconds=time.monotonic() - start_time, frames=sum(r["frames"] for r in collected),
                              verified_source_bytes=sum(r["source_bytes"] for r in collected), last_source=result["source"])
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        write_json(output / "source_manifest.json", collected)
        write_json(output / "status.json", dict(phase="fitting_statistics", completed=len(collected), updated_at_utc=utc()))
        stats = fit_statistics(output, collected, plan)
        stats_hash = sha256(output / "normalization.json")
        print(json.dumps({"phase": "statistics_fitted", "ft_std": stats["ft"]["std"], "ft_scale": stats["ft"]["scale"], "ft_epsilon": .01}), flush=True)
        completed = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for result in pool.map(normalize_one, [(str(output), r, stats, stats_hash) for r in collected]):
                completed.append(result)
                status = dict(phase="normalizing_and_verifying", completed=len(completed), total=len(collected), updated_at_utc=utc())
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        total_frames = sum(r["frames"] for r in completed)
        if args.limit_demos is None:
            require(len(completed) == 450 and total_frames == 138985, "Full-cohort count mismatch")
        require(stats["state"]["count"] == total_frames and stats["ft"]["count"] == total_frames * 50, "Statistics sample count mismatch")
        require(stats["action_pose"]["count"] == sum(r["valid_action_entries"] for r in completed), "Action fit count mismatch")
        specs = {v["id"]: variant_spec(plan, v["id"]) for v in plan["variants"]}
        contract = dict(version=1, dataset="default_sfm", normalized=True, shared_storage=True,
            observation_keys=["state", "ft_history", *CAMERAS], state_dim=13, raw_state_indices=list(STATE_INDICES),
            ft_shape=[50, 6], ft_hz=500, ft_hz_basis="user provided; per-sample timestamps unavailable",
            ft_normalized_in_dataset=True, model_must_not_normalize_ft_again=True,
            action_shape=[5, 7], action_frame="robot_base", action_rotation="rotvec_rad", action_units="meters_radians_held_gripper",
            action_reference="each corresponding action timestep observation, not chunk-start observation",
            gripper_labels="held command target from receipts, fallback to binary command latch", action_padding="zero_after_normalization",
            save_next_observations=False, split="all_450_train_separate_evaluation", variants=specs,
            normalization_file="normalization.json", normalization_sha256=stats_hash,
            image_preprocessing=plan["vision"], state_rpy=dict(plan["state"]["rpy"], applied=True),
            instruction_mapping=plan["dataset"]["instruction_mapping"], source_code_sha256=code_hashes)
        write_json(output / "data_contract.json", contract)
        write_json(output / "inference_config.json", dict(plan=plan, normalization_file="normalization.json", normalization_sha256=stats_hash, variants=specs,
            transform_module="turbovla.data.sfm_preprocessing.preprocess_observation", raw_image_shape=[480, 640, 3], raw_state_dim=19, raw_ft_shape=[50, 6]))
        (output / "variants").mkdir()
        for name, spec in specs.items():
            write_json(output / "variants" / (name + ".json"), dict(spec, dataset_manifest="../manifest.json", normalization_file="../normalization.json", normalization_sha256=stats_hash))
        for path, digest in code_hashes.items():
            require(sha256(REPO / path) == digest, f"Source code/plan changed while preprocessing: {path}")
        for r in inventory["files"]:
            st = (root / r["path"]).stat()
            require((st.st_size, st.st_mtime_ns) == (r["bytes"], r["mtime_ns"]), "Source metadata changed during preprocessing")
        manifest = dict(complete=True, smoke_test=args.limit_demos is not None, finished_at_utc=utc(), elapsed_seconds=time.monotonic() - start_time,
            episodes=len(completed), original_frames=sum(r["original_frames"] for r in completed), frames=total_frames,
            removed_initial_frames=sum(r["removed_initial_frames"] for r in completed),
            valid_action_entries=sum(r["valid_action_entries"] for r in completed),
            all_zero_ft_windows=sum(r["all_zero_ft_windows"] for r in completed),
            receipt_fallback_frames_all=sum(r["receipt_fallback_frames_all"] for r in completed),
            total_hdf5_bytes=sum(r["bytes"] for r in completed), normalization_sha256=stats_hash,
            data_contract_sha256=sha256(output / "data_contract.json"), raw_pkl_sha256_verified=len(completed),
            raw_config_sha256_verified=len(collection_hashes), raw_data_modified=False, training_started=False,
            shards=completed)
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", dict(phase="complete", completed=len(completed), total=len(completed), frames=total_frames, updated_at_utc=utc(), training_started=False))
        print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2), flush=True)
    except BaseException as error:
        write_json(output / "status.json", dict(phase="failed", error=repr(error), updated_at_utc=utc(), training_started=False))
        raise


if __name__ == "__main__":
    main()
