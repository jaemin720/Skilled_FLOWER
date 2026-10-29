#!/usr/bin/env python3
"""Apply the pinned TurboVLA crop-v2 transforms to the live SFM cohort.

Run on the raw-data host. Each task must have p1..p5 and ten demos per position.
Original pickles are read once, then small HDF5 intermediates are normalized
and exported to FLOWER's per-frame NPZ format. No training is performed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import sys
import tarfile
import time
import zipfile

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from sfm_preprocessing import (
    CAMERAS, STATE_COLUMNS, STATE_INDICES, Moments, chunk_actions, delta_actions,
    held_targets, initial_start, normalize, preprocess_image, require, sample,
    state_physical,
)

HERE = Path(__file__).resolve().parent
LANG_FOLDER = "lang_annotations"
IMAGE_KEYS = {"rgb_static": "head_cam_d455", "rgb_gripper": "wrist_cam_d405_a"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def references():
    """Reuse the original validator/reader and independent reference functions."""
    scope = dict(np=np, cv2=cv2, Rotation=Rotation, hashlib=hashlib,
                 sample=sample, require=require, CAMERAS=CAMERAS,
                 STATE_INDICES=STATE_INDICES, PreprocessError=ValueError)
    for filename, names in [
        ("turbovla_preprocess.py", {"validate_plan", "HashingReader", "Decoder"}),
        ("csilab_convert.py", {"latest", "numeric", "crop_start", "image_sample"}),
    ]:
        path = HERE / "references" / filename
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        require(len(nodes) == len(names), f"Incomplete reference: {filename}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    spec = importlib.util.spec_from_file_location("held_reference", HERE / "references/gripper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scope["GripperLabels"] = module.GripperLabels
    return scope


def inventory(root, plan):
    tasks = plan["dataset"]["tasks"]
    expected = {f"{task}_p{position}" for task in tasks for position in range(1, 6)}
    found = {p.name for p in root.iterdir() if p.is_dir() and "_p" in p.name}
    require(found == expected, f"Unexpected position folders: missing={expected-found}, extra={found-expected}")
    records, configs, contracts = [], [], []
    for folder in sorted(expected):
        paths = sorted((root / folder).rglob("*.pkl"))
        require(len(paths) == 10, f"Expected 10 demos in {folder}, found {len(paths)}")
        task, position = folder.rsplit("_p", 1)
        for path in paths:
            stat = path.stat()
            records.append(dict(path=str(path.relative_to(root)), bytes=stat.st_size,
                                mtime_ns=stat.st_mtime_ns, task=task, position=int(position)))
        config_paths = sorted((root / folder).rglob("env_config*.json"))
        require(config_paths, f"No collection config for {folder}")
        for path in config_paths:
            config = json.loads(path.read_text())
            train = config.get("train", {})
            contract = dict(train={key: train.get(key) for key in ("arm_type", "hz", "proprio_keys")},
                            env={"right": {"gripper_control": config.get("env", {}).get("right", {}).get("gripper_control", {})}},
                            gripper_interfaces=config.get("gripper_interfaces"))
            contracts.append(contract)
            configs.append(dict(path=str(path.relative_to(root)), sha256=sha256(path), contract=contract))
    require(len(records) == 450 and all(c == contracts[0] for c in contracts), "Cohort/collection contracts differ")
    collection = contracts[0]
    require(collection["train"] == dict(arm_type="right", hz=10, proprio_keys=[
        "right/gripper_pose", "right/tcp_force", "right/tcp_pose", "right/tcp_torque", "right/tcp_vel"]),
        "Unexpected raw state layout or control frequency")
    require(collection["gripper_interfaces"] is None and
            collection["env"]["right"]["gripper_control"]["mode"] == "binary", "Unexpected gripper semantics")
    return dict(records=records, configs=configs, collection=collection,
                excluded_directories=sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in expected),
                total_bytes=sum(r["bytes"] for r in records))


def collect_one(job):
    root, output, index, record, plan, collection = job
    cv2.setNumThreads(1)
    source = Path(root) / record["path"]
    target = Path(output) / "physical" / f"episode_{index:04d}.h5"
    audit = target.with_suffix(".json")
    if target.exists() and audit.exists():
        result = json.loads(audit.read_text())
        require(result["source"] == record["path"] and result["source_bytes"] == record["bytes"]
                and result["source_mtime_ns"] == record["mtime_ns"], "Resume input differs")
        require(sha256(target) == result["physical_sha256"], "Resume intermediate is corrupt")
        return result
    start_time = time.monotonic()
    before = source.stat()
    require((before.st_size, before.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), "Raw file changed")
    ref = references()
    with source.open("rb") as stream:
        reader = ref["HashingReader"](stream)
        rows = pickle.load(reader)
        require(reader.read(1) == b"" and reader.bytes == before.st_size, "Trailing/truncated pickle")
    after = source.stat()
    require((after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns), "Raw file changed during read")
    require(isinstance(rows, list) and len(rows) >= 5, "Expected one nonempty episode")
    terminals = np.asarray([bool(r.get("dones", False) or r.get("terminated", False) or r.get("truncated", False)) for r in rows])
    require(terminals[-1] and terminals.sum() == 1, "Expected exactly one complete episode per pickle")
    require({r.get("infos", {}).get("action_format") for r in rows} == {"absolute_tcp_xyz_rpy_gripper"}, "Unexpected action format")
    raw_state = np.stack([sample(r["observations"]["state"], (19,), "state") for r in rows])
    targets = np.stack([sample(r["actions"], (7,), "actions") for r in rows])
    timestamps = np.asarray([r["time"] for r in rows], np.float64)
    require(np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0), "Invalid timestamps")
    cfg = plan["dataset"]["frame_selection"]["detection_config"]
    start, motion = initial_start(raw_state, cfg)
    ref_start, _ = ref["crop_start"](rows, ref["Decoder"](), cfg)
    require(start == ref_start, "Initial trim differs from csilab reference")
    require(start is not None and len(rows) - start >= 5, f"No usable motion: {source}")
    held, _, receipt_present, fallback = held_targets(rows, raw_state, collection["env"]["right"]["gripper_control"])
    ref_held = ref["GripperLabels"](rows, collection, ref["Decoder"](), dict(
        source_mode="binary", mode="absolute", command_range=[-1., 1.],
        observation_range=[0., 1.], grasp={"enabled": False}), ["right"]).labels[:, 0]
    require(np.array_equal(held, ref_held), "Held targets differ from pinned FAFLOW reference")
    delta = delta_actions(raw_state, targets, held)
    xyz_error = float(np.max(np.abs(raw_state[:, 4:7] + delta[:, :3] - targets[:, :3])))
    rotations = Rotation.from_rotvec(delta[:, 3:6]) * Rotation.from_euler("xyz", raw_state[:, 7:10])
    rot_error = float(np.max((rotations.inv() * Rotation.from_euler("xyz", targets[:, 3:6])).magnitude()))
    require(xyz_error < 1e-10 and rot_error < 1e-10, "Action geometry round-trip failed")
    ft = np.stack([sample(r["observations"]["right_force_history"], (50, 6), "FT") for r in rows[start:]])
    require(raw_state.dtype == np.float32 and ft.dtype == np.float32, "Unexpected numeric dtype")
    count = len(rows) - start
    numeric = dict(robot_obs=state_physical(raw_state[start:]), right_force_history=ft,
                   rel_actions=delta[start:], source_state=raw_state[start:], source_actions=targets[start:],
                   source_indices=np.arange(start, len(rows), dtype=np.int64), timestamps=timestamps[start:])
    temporary = target.with_suffix(".h5.tmp")
    pixel_hashes = {}
    with h5py.File(temporary, "w") as f:
        for key, value in numeric.items():
            f.create_dataset(key, data=value, compression="lzf")
        for key, camera in IMAGE_KEYS.items():
            ds = f.create_dataset(key, (count, 224, 224, 3), dtype=np.uint8,
                                  chunks=(1, 224, 224, 3), compression="lzf")
            digest = hashlib.sha256()
            entry = dict(crop=plan["vision"]["crop_tlhw"][camera], resize=[224, 224],
                         resize_mode="letterbox", interpolation="area", input_color="rgb")
            for i, row in enumerate(rows[start:]):
                image = preprocess_image(row["observations"][camera], camera, plan["vision"])
                require(np.array_equal(image, ref["image_sample"](row["observations"][camera], entry)), "Image differs from csilab reference")
                ds[i] = image
                digest.update(image.tobytes())
            pixel_hashes[key] = digest.hexdigest()
    with h5py.File(temporary, "r") as f:
        for key, value in numeric.items():
            require(np.array_equal(f[key][:], value), f"Intermediate readback differs: {key}")
        for key, digest in pixel_hashes.items():
            require(hashlib.sha256(f[key][:].tobytes()).hexdigest() == digest, "Image readback differs")
    temporary.replace(target)
    result = dict(source=record["path"], source_bytes=record["bytes"], source_mtime_ns=record["mtime_ns"],
                  source_sha256=reader.digest.hexdigest(), task=record["task"], position=record["position"],
                  instruction=plan["dataset"]["instruction_mapping"][record["task"]],
                  physical_file=str(target.relative_to(output)), physical_sha256=sha256(target),
                  original_frames=len(rows), frames=count, start=start, detected_motion_frame=motion,
                  all_zero_ft_windows=int(np.all(ft == 0, axis=(1, 2)).sum()),
                  receipt_frames_all=int(receipt_present.sum()), receipt_fallback_frames_all=fallback,
                  held_counts={str(v): int((held[start:] == v).sum()) for v in (-1., 1.)},
                  delta_geometry_max_error_m=xyz_error, delta_geometry_max_error_rad=rot_error,
                  processed_image_pixel_sha256=pixel_hashes,
                  reference_initial_start_matches=True, reference_held_labels_match=True,
                  reference_image_pixels_match=True, seconds=time.monotonic() - start_time)
    write_json(audit, result)
    return result


def fit_statistics(output, records, plan):
    state, ft, action = Moments(), Moments(), Moments()
    for row in records:
        with h5py.File(output / row["physical_file"], "r") as f:
            state.add(f["robot_obs"][:])
            ft.add(f["right_force_history"][:])
            chunks, valid = chunk_actions(f["rel_actions"][:], plan["action"]["chunk_length"])
            action.add(chunks[valid][:, :6])
    return dict(version=1, state=state.result("standard", 1e-6),
                ft=ft.result("standard", plan["force_torque"]["normalization"]["std_floor"]),
                action_pose=action.result("symlog_minmax", 1e-6, 20.),
                gripper=dict(method="none", closed=-1., open=1.),
                state_columns=list(STATE_COLUMNS), ft_columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                fit_scope="all retained training frames; all 50 FT samples including zeros; valid H=5 action entries",
                ft_application_count=1, live_cohort=True)


def save_npz(path, payload):
    """Low compression level reduces CPU time while keeping exact pixels."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in payload.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def export_one(job):
    output, row, stats = job
    output = Path(output)
    frame_dir = output / "dataset/training"
    error = dict(state=0., ft=0., action_pose=0.)
    digest = hashlib.sha256()
    with h5py.File(output / row["physical_file"], "r") as f:
        state = normalize(f["robot_obs"][:], stats["state"]).astype(np.float32)
        ft = normalize(f["right_force_history"][:], stats["ft"]).astype(np.float32)
        actions = f["rel_actions"][:]
        normalized = actions.copy()
        normalized[:, :6] = normalize(actions[:, :6], stats["action_pose"])
        normalized = normalized.astype(np.float32)
        for name, values, original, coefficients in [
            ("state", state, f["robot_obs"][:], stats["state"]),
            ("ft", ft, f["right_force_history"][:], stats["ft"]),
            ("action_pose", normalized[:, :6], actions[:, :6], stats["action_pose"]),
        ]:
            restored = normalize(values, coefficients, inverse=True)
            np.testing.assert_allclose(restored, original, atol=2e-6, rtol=1e-6)
            error[name] = float(np.max(np.abs(restored - original)))
        require(np.array_equal(normalized[:, 6], actions[:, 6]), "Gripper changed by normalization")
        require(np.max(np.abs(normalized)) <= 1.00001, "Actions exceed normalized range")
        for i in range(row["frames"]):
            filename = f"episode_{row['start_frame'] + i:07d}.npz"
            payload = dict(rgb_static=f["rgb_static"][i], rgb_gripper=f["rgb_gripper"][i],
                           robot_obs=state[i], rel_actions=normalized[i], right_force_history=ft[i],
                           scene_obs=np.zeros(24, dtype=np.float32),
                           source_index=f["source_indices"][i], timestamp=f["timestamps"][i])
            path = frame_dir / filename
            save_npz(path, payload)
            with np.load(path, allow_pickle=False) as stored:
                require(set(stored.files) == set(payload), "NPZ keys differ")
                for key, value in payload.items():
                    require(np.array_equal(stored[key], value), f"NPZ readback differs: {filename}/{key}")
            digest.update(filename.encode() + b"\0" + bytes.fromhex(sha256(path)))
    return dict(row, frame_files_sha256=digest.hexdigest(), normalization_roundtrip_max_errors=error, verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=HERE / "experiment_plan_crop_v2.yaml")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-demos", type=int, help="Smoke run only; not production statistics")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, output = args.raw_root.resolve(), args.output.resolve()
    require(root != output and root not in output.parents and output not in root.parents, "Raw/output paths overlap")
    require(1 <= args.workers <= 4, "Use 1..4 workers (large raw pickles)")
    require(not output.exists() or args.resume, "Output exists; pass --resume or use a new directory")
    for filename, info in json.loads((HERE / "source_provenance.json").read_text()).items():
        require(sha256(HERE / filename) == info["sha256"], f"Pinned source changed: {filename}")
    plan = yaml.safe_load(args.plan.read_text())
    references()["validate_plan"](plan)
    inv = inventory(root, plan)
    sources = {str(p.relative_to(HERE)): sha256(p) for p in HERE.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    invocation = dict(raw_root=str(root), sources=sources, inventory=inv,
                      limit_demos=args.limit_demos, plan_sha256=sha256(args.plan))
    if (output / "invocation.json").exists():
        require(json.loads((output / "invocation.json").read_text()) == invocation, "Resume config/code/inventory differs")
    for folder in ["physical", "dataset/training", "dataset/source_snapshot", "archives"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    write_json(output / "invocation.json", invocation)
    dataset = output / "dataset"
    for filename in sources:
        dest = dataset / "source_snapshot" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / filename, dest)
    write_json(dataset / "raw_inventory.json", inv)
    write_json(dataset / "preprocessing_config.json", plan)
    write_json(dataset / "run.json", dict(started_at_utc=utc(), raw_root=str(root),
               command=sys.argv, raw_data_modified=False, training_started=False,
               smoke_test=args.limit_demos is not None, source_sha256=sources,
               python=sys.version, numpy=np.__version__, opencv=cv2.__version__, h5py=h5py.__version__))
    records = inv["records"]
    if args.limit_demos is not None:
        require(args.limit_demos > 0, "Smoke limit must be positive")
        records = records[:args.limit_demos]
    started = time.monotonic()

    def progress(phase, completed, **kwargs):
        value = dict(phase=phase, completed=completed, total=len(records),
                     elapsed_seconds=round(time.monotonic() - started, 1), updated_at_utc=utc(), **kwargs)
        write_json(output / "status.json", value)
        print(json.dumps(value), flush=True)

    context = multiprocessing.get_context("spawn")
    try:
        jobs = [(str(root), str(output), i, r, plan, inv["collection"]) for i, r in enumerate(records)]
        collected = []
        progress("collecting", 0, source_bytes=inv["total_bytes"])
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for row in pool.map(collect_one, jobs):
                collected.append(row)
                progress("collecting", len(collected), last_source=row["source"],
                         frames=sum(r["frames"] for r in collected))
        frame = 0
        for row in collected:
            row.update(start_frame=frame, end_frame=frame + row["frames"] - 1)
            frame += row["frames"]
        write_json(dataset / "source_manifest.json", collected)
        progress("fitting_statistics", len(collected))
        stats = fit_statistics(output, collected, plan)
        require(stats["state"]["count"] == frame and stats["ft"]["count"] == frame * 50, "Statistics count mismatch")
        require(stats["action_pose"]["count"] == 5 * frame - 10 * len(collected), "Action chunk count mismatch")
        write_json(dataset / "normalization.json", stats)
        complete, packages = [], []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for task in plan["dataset"]["tasks"]:
                task_rows = [r for r in collected if r["task"] == task]
                if not task_rows:
                    continue
                for row in pool.map(export_one, [(str(output), r, stats) for r in task_rows]):
                    complete.append(row)
                    progress("exporting_and_verifying", len(complete), last_source=row["source"])
                path = output / "archives" / f"{task}.tar"
                temporary = path.with_suffix(".tar.tmp")
                with tarfile.open(temporary, "w") as archive:
                    for row in task_rows:
                        for i in range(row["start_frame"], row["end_frame"] + 1):
                            relative = f"training/episode_{i:07d}.npz"
                            archive.add(dataset / relative, arcname=relative, recursive=False)
                temporary.replace(path)
                package = dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path), task=task)
                packages.append(package)
                write_json(output / "archives" / f"{task}.json", package)
                progress("task_packaged", len(complete), archive=package)
        boundaries = np.asarray([[r["start_frame"], r["end_frame"]] for r in complete], dtype=np.int64)
        np.save(dataset / "training/ep_start_end_ids.npy", boundaries)
        lang = dict(language=dict(ann=[r["instruction"] for r in complete], task=[r["task"] for r in complete],
                                 emb=np.zeros((len(complete), 1, 1), dtype=np.float32)),
                    info=dict(indx=boundaries.tolist(), episodes=list(range(len(complete)))))
        (dataset / "training" / LANG_FOLDER).mkdir(exist_ok=True)
        np.save(dataset / "training" / LANG_FOLDER / "auto_lang_ann.npy", lang)
        extracted = dataset / "training/extracted"
        extracted.mkdir(exist_ok=True)
        action_array = np.lib.format.open_memmap(extracted / "ep_rel_actions.npy", mode="w+", dtype=np.float32, shape=(frame, 7))
        for row in complete:
            with h5py.File(output / row["physical_file"], "r") as f:
                actions = f["rel_actions"][:]
                actions[:, :6] = normalize(actions[:, :6], stats["action_pose"])
                action_array[row["start_frame"]:row["end_frame"] + 1] = actions.astype(np.float32)
        action_array.flush()
        del action_array
        (extracted / "ep_npz_names.list").write_text("".join(f"{i}\n" for i in range(frame)))
        contract = dict(schema="flower_calvin_npz_sfm_v1", normalized=True,
                        rgb_keys=list(IMAGE_KEYS), image_shape=[224, 224, 3], image_color_order="RGB",
                        robot_obs_columns=list(STATE_COLUMNS), raw_state_indices=list(STATE_INDICES),
                        state_conditions={c["id"]: dict(indices=[STATE_INDICES.index(i) for i in c["retained_raw_indices"]],
                                                       dimension=c["dimension_after_ft_removal"])
                                          for c in plan["state"]["conditions"]},
                        right_force_history_shape=[50, 6], ft_already_normalized=True,
                        action_dim=7, action_storage="per_timestep",
                        action_horizon_source="training_config.act_seq_len",
                        normalization_fit_action_horizon=plan["action"]["chunk_length"],
                        action_frame="robot_base", action_rotation="rotvec_rad",
                        action_reference="measured TCP at each corresponding action timestep",
                        gripper="held command target, closed=-1/open=+1", end_indices="inclusive",
                        action_windows="complete windows within one demo; no zero padding without a loss mask",
                        lang_folder=LANG_FOLDER, language_embeddings="unused compatibility placeholders; FLOWER consumes lang_text",
                        split="all_450_train_separate_evaluation", validation_present=False,
                        normalization_file="normalization.json", normalization_sha256=sha256(dataset / "normalization.json"),
                        image_preprocessing=plan["vision"], instruction_mapping=plan["dataset"]["instruction_mapping"],
                        prohibit_additional_crop=True, prohibit_repeated_numeric_normalization=True)
        write_json(dataset / "data_contract.json", contract)
        for record in inv["records"]:
            stat = (root / record["path"]).stat()
            require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), "Raw source changed")
        for config in inv["configs"]:
            require(sha256(root / config["path"]) == config["sha256"], "Collection config changed")
        counts = dict(Counter(r["task"] for r in complete))
        if args.limit_demos is None:
            require(counts == {task: 50 for task in plan["dataset"]["tasks"]}, "Expected 50 demos per task")
        manifest = dict(complete=True, smoke_test=args.limit_demos is not None, episodes=len(complete),
                        frames=frame, original_frames=sum(r["original_frames"] for r in complete),
                        removed_initial_frames=sum(r["start"] for r in complete), task_demo_counts=counts,
                        all_zero_ft_windows=sum(r["all_zero_ft_windows"] for r in complete),
                        receipt_fallback_frames_all=sum(r["receipt_fallback_frames_all"] for r in complete),
                        raw_data_modified=False, training_started=False, excluded_directories=inv["excluded_directories"],
                        raw_source_bytes=sum(r["source_bytes"] for r in complete),
                        finished_at_utc=utc(), elapsed_seconds=time.monotonic() - started, episodes_detail=complete)
        write_json(dataset / "manifest.json", manifest)
        metadata = output / "archives/metadata.tar"
        temporary = metadata.with_suffix(".tar.tmp")
        with tarfile.open(temporary, "w") as archive:
            for path in sorted(dataset.rglob("*")):
                if path.is_file() and path.suffix != ".npz":
                    archive.add(path, arcname=str(path.relative_to(dataset)), recursive=False)
        temporary.replace(metadata)
        packages.append(dict(file=metadata.name, bytes=metadata.stat().st_size, sha256=sha256(metadata)))
        write_json(output / "archives/packages.json", dict(complete=True, packages=packages, frames=frame, episodes=len(complete)))
        progress("complete", len(complete), frames=frame, archive_bytes=sum(p["bytes"] for p in packages))
    except BaseException as error:
        progress("failed", 0, error=repr(error))
        raise


if __name__ == "__main__":
    main()
