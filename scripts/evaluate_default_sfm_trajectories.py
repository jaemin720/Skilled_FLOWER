#!/usr/bin/env python3
"""CPU-only, recorded-observation action replay and physical-unit PDF report.

Every input is a complete training-window start. Plot the first action from
each predicted chunk, decoded against that observation's recorded TCP pose.
Predicted actions never replace the following recorded observation.
"""

import os

# Set before importing torch, including in spawned workers. Never use a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/flower-hf-modules")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flower-mpl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torch
from torchvision.transforms import Compose, Normalize

from flower.utils.transforms import ScaleImageTensor
from flower.datasets.sfm_proprioception import config_proprio_source, load_physical_proprio
from preprocess.default_sfm.sfm_preprocessing import normalize


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_episodes(manifest, demos_per_task):
    grouped = defaultdict(list)
    for episode_id, row in enumerate(manifest["episodes_detail"]):
        grouped[row["task"]].append(dict(row, episode_id=episode_id))
    selected = []
    for task in sorted(grouped):
        group = grouped[task]
        count = len(group) if demos_per_task == 0 else demos_per_task
        if not 1 <= count <= len(group):
            raise ValueError(f"Invalid demonstration count for {task}: {count}")
        indices = np.linspace(0, len(group) - 1, count).astype(int)
        for index in indices:
            selected.append(dict(group[int(index)], task_demo_number=int(index) + 1))
    return selected


def decode_targets(state_physical, actions_normalized, statistics):
    """Delta translation is in the base frame, rotation left-multiplies TCP."""
    action = np.asarray(actions_normalized, dtype=np.float64).copy()
    action[..., :6] = normalize(action[..., :6], statistics["action_pose"], inverse=True)
    xyz = state_physical[:, 1:4] + action[:, :3]
    rotation = (Rotation.from_rotvec(action[:, 3:6])
                * Rotation.from_euler("xyz", state_physical[:, 4:7]))
    return action, xyz, rotation.as_quat()


def metric_arrays(record):
    xyz = np.linalg.norm(record["pred_target_xyz"] - record["gt_target_xyz"], axis=1) * 1000
    rotation = (Rotation.from_quat(record["gt_target_quat"]).inv()
                * Rotation.from_quat(record["pred_target_quat"])).magnitude() * 180 / np.pi
    grip_error = np.abs(record["pred_action_physical"][:, 6] - record["gt_action_physical"][:, 6])
    grip_correct = ((record["pred_action_physical"][:, 6] >= 0)
                    == (record["gt_action_physical"][:, 6] >= 0))
    return dict(xyz_error_mm=xyz, orientation_error_deg=rotation,
                gripper_abs_error=grip_error, gripper_correct=grip_correct)


def summarize(errors):
    xyz, rot = errors["xyz_error_mm"], errors["orientation_error_deg"]
    return dict(frames=len(xyz), xyz_mean_mm=float(xyz.mean()),
                xyz_median_mm=float(np.median(xyz)), xyz_p95_mm=float(np.percentile(xyz, 95)),
                xyz_max_mm=float(xyz.max()), xyz_rmse_mm=float(np.sqrt(np.mean(xyz ** 2))),
                orientation_mean_deg=float(rot.mean()), orientation_p95_deg=float(np.percentile(rot, 95)),
                orientation_max_deg=float(rot.max()),
                gripper_accuracy=float(np.mean(errors["gripper_correct"])),
                gripper_mae=float(np.mean(errors["gripper_abs_error"])))


def image_transforms(cfg):
    result = {}
    for key in ("rgb_static", "rgb_gripper"):
        transforms = [hydra.utils.instantiate(spec) for spec in cfg.datamodule.transforms.val[key]]
        if not all(isinstance(t, (ScaleImageTensor, Normalize)) for t in transforms):
            raise ValueError("Replay only permits scaling/normalization of the existing 224x224 crops")
        result[key] = Compose(transforms)
    return result


def read_batch(root, frame_ids, instruction, keep_indices, transforms, physical_proprio=None):
    rows = []
    for frame_id in frame_ids:
        with np.load(root / "training" / f"episode_{frame_id:07d}.npz", allow_pickle=False) as stored:
            rows.append({key: stored[key].copy() for key in stored.files})
    stack = lambda key: np.stack([row[key] for row in rows])
    raw = {key: stack(key) for key in ("robot_obs", "rel_actions", "timestamp", "source_index")}
    # Keep raw['robot_obs'] standardized for the existing report's physical
    # decoding; select physical inputs only for the model when training used it.
    model_state = raw["robot_obs"] if physical_proprio is None else np.array(physical_proprio[frame_ids])
    batch = {
        "rgb_obs": {},
        "robot_obs": torch.from_numpy(model_state[:, keep_indices]).unsqueeze(1),
        "right_force_history": torch.from_numpy(stack("right_force_history")).unsqueeze(1),
        "lang_text": [instruction] * len(rows),
    }
    for key in ("rgb_static", "rgb_gripper"):
        pixels = stack(key)
        if pixels.shape[1:] != (224, 224, 3) or pixels.dtype != np.uint8:
            raise ValueError(f"Unexpected stored image geometry/dtype for {key}")
        batch["rgb_obs"][key] = transforms[key](torch.from_numpy(pixels).permute(0, 3, 1, 2)).unsqueeze(1)
    return batch, raw


def load_cpu_model(options):
    from pytorch_lightning import seed_everything
    from flower.models.flower import FLOWERVLA

    torch.set_num_threads(options["threads"])
    torch.set_num_interop_threads(1)
    cfg = OmegaConf.load(options["training_config"])
    seed_everything(int(cfg.seed), workers=True)
    checkpoint = torch.load(options["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    hparams = dict(checkpoint["hyper_parameters"])
    hparams["load_pretrained"] = False
    # Use the original cached Florence constructor, with the training seed.
    # prompt_embeds is a plain frozen tensor absent from state_dict. Rebuilding
    # from random VLM config or replacing it with the trained token would change
    # the prompt that this training run actually used.
    model = FLOWERVLA(**hparams)
    original = checkpoint["state_dict"]
    if list(model.state_dict()) != list(original):
        raise ValueError("Model/checkpoint key order differs; refusing positional EMA loading")
    if options["weights"] == "ema":
        ema = checkpoint["callbacks"]["EMA"]
        if ema["cur_step"] != checkpoint["global_step"]:
            raise ValueError("EMA step does not match checkpoint step")
        values = ema["ema_weights"]
        if len(values) != len(original):
            raise ValueError("Incomplete EMA state")
        state = dict(zip(original, values))
    else:
        state = original
    for key, tensor in model.state_dict().items():
        if state[key].shape != tensor.shape or state[key].dtype != tensor.dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {key}")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    assert not torch.cuda.is_initialized()
    audit = dict(device="cpu", dtype="float32", state_tensors=len(state), strict_load=True,
                 weights=options["weights"], epoch=int(checkpoint["epoch"]) + 1,
                 global_step=int(checkpoint["global_step"]),
                 frozen_prompt_sha256=hashlib.sha256(model.prompt_embeds.detach().numpy().tobytes()).hexdigest(),
                 prompt_restoration="Original cached Florence constructor and training seed, before checkpoint load",
                 train_seed=int(cfg.seed), action_horizon=model.act_window_size,
                 proprio_source=config_proprio_source(cfg),
                 sampling_steps=model.num_sampling_steps, proprio_dim=model.lowdim_obs_dim,
                 ft_history_length=model.force_history_length, cuda_initialized=False)
    del state, original, checkpoint
    if options["weights"] == "ema":
        del ema, values
    gc.collect()
    return model, cfg, audit


def save_episode_csv(path, record):
    columns = ["source_index", "time_s"]
    arrays = [record["source_index"][:, None], record["time_s"][:, None]]
    for key, names in (("state_physical", ["gripper", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"]),
                       ("gt_action_physical", ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]),
                       ("pred_action_physical", ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]),
                       ("gt_target_xyz", ["x", "y", "z"]), ("pred_target_xyz", ["x", "y", "z"])):
        columns.extend(f"{key}_{name}" for name in names)
        arrays.append(record[key])
    for key, value in metric_arrays(record).items():
        columns.append(key)
        arrays.append(value[:, None])
    np.savetxt(path, np.concatenate(arrays, axis=1), delimiter=",", header=",".join(columns), comments="", fmt="%.10g")


def replay_worker(worker_id, episodes, options):
    output = Path(options["output"])
    status_path = output / "workers" / f"worker_{worker_id:02d}.json"
    start = time.monotonic()
    completed_frames = 0
    try:
        os.nice(max(0, 10 - os.nice(0)))
        write_json(status_path, dict(state="loading", frames_done=0, worker_id=worker_id))
        model, cfg, audit = load_cpu_model(options)
        write_json(output / "workers" / f"model_audit_{worker_id:02d}.json", audit)
        root = Path(cfg.root_data_dir)
        physical_proprio = load_physical_proprio(root) if config_proprio_source(cfg) == "physical" else None
        statistics = json.loads((root / "normalization.json").read_text())
        transforms = image_transforms(cfg)
        keep = [i for a, b in cfg.datamodule.proprioception_dims.keep_indices for i in range(a, b)]
        assert len(keep) == model.lowdim_obs_dim
        extracted = np.load(root / "training/extracted/ep_rel_actions.npy", mmap_mode="r")
        with torch.inference_mode():
            for episode in episodes:
                stem = f"episode_{episode['episode_id']:04d}"
                destination = output / "episodes" / f"{stem}.npz"
                count = episode["frames"] - model.act_window_size + 1
                if destination.exists():
                    with np.load(destination, allow_pickle=False) as stored:
                        assert stored["pred_actions_normalized"].shape == (count, model.act_window_size, 7)
                    completed_frames += count
                    continue
                predictions, ground_truth, states, timestamps, source_indices = [], [], [], [], []
                for offset in range(0, count, options["batch_size"]):
                    frame_ids = list(range(episode["start_frame"] + offset,
                                           episode["start_frame"] + min(count, offset + options["batch_size"])))
                    batch, raw = read_batch(root, frame_ids, episode["instruction"], keep, transforms, physical_proprio)
                    np.testing.assert_array_equal(raw["rel_actions"], extracted[frame_ids])
                    assert np.array_equal(raw["source_index"],
                                          episode["start"] + np.arange(offset, offset + len(frame_ids)))
                    noise = torch.stack([torch.randn(model.act_window_size, model.action_dim,
                                                     generator=torch.Generator(device="cpu").manual_seed(options["seed"] + fid))
                                         for fid in frame_ids])
                    features = model.encode_observations(batch)
                    actions = model.sample_actions(noise, features, inference=True)
                    assert actions.device.type == "cpu" and actions.shape == (len(frame_ids), model.act_window_size, 7)
                    if not torch.isfinite(actions).all():
                        raise ValueError(f"Nonfinite prediction in episode {episode['episode_id']}")
                    predictions.append(actions.numpy().copy())
                    ground_truth.append(raw["rel_actions"])
                    states.append(raw["robot_obs"])
                    timestamps.append(raw["timestamp"])
                    source_indices.append(raw["source_index"])
                    completed_frames += len(frame_ids)
                    write_json(status_path, dict(state="running", worker_id=worker_id, frames_done=completed_frames,
                                                 episode_id=episode["episode_id"], episode_frames_done=offset + len(frame_ids),
                                                 episode_frames=count, elapsed_seconds=time.monotonic() - start))
                predicted = np.concatenate(predictions)
                target = np.concatenate(ground_truth)
                normalized_state = np.concatenate(states)
                physical_state = normalize(normalized_state, statistics["state"], inverse=True)
                np.testing.assert_allclose(normalize(physical_state, statistics["state"]), normalized_state, atol=1e-6)
                pred_action, pred_xyz, pred_quat = decode_targets(physical_state, predicted[:, 0], statistics)
                gt_action, gt_xyz, gt_quat = decode_targets(physical_state, target, statistics)
                np.testing.assert_allclose(normalize(gt_action[:, :6], statistics["action_pose"]), target[:, :6], atol=1e-6)
                timestamp = np.concatenate(timestamps)
                assert np.all(np.diff(timestamp) > 0)
                record = dict(pred_actions_normalized=predicted, gt_first_action_normalized=target,
                              state_physical=physical_state, pred_action_physical=pred_action,
                              gt_action_physical=gt_action, pred_target_xyz=pred_xyz, gt_target_xyz=gt_xyz,
                              pred_target_quat=pred_quat, gt_target_quat=gt_quat,
                              timestamp=timestamp, time_s=timestamp - timestamp[0],
                              source_index=np.concatenate(source_indices))
                pending = destination.with_suffix(".tmp.npz")
                np.savez_compressed(pending, **record)
                pending.replace(destination)
                metrics = dict(episode_id=episode["episode_id"], task=episode["task"],
                               position=episode["position"], task_demo_number=episode["task_demo_number"],
                               **summarize(metric_arrays(record)))
                write_json(destination.with_suffix(".json"), metrics)
                save_episode_csv(destination.with_suffix(".csv"), record)
                print(json.dumps(dict(event="episode_completed", worker_id=worker_id, **metrics)), flush=True)
        assert not torch.cuda.is_initialized()
        write_json(status_path, dict(state="completed", worker_id=worker_id, frames_done=completed_frames,
                                     elapsed_seconds=time.monotonic() - start, cuda_initialized=False))
    except BaseException as error:
        write_json(status_path, dict(state="failed", worker_id=worker_id, frames_done=completed_frames,
                                     error=str(error), traceback=traceback.format_exc()))
        raise


def episode_figure(episode, record, index, count, audit):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(5, 2, left=.065, right=.98, bottom=.135, top=.805,
                           width_ratios=[1.15, 1], hspace=.48, wspace=.2)
    trajectory = fig.add_subplot(grid[:3, 0], projection="3d")
    grip = fig.add_subplot(grid[3:, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    xyz_error, orientation_error = fig.add_subplot(grid[3, 1]), fig.add_subplot(grid[4, 1])
    t = record["time_s"]
    measured = record["state_physical"][:, 1:4] * 1000
    truth, pred = record["gt_target_xyz"] * 1000, record["pred_target_xyz"] * 1000
    errors = metric_arrays(record)
    metrics = summarize(errors)
    color = "#2467ba"
    label = f"FLOWER GRU500 ({audit['weights'].upper()})"
    trajectory.plot(*measured.T, color=".6", linestyle="--", lw=.85, label="Recorded TCP position")
    trajectory.plot(*truth.T, color="black", lw=1.25, label="GT requested action target")
    trajectory.plot(*pred.T, color=color, lw=.9, alpha=.9, label=label)
    trajectory.scatter(*measured[0], color="#168547", s=30, label="Recorded start")
    trajectory.scatter(*measured[-1], color="#c3262b", marker="x", s=45, label="Last evaluated observation")
    trajectory.set(xlabel="World X (mm)", ylabel="World Y (mm)", zlabel="World Z (mm)")
    trajectory.set_title("Recorded TCP and predicted command targets", fontsize=12, pad=10)
    trajectory.view_init(elev=24, azim=-58)
    trajectory.set_box_aspect(np.maximum(np.ptp(np.concatenate([measured, truth, pred]), axis=0), 35))
    for axis in (trajectory.xaxis, trajectory.yaxis, trajectory.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.set_tick_params(labelsize=8)
    trajectory.legend(loc="upper left", fontsize=7.7, framealpha=.88)
    for dim, axis in enumerate(axes):
        axis.plot(t, measured[:, dim], color=".65", linestyle="--", lw=.8)
        axis.plot(t, truth[:, dim], color="black", lw=1.2)
        axis.plot(t, pred[:, dim], color=color, lw=.85, alpha=.9)
        axis.set_ylabel(f"Target {'XYZ'[dim]} (mm)", fontsize=9)
    axes[0].set_title("First command from each recorded training observation", fontsize=11)
    axes[-1].set_xlabel("Time from first retained observation (s)", fontsize=9)
    grip.plot(t, record["gt_action_physical"][:, 6], color="black", lw=1.25, label="GT held goal")
    grip.plot(t, record["pred_action_physical"][:, 6], color=color, lw=.85, label=label)
    grip.set(ylim=(-1.12, 1.12), yticks=[-1, 0, 1], yticklabels=["-1 closed", "0", "+1 open"],
             ylabel="Gripper desired goal", xlabel="Time from first retained observation (s)")
    grip.set_title("Held gripper targets; continuous model outputs", fontsize=11)
    grip.legend(loc="best", fontsize=8)
    xyz_error.plot(t, errors["xyz_error_mm"], color=color, lw=.8)
    xyz_error.set_ylabel("XYZ error (mm)", fontsize=9)
    orientation_error.plot(t, errors["orientation_error_deg"], color=color, lw=.8)
    orientation_error.set_ylabel("Rotation error (deg)", fontsize=9)
    orientation_error.set_xlabel("Time from first retained observation (s)", fontsize=9)
    for axis in [grip, *axes, xyz_error, orientation_error]:
        axis.grid(alpha=.18, lw=.5)
        axis.set_xlim(t[0], t[-1])
        axis.tick_params(labelsize=8)
    fig.suptitle(f"FLOWER GRU500 | {episode['task']} p{episode['position']} | episode {index}/{count} | RAW", y=.974, fontsize=15)
    fig.text(.5, .938, Path(episode["source"]).name, ha="center", fontsize=11)
    fig.text(.5, .91, f"{len(t)} evaluated / {episode['frames']} retained / {episode['original_frames']} original frames"
             f" | source starts at frame {episode['start']} | action rate 10 Hz", ha="center", fontsize=9)
    fig.text(.5, .879, f"Mean XYZ target error: {metrics['xyz_mean_mm']:.3f} mm"
             f" | P95: {metrics['xyz_p95_mm']:.3f} mm | mean rotation error: {metrics['orientation_mean_deg']:.3f} deg"
             f" | gripper sign accuracy: {100 * metrics['gripper_accuracy']:.1f}%", ha="center", fontsize=10)
    fig.text(.5, .079, f"H={audit['action_horizon']}; first action at every complete training-window start."
             " Targets = recorded TCP + decoded base-frame delta; rotations use left multiplication.", ha="center", fontsize=8)
    fig.text(.5, .058, f"Recorded RGB, {audit['proprio_dim']}-D state, task instruction and 500 Hz FT history."
             " Recorded-observation replay; no cumulative integration, robot rollout or output smoothing.", ha="center", fontsize=8)
    fig.text(.5, .037, f"CPU FP32 | {audit['sampling_steps']} Euler sampling steps | {audit['weights'].upper()} epoch {audit['epoch']}"
             f" @ {audit['global_step']:,} steps | last H-1 observations excluded | training data, not a held-out evaluation", ha="center", fontsize=8, color=".3")
    return fig


def render_report(output, options, episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    audits = [json.loads(p.read_text()) for p in sorted((output / "workers").glob("model_audit_*.json"))]
    if not audits or any(item != audits[0] for item in audits):
        raise ValueError("Workers disagree on model/checkpoint/prompt provenance")
    audit = audits[0]
    grouped = defaultdict(list)
    all_errors = []
    episode_metrics = []
    for episode in episodes:
        with np.load(output / "episodes" / f"episode_{episode['episode_id']:04d}.npz", allow_pickle=False) as record:
            errors = metric_arrays(record)
        all_errors.append(errors)
        grouped[episode["task"]].append(errors)
        episode_metrics.append(dict(episode_id=episode["episode_id"], task=episode["task"], position=episode["position"], **summarize(errors)))
    combine = lambda values: {key: np.concatenate([v[key] for v in values]) for key in values[0]}
    total = summarize(combine(all_errors))
    tasks = {task: summarize(combine(values)) for task, values in grouped.items()}
    summary = dict(model=audit, total=total, tasks=tasks, episodes=episode_metrics,
                   evaluation="First action, recorded observations, complete training-window starts only",
                   smoothing=False, cumulative_integration=False, robot_rollout=False,
                   data_scope="Training episodes; not held-out performance", checkpoint=options["checkpoint"],
                   noise_seed_rule=f"CPU generator seed = {options['seed']} + global frame index")
    write_json(output / "metrics.json", summary)
    with (output / "episode_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_metrics[0]))
        writer.writeheader()
        writer.writerows(episode_metrics)
    final = output / "flower_sfm_training_action_trajectories_cpu.pdf"
    pending = final.with_suffix(".tmp.pdf")
    with PdfPages(pending) as pdf:
        pdf.infodict().update(Title="FLOWER GRU500: training action trajectories, CPU FP32",
                              Author="FLOWER", Subject=f"{len(episodes)} training demonstrations; unsmoothed first-action replay")
        fig = plt.figure(figsize=(16, 10.5))
        fig.text(.065, .92, "FLOWER: training-data action trajectory comparison", fontsize=22)
        fig.text(.065, .875, f"{len(episodes)} demonstrations / {len(tasks)} tasks / {total['frames']:,} recorded observations", fontsize=15)
        fig.text(.065, .825, f"{audit['weights'].upper()} checkpoint: epoch {audit['epoch']} / step {audit['global_step']:,}"
                 f" | CPU FP32 | {audit['proprio_dim']}-D proprioception + 500 Hz FT GRU / AdaLN", fontsize=12)
        fig.text(.065, .755, f"Mean XYZ error  {total['xyz_mean_mm']:.3f} mm     P95  {total['xyz_p95_mm']:.3f} mm"
                 f"     Mean rotation error  {total['orientation_mean_deg']:.3f} deg"
                 f"     Gripper sign accuracy  {100 * total['gripper_accuracy']:.1f}%", fontsize=13)
        axis = fig.add_axes([.065, .345, .87, .34])
        axis.axis("off")
        rows = [[task, str(value["frames"]), f"{value['xyz_mean_mm']:.3f}", f"{value['xyz_p95_mm']:.3f}",
                 f"{value['orientation_mean_deg']:.3f}", f"{100 * value['gripper_accuracy']:.1f}%"] for task, value in tasks.items()]
        table = axis.table(cellText=rows, colLabels=["Task", "Frames", "XYZ mean (mm)", "XYZ P95 (mm)", "Rotation mean (deg)", "Gripper accuracy"],
                           cellLoc="center", colLoc="center", bbox=[0, 0, 1, 1], colWidths=[.2, .11, .17, .17, .2, .15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d5dce3")
            cell.set_linewidth(.5)
            if row == 0:
                cell.set_facecolor("#e8eff6")
                cell.set_text_props(weight="bold")
        demo_numbers = ", ".join(map(str, sorted({r["task_demo_number"] for r in episodes})))
        positions = ", ".join(map(str, sorted({r["position"] for r in episodes})))
        notes = [f"Selection: task demonstrations {demo_numbers}; collection positions {positions}; no selection by prediction quality.",
                 "The first of 10 predicted actions is decoded at every complete training-window start (4 Euler steps, fixed per-frame noise seed).",
                 "Ground truth is the requested target reconstructed from the stored delta action and recorded TCP. Rotation error is geodesic.",
                 "Existing 224x224 crops; scaling and CLIP normalization only. No color jitter, second crop, extra state/FT normalization or smoothing.",
                 f"Original state/FT statistics and symlog-minmax action inverse are used. All velocities remain in the {audit['proprio_dim']}-D state; FT is a separate input.",
                 "This is recorded-observation replay on training data. Predictions do not change subsequent inputs; these are not robot rollout results."]
        for i, line in enumerate(notes):
            fig.text(.065, .277 - .031 * i, line, fontsize=10, color=".22")
        pdf.savefig(fig)
        fig.savefig(output / "overview.png", dpi=120)
        plt.close(fig)
        for index, episode in enumerate(episodes, 1):
            with np.load(output / "episodes" / f"episode_{episode['episode_id']:04d}.npz", allow_pickle=False) as record:
                fig = episode_figure(episode, record, index, len(episodes), audit)
                pdf.savefig(fig)
                if index == 1 or episode["task"] == "usb" and episode["position"] == 3:
                    fig.savefig(output / f"preview_{episode['task']}_p{episode['position']}.png", dpi=120)
                plt.close(fig)
    pending.replace(final)
    return final, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--demos-per-task", type=int, default=3)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.threads, args.batch_size) < 1:
        parser.error("Workers, threads and batch size must be positive")
    os.nice(10)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("episodes", "workers"):
        (output / name).mkdir(exist_ok=True)
    config_path = args.run_dir.resolve() / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    root = Path(cfg.root_data_dir)
    contract = json.loads((root / "data_contract.json").read_text())
    assert sha256(root / "normalization.json") == contract["normalization_sha256"]
    assert contract["prohibit_additional_crop"] and contract["ft_already_normalized"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["complete"] and not manifest["smoke_test"]
    selected = select_episodes(manifest, args.demos_per_task)
    checkpoint = (args.checkpoint or args.run_dir / "seed_42/saved_models/epoch019.ckpt").resolve()
    assert checkpoint.is_file()
    options = dict(checkpoint=str(checkpoint), training_config=str(config_path), output=str(output),
                   weights=args.weights, threads=args.threads, batch_size=args.batch_size, seed=args.seed)
    identity = dict(options, checkpoint_bytes=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    normalization_sha256=contract["normalization_sha256"],
                    training_config_sha256=sha256(config_path), episodes=[r["episode_id"] for r in selected])
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory belongs to a different evaluation; choose a new output path")
    write_json(identity_path, identity)
    write_json(output / "selection.json", selected)
    shutil.copy2(config_path, output / "training_config.yaml")
    shutil.copy2(root / "normalization.json", output / "normalization.json")
    total_frames = sum(row["frames"] - cfg.act_seq_len + 1 for row in selected)
    start = time.monotonic()
    processes = []
    if not args.render_only:
        count = min(args.workers, len(selected))
        groups, loads = [[] for _ in range(count)], [0] * count
        for episode in sorted(selected, key=lambda row: row["frames"], reverse=True):
            index = int(np.argmin(loads))
            groups[index].append(episode)
            loads[index] += episode["frames"] - cfg.act_seq_len + 1
        ctx = mp.get_context("spawn")
        try:
            for worker_id, group in enumerate(groups):
                write_json(output / "workers" / f"worker_{worker_id:02d}.json",
                           dict(state="pending", worker_id=worker_id, frames_done=0))
                process = ctx.Process(target=replay_worker, args=(worker_id, group, options))
                process.start()
                processes.append(process)
                time.sleep(.5)
            while any(process.is_alive() for process in processes):
                statuses = [json.loads((output / "workers" / f"worker_{i:02d}.json").read_text()) for i in range(count)]
                if any(s["state"] == "failed" for s in statuses) or any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("CPU inference worker failed; see workers/*.json")
                done = sum(s["frames_done"] for s in statuses)
                progress = dict(state="running", updated_at=datetime.now(timezone.utc).isoformat(),
                                frames_done=done, total_frames=total_frames, episodes=len(selected),
                                elapsed_seconds=time.monotonic() - start, workers=count, threads_per_worker=args.threads,
                                device="cpu", dtype="float32")
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                time.sleep(15)
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"CPU worker exited with code {process.exitcode}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
    final, summary = render_report(output, options, selected)
    write_json(output / "progress.json", dict(state="completed", updated_at=datetime.now(timezone.utc).isoformat(),
                                              frames_done=summary["total"]["frames"], total_frames=total_frames,
                                              episodes=len(selected), elapsed_seconds=time.monotonic() - start,
                                              device="cpu", dtype="float32", cuda_initialized=torch.cuda.is_initialized(), pdf=str(final)))
    print(json.dumps(dict(event="completed", pdf=str(final), metrics=summary["total"])), flush=True)


if __name__ == "__main__":
    main()
