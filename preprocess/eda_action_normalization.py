"""
EDA for action distributions before and after dataset-level normalization.

Typical usage:

python preprocess/eda_action_normalization.py \
  --input /path/to/multitask_root \
  --pkl_glob "**/*.pkl" \
  --action_source raw

python preprocess/eda_action_normalization.py \
  --input /path/to/multitask_root \
  --pkl_glob "**/*.pkl" \
  --action_source raw \
  --stats_path /path/to/converted_dataset/action_statistics.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flower_bed.Skill_flower.FLOWER_Calvin_Custom.preprocess.convert_multitask_raw_pkl_to_flower import (  # noqa: E402
    EpisodeRecord,
    build_rel_action,
    load_episode_records,
    parse_slice,
)


ACTION_DIM_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Root directory containing task folders.")
    parser.add_argument("--task_names", nargs="*", default=None, help="Optional subset of task folder names to include.")
    parser.add_argument("--pkl_glob", default="**/*.pkl", help="Glob pattern used inside each task folder.")
    parser.add_argument("--min_episode_steps", type=int, default=11, help="Episodes shorter than this are skipped.")
    parser.add_argument("--action_source", choices=["raw", "state_delta"], default="raw")
    parser.add_argument("--prefer_intervene_action", action="store_true")
    parser.add_argument("--action_slice", default="7:14")
    parser.add_argument("--obs_state_key", default="state")
    parser.add_argument("--state_pose_slice", default="23:29")
    parser.add_argument("--stats_path", type=Path, default=None, help="Optional action_statistics.npz to reuse.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Fallback std floor when computing stats locally.")
    parser.add_argument("--save_json", type=Path, default=None, help="Optional path to save the full EDA summary as JSON.")
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="Optional directory to save histogram and boxplot figures.",
    )
    return parser.parse_args()


def collect_actions(
    records: Sequence[EpisodeRecord],
    *,
    action_source: str,
    state_key: str,
    pose_slice: slice,
    action_slice: slice,
    prefer_intervene_action: bool,
) -> np.ndarray:
    actions: List[np.ndarray] = []
    for record in records:
        for step in record.steps:
            actions.append(
                build_rel_action(
                    step,
                    action_source=action_source,
                    state_key=state_key,
                    pose_slice=pose_slice,
                    action_slice=action_slice,
                    prefer_intervene_action=prefer_intervene_action,
                )
            )
    if not actions:
        raise ValueError("No action steps found.")
    return np.asarray(actions, dtype=np.float32)


def compute_stats(actions: np.ndarray, epsilon: float) -> Dict[str, np.ndarray]:
    std = actions.std(axis=0).astype(np.float32)
    std = np.maximum(std, epsilon).astype(np.float32)
    return {
        "mean": actions.mean(axis=0).astype(np.float32),
        "std": std,
        "min": actions.min(axis=0).astype(np.float32),
        "max": actions.max(axis=0).astype(np.float32),
    }


def normalize(actions: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((actions - stats["mean"]) / stats["std"]).astype(np.float32)


def quantiles(values: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "q01": np.quantile(values, 0.01, axis=0),
        "q05": np.quantile(values, 0.05, axis=0),
        "q25": np.quantile(values, 0.25, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q75": np.quantile(values, 0.75, axis=0),
        "q95": np.quantile(values, 0.95, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def near_zero_ratios(actions: np.ndarray) -> Dict[str, float]:
    pos_norm = np.linalg.norm(actions[:, :3], axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
    full_norm = np.linalg.norm(actions[:, :6], axis=1)
    return {
        "pos_lt_1e-4": float(np.mean(pos_norm < 1e-4)),
        "pos_lt_1e-3": float(np.mean(pos_norm < 1e-3)),
        "pos_lt_1e-2": float(np.mean(pos_norm < 1e-2)),
        "rot_lt_1e-4": float(np.mean(rot_norm < 1e-4)),
        "rot_lt_1e-3": float(np.mean(rot_norm < 1e-3)),
        "rot_lt_1e-2": float(np.mean(rot_norm < 1e-2)),
        "full_lt_1e-4": float(np.mean(full_norm < 1e-4)),
        "full_lt_1e-3": float(np.mean(full_norm < 1e-3)),
        "full_lt_1e-2": float(np.mean(full_norm < 1e-2)),
    }


def zscore_band_ratios(values: np.ndarray) -> Dict[str, np.ndarray]:
    abs_values = np.abs(values)
    return {
        "abs_gt_1": np.mean(abs_values > 1.0, axis=0),
        "abs_gt_2": np.mean(abs_values > 2.0, axis=0),
        "abs_gt_3": np.mean(abs_values > 3.0, axis=0),
    }


def mse_if_zero(actions: np.ndarray) -> np.ndarray:
    return np.mean(np.square(actions), axis=0)


def summarize(actions: np.ndarray) -> Dict[str, object]:
    q = quantiles(actions)
    return {
        "count": int(actions.shape[0]),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": q["q01"].tolist(),
        "q05": q["q05"].tolist(),
        "q25": q["q25"].tolist(),
        "q50": q["q50"].tolist(),
        "q75": q["q75"].tolist(),
        "q95": q["q95"].tolist(),
        "q99": q["q99"].tolist(),
        "mse_if_predict_zero": mse_if_zero(actions).tolist(),
        "near_zero_ratios": near_zero_ratios(actions),
    }


def print_dim_table(raw_actions: np.ndarray, norm_actions: np.ndarray, stats: Dict[str, np.ndarray]) -> None:
    raw_q = quantiles(raw_actions)
    norm_q = quantiles(norm_actions)
    norm_bands = zscore_band_ratios(norm_actions)

    header = (
        "dim        raw_mean   raw_std    raw_q01    raw_q50    raw_q99    "
        "norm_mean  norm_std   |z|>1     |z|>2     |z|>3"
    )
    print("\n[Per-dimension summary]")
    print(header)
    for idx, name in enumerate(ACTION_DIM_NAMES):
        print(
            f"{name:<8} "
            f"{raw_actions[:, idx].mean():>9.5f} "
            f"{raw_actions[:, idx].std():>9.5f} "
            f"{raw_q['q01'][idx]:>10.5f} "
            f"{raw_q['q50'][idx]:>10.5f} "
            f"{raw_q['q99'][idx]:>10.5f} "
            f"{norm_actions[:, idx].mean():>10.5f} "
            f"{norm_actions[:, idx].std():>9.5f} "
            f"{norm_bands['abs_gt_1'][idx]:>9.4f} "
            f"{norm_bands['abs_gt_2'][idx]:>9.4f} "
            f"{norm_bands['abs_gt_3'][idx]:>9.4f}"
        )

    print("\n[Normalization statistics used]")
    for idx, name in enumerate(ACTION_DIM_NAMES):
        print(
            f"{name:<8} mean={stats['mean'][idx]: .6f} std={stats['std'][idx]: .6f} "
            f"min={stats['min'][idx]: .6f} max={stats['max'][idx]: .6f}"
        )


def save_histograms(raw_actions: np.ndarray, norm_actions: np.ndarray, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(ACTION_DIM_NAMES), 2, figsize=(12, 3 * len(ACTION_DIM_NAMES)))
    for idx, name in enumerate(ACTION_DIM_NAMES):
        raw_ax = axes[idx, 0]
        norm_ax = axes[idx, 1]

        raw_ax.hist(raw_actions[:, idx], bins=80, color="#1f77b4", alpha=0.85)
        raw_ax.set_title(f"Raw histogram: {name}")
        raw_ax.set_xlabel(name)
        raw_ax.set_ylabel("count")

        norm_ax.hist(norm_actions[:, idx], bins=80, color="#ff7f0e", alpha=0.85)
        norm_ax.set_title(f"Normalized histogram: {name}")
        norm_ax.set_xlabel(name)
        norm_ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(plot_dir / "action_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_boxplots(raw_actions: np.ndarray, norm_actions: np.ndarray, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].boxplot([raw_actions[:, idx] for idx in range(raw_actions.shape[1])], labels=ACTION_DIM_NAMES, showfliers=False)
    axes[0].set_title("Raw action boxplot")
    axes[0].set_ylabel("value")

    axes[1].boxplot([norm_actions[:, idx] for idx in range(norm_actions.shape[1])], labels=ACTION_DIM_NAMES, showfliers=False)
    axes[1].set_title("Normalized action boxplot")
    axes[1].set_ylabel("z-score")

    fig.tight_layout()
    fig.savefig(plot_dir / "action_boxplots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pose_slice = parse_slice(args.state_pose_slice)
    action_slice = parse_slice(args.action_slice)

    records = load_episode_records(args.input, args.pkl_glob, args.task_names)
    records = [record for record in records if len(record.steps) >= args.min_episode_steps]
    if not records:
        raise ValueError("No episodes left after filtering.")

    actions = collect_actions(
        records,
        action_source=args.action_source,
        state_key=args.obs_state_key,
        pose_slice=pose_slice,
        action_slice=action_slice,
        prefer_intervene_action=args.prefer_intervene_action,
    )

    if args.stats_path is not None:
        stats_npz = np.load(args.stats_path)
        stats = {
            "mean": stats_npz["mean"].astype(np.float32),
            "std": stats_npz["std"].astype(np.float32),
            "min": stats_npz["min"].astype(np.float32) if "min" in stats_npz.files else actions.min(axis=0).astype(np.float32),
            "max": stats_npz["max"].astype(np.float32) if "max" in stats_npz.files else actions.max(axis=0).astype(np.float32),
        }
    else:
        stats = compute_stats(actions, args.epsilon)

    normalized_actions = normalize(actions, stats)

    raw_summary = summarize(actions)
    normalized_summary = summarize(normalized_actions)

    print("[Dataset]")
    print("episodes:", len(records))
    print("steps:", actions.shape[0])
    print("action_source:", args.action_source)
    print("task_names:", sorted({record.task_name for record in records}))

    print_dim_table(actions, normalized_actions, stats)

    print("\n[Raw action near-zero ratios]")
    print(raw_summary["near_zero_ratios"])

    print("\n[Normalized action near-zero ratios]")
    print(normalized_summary["near_zero_ratios"])

    print("\n[MSE if model predicts all zeros]")
    print("raw       :", np.asarray(raw_summary["mse_if_predict_zero"]))
    print("normalized:", np.asarray(normalized_summary["mse_if_predict_zero"]))

    if args.plot_dir is not None:
        save_histograms(actions, normalized_actions, args.plot_dir)
        save_boxplots(actions, normalized_actions, args.plot_dir)
        print(f"\nSaved plots to {args.plot_dir}")

    summary = {
        "episodes": len(records),
        "steps": int(actions.shape[0]),
        "action_source": args.action_source,
        "tasks": sorted({record.task_name for record in records}),
        "stats_used": {key: value.tolist() for key, value in stats.items()},
        "raw": raw_summary,
        "normalized": normalized_summary,
    }

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved EDA summary to {args.save_json}")


if __name__ == "__main__":
    main()
