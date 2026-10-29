#!/usr/bin/env python3
"""Small deterministic offline-training fixtures; no weights or robot imports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess.default_sfm.sfm_preprocessing import normalize, preprocess_image, state_physical


def make_fixture(dataset, output):
    stats = json.loads((dataset / "normalization.json").read_text())
    plan = json.loads((dataset / "preprocessing_config.json").read_text())
    y, x = np.indices((480, 640))
    image = np.stack([(x + y) % 256, (3 * x - y) % 256, (x // 7 + y // 5) % 256], axis=-1).astype(np.uint8)
    data = {}
    states = np.zeros((3, 1, 19), np.float32)
    states[:, 0, 0] = [0., 1., .45]
    states[:, 0, 4:10] = [[.3, -.2, .1, -np.pi, .01, np.pi],
                           [.25, -.4, .15, -1e-9, -.5, -3 * np.pi],
                           [.4, -.1, .2, 2 * np.pi + .1, .5, -1e-9]]
    states[:, 0, 13:] = np.linspace(-.4, .4, 18).reshape(3, 6)
    ft = np.zeros((3, 1, 50, 6), np.float32)
    ft[1, 0] = np.linspace(-10, 10, 300).reshape(50, 6)
    ft[2, 0] = np.sin(np.arange(300).reshape(50, 6))
    data["raw__state"], data["raw__right_force_history"] = states, ft
    data["expected__robot_obs"] = normalize(state_physical(states[:, 0]), stats["state"]).astype(np.float32)
    data["expected__right_force_history"] = normalize(ft[:, 0], stats["ft"]).astype(np.float32)
    for camera, target in (("head_cam_d455", "rgb_static"), ("wrist_cam_d405_a", "rgb_gripper")):
        frames = np.stack([np.roll(image, i * 31, axis=1) for i in range(3)])
        if target == "rgb_gripper":
            frames = frames[..., ::-1].copy()
        data["raw__" + camera] = frames[:, None]
        data["expected__" + target] = np.stack([preprocess_image(f, camera, plan["vision"]) for f in frames])
    actions = np.random.default_rng(20261028).uniform(-1, 1, (10, 7)).astype(np.float32)
    actions[:, 6] = np.resize([-1., 1., 0.], 10)
    physical = actions.astype(np.float64)
    physical[:, :6] = normalize(actions[:, :6], stats["action_pose"], inverse=True)
    data["normalized_actions"], data["physical_actions"] = actions, physical
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(json.dumps(dict(output=str(output), observations=3, action_horizon=10, bytes=output.stat().st_size)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/default_sfm_crop_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/default_sfm/verification.npz")
    args = parser.parse_args()
    make_fixture(args.dataset, args.output)
