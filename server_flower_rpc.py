#!/usr/bin/env python3
"""FLOWER inference for the unchanged Real_Robo.env_rpc_client HTTP/NPZ API.

The client sends cropped uint8 RGB, standardized state, and standardized F/T.
This server applies the saved image transform and proprioception configuration,
and returns normalized actions. A physical-state checkpoint restores state
units before channel selection. The client performs the action conversion.
Importing this module never loads model weights or starts a server.
"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
from flower.datasets.sfm_proprioception import config_proprio_source, physical_from_normalized

from deployment.default_sfm.reference.Real_Robo.env_rpc_server import (
    InferenceBundle, bundle_id, serve_policy, validate_observation,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path("/data/jack/Skilled_FLOWER/checkpoints/default_sfm/flower_sfm_b64_20261028_043405/seed_42/saved_models/epoch019.ckpt")
LOG = logging.getLogger("flower_rpc")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_training_config(checkpoint):
    for directory in Path(checkpoint).resolve().parents:
        path = directory / ".hydra/config.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError("Specify --training-config; no .hydra/config.yaml above checkpoint")


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "bundle_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    bundle = InferenceBundle(path)
    if bundle_id(bundle) != manifest["bundle_id"]:
        raise ValueError("Bundle ID differs from the exported manifest")
    return bundle, manifest


def state_indices(config):
    indices = [i for start, end in config.datamodule.proprioception_dims.keep_indices for i in range(start, end)]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= 13:
        raise ValueError("Training config has invalid SFM state indices")
    return indices


def image_transforms(config):
    import hydra
    from torchvision.transforms import Compose, Normalize
    from flower.utils.transforms import ScaleImageTensor

    transforms = {}
    for key in ("rgb_static", "rgb_gripper"):
        items = [hydra.utils.instantiate(item) for item in config.datamodule.transforms.val[key]]
        if not all(isinstance(item, (ScaleImageTensor, Normalize)) for item in items):
            raise ValueError("RPC inputs are already cropped: only saved image scaling/normalization is allowed")
        transforms[key] = Compose(items)
    return transforms


def observation_batch(observation, instruction, indices, transforms):
    import torch

    batch = dict(rgb_obs={}, robot_obs=torch.from_numpy(observation["robot_obs"][indices].copy())[None, None],
                 right_force_history=torch.from_numpy(observation["right_force_history"].copy())[None, None],
                 lang_text=[instruction])
    for key in ("rgb_static", "rgb_gripper"):
        image = torch.from_numpy(observation[key].copy()).permute(2, 0, 1)[None]
        batch["rgb_obs"][key] = transforms[key](image)[None]
    return batch


def checkpoint_state(checkpoint, model_state, weights):
    original = checkpoint["state_dict"]
    if list(model_state) != list(original):
        raise ValueError("Checkpoint/model state order differs; refusing ambiguous EMA mapping")
    if weights == "ema":
        ema = checkpoint.get("callbacks", {}).get("EMA")
        if not ema or ema.get("cur_step") != checkpoint["global_step"]:
            raise ValueError("Matching EMA weights are unavailable; explicitly select --weights raw if intended")
        if len(ema["ema_weights"]) != len(original):
            raise ValueError("Incomplete EMA state")
        selected = dict(zip(original, ema["ema_weights"]))
    else:
        selected = original
    for key, expected in model_state.items():
        if selected[key].shape != expected.shape or selected[key].dtype != expected.dtype:
            raise ValueError(f"Checkpoint tensor differs: {key}")
    return selected


class FlowerRPCPolicy:
    simulation_only = False

    def __init__(self, model, config, bundle, instructions, *, device, seed=42, default_task=None,
                 precision="float32"):
        import torch

        self.model, self.bundle, self.instructions = model, bundle, instructions
        self.device = torch.device(device)
        self.precision = precision
        self.proprio_source = config_proprio_source(config)
        self.indices = state_indices(config)
        self.transforms = image_transforms(config)
        self.horizon, self.action_dim = int(model.act_window_size), int(model.action_dim)
        if len(self.indices) != model.lowdim_obs_dim or self.action_dim != 7:
            raise ValueError("Model dimensions differ from the training data selection")
        if not model.use_second_view or not model.use_force_history_adaln or model.force_history_length != 50:
            raise ValueError("This SFM adapter requires both views and the trained 50x6 F/T history")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.default_task = default_task
        self.instruction = None
        self.calls = 0

    def reset(self, text="", previous_episode=None):
        text = text.strip()
        if not text and self.default_task:
            text = self.default_task
        instruction = self.instructions.get(text, text)
        if instruction not in self.instructions.values():
            raise ValueError("Choose a trained task or send its exact instruction with --prompt")
        self.instruction = instruction
        self.model.reset()
        LOG.info("Episode reset: %s; previous=%s", instruction, previous_episode)

    def predict(self, observation):
        import torch

        if self.instruction is None:
            raise ValueError("Call /reset with a task instruction before /query")
        observation = validate_observation(self.bundle, observation)
        if self.proprio_source == "physical":
            # Keep the original robot client and wire bundle. Restore only state
            # here, before selecting the checkpoint's 13/11/9 model channels.
            observation = dict(observation, robot_obs=physical_from_normalized(
                observation["robot_obs"], self.bundle.stats["observations"]["robot_obs"]))
        batch = observation_batch(observation, self.instruction, self.indices, self.transforms)
        started = time.monotonic()
        # Match training's mixed precision. Internal frequency/time tensors and
        # flow integration stay float32; autocast handles the learned layers.
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16,
                enabled=self.device.type == "cuda" and self.precision == "float16"):
            features = self.model.encode_observations(batch)
            noise = torch.randn(1, self.horizon, self.action_dim, device=self.device, generator=self.generator)
            actions = self.model.sample_actions(noise, features, inference=True)
            result = actions[0].float().cpu().numpy()
        if result.shape != (self.horizon, 7) or not np.isfinite(result).all() or np.max(np.abs(result)) > 1.00001:
            raise ValueError("Model returned invalid normalized actions")
        self.calls += 1
        if self.calls == 1 or self.calls % 100 == 0:
            LOG.info("Query %d: horizon=%d, inference=%.3fs", self.calls, self.horizon, time.monotonic() - started)
        return result


def load_policy(args, bundle, bundle_manifest):
    # These imports and all device/model operations happen only on explicit start.
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from flower.models.flower import FLOWERVLA

    config_path = args.training_config or find_training_config(args.checkpoint)
    config = OmegaConf.load(config_path)
    dataset_stats = Path(config.root_data_dir) / "normalization.json"
    if not dataset_stats.is_file() or sha256(dataset_stats) != bundle_manifest["source_normalization_sha256"]:
        raise ValueError("Bundle normalization does not match the training dataset")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback is performed")
    device = torch.device(args.device)
    if device.type == "cpu" and args.precision != "float32":
        raise ValueError("Use --precision float32 for CPU inference")
    torch.set_num_threads(args.cpu_threads)
    # prompt_embeds is not in state_dict. Its creation must reproduce the
    # original pretrained Florence constructor and TRAINING seed before load.
    seed_everything(int(config.seed), workers=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    parameters = dict(checkpoint["hyper_parameters"])
    parameters["load_pretrained"] = False
    if args.vlm_path:
        parameters["vlm_path"] = str(args.vlm_path)
    if not Path(parameters["vlm_path"]).is_dir():
        raise FileNotFoundError("The original Florence snapshot must be present locally; specify --vlm-path")
    model = FLOWERVLA(**parameters)
    prompt_hash = hashlib.sha256(model.prompt_embeds.detach().cpu().numpy().tobytes()).hexdigest()
    selected = checkpoint_state(checkpoint, model.state_dict(), args.weights)
    model.load_state_dict(selected, strict=True)
    if model.act_window_size != int(config.act_seq_len):
        raise ValueError("Checkpoint action horizon differs from its saved training config")
    audit = dict(checkpoint=str(args.checkpoint.resolve()), training_config=str(Path(config_path).resolve()),
                 epoch=int(checkpoint["epoch"]) + 1, global_step=int(checkpoint["global_step"]), weights=args.weights,
                 strict_load=True, frozen_prompt_sha256=prompt_hash, train_seed=int(config.seed),
                 action_horizon=model.act_window_size, proprio_dim=model.lowdim_obs_dim,
                 ft_history_length=model.force_history_length, device=str(device), precision=args.precision,
                 parameter_dtype="float32", training_config_sha256=sha256(config_path),
                 proprio_source=config_proprio_source(config),
                 bundle_id=bundle_id(bundle))
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    model.prompt_embeds = model.prompt_embeds.to(device=device, dtype=torch.float32)
    del selected, checkpoint
    gc.collect()
    instructions = json.loads((args.preprocess_bundle / "instructions.json").read_text())
    policy = FlowerRPCPolicy(model, config, bundle, instructions, device=device, seed=args.seed,
                             default_task=args.task, precision=args.precision)
    return policy, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--preprocess-bundle", type=Path, default=ROOT / "datasets/default_sfm_crop_v2/rpc_bundle")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=45587)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16",
                        help="float16 uses CUDA autocast with float32 parameters, as in training")
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--vlm-path", type=Path)
    parser.add_argument("--task", help="Optional task used when the client's reset prompt is empty")
    parser.add_argument("--seed", type=int, default=42, help="Inference noise seed; training seed is restored independently")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--audit-file", type=Path, default=ROOT / "logs/default_sfm_rpc/model_audit.json")
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        parser.error("Set CUDA_VISIBLE_DEVICES to the GPU reserved for this server before starting")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".cache/flower_rpc_hf_modules"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/flower_rpc_matplotlib"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundle, manifest = verify_bundle(args.preprocess_bundle)
    policy, audit = load_policy(args, bundle, manifest)
    args.audit_file.parent.mkdir(parents=True, exist_ok=True)
    args.audit_file.write_text(json.dumps(audit, indent=2) + "\n")
    LOG.info("Model loaded: %s", json.dumps(audit))
    serve_policy(bundle, policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
