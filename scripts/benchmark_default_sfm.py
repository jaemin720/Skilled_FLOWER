#!/usr/bin/env python3
"""Measure full FLOWER DDP batch memory without taking optimizer steps.

Uses real SFM images/state/FT/actions and the existing FLOWER checkpoint.
Florence architecture/tokenizer files must already be cached. Its separate
backbone weights are not loaded: all VLM weights come from the FLOWER checkpoint.
AdamW moments, EMA copies, and conservative update workspace are allocated to
avoid reporting a forward/backward-only memory figure as training capacity.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/flower-hf-modules")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flower-mpl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra
from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate
from transformers import AutoConfig, AutoModelForCausalLM

from flower.models.flower import FLOWERVLA

GIB = 1024 ** 3
FLORENCE_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"


def read_config():
    with hydra.initialize_config_dir(config_dir=str(REPO / "conf")):
        return hydra.compose(config_name="config_default_sfm")


def vlm_from_config(vlm_path, **kwargs):
    config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True, local_files_only=True)
    return AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation=kwargs["attn_implementation"],
        # from_config otherwise inherits Florence's float16 metadata, whereas
        # the production from_pretrained call uses FP32 master parameters.
        torch_dtype=torch.float32,
    )


def load_policy(config):
    if not config.model.load_pretrained:
        raise ValueError("A complete FLOWER checkpoint is required for this benchmark")
    load_weights = FLOWERVLA._load_pretrained_weights
    vlm_path = config.model.vlm_path
    if not Path(vlm_path).is_dir():
        vlm_path = snapshot_download(vlm_path, revision=FLORENCE_REVISION, local_files_only=True)

    def checked_load(model, *args, **kwargs):
        missing, unexpected = load_weights(model, *args, **kwargs)
        missing_vlm = [key for key in missing if key.startswith("vlm.")]
        if missing_vlm:
            raise RuntimeError(f"FLOWER checkpoint does not cover the complete VLM: {missing_vlm}")
        return missing, unexpected

    with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=vlm_from_config), \
            patch.object(FLOWERVLA, "_load_pretrained_weights", checked_load):
        model = hydra.utils.instantiate(config.model, vlm_path=vlm_path)
    # The frozen prompt tensor was created before checkpoint loading. Derive it
    # from the loaded embedding table for this diagnostic run.
    token = torch.tensor(model.tokenizer.convert_tokens_to_ids("<Flow>"))
    model.prompt_embeds = model.vlm.get_input_embeddings()(token).detach()[None, None]
    assert all(param.dtype == torch.float32 for param in model.parameters())
    return model


class PolicyLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        features = self.model.encode_observations(batch)
        loss, _ = self.model.rf_loss(features, batch["actions"])
        return loss


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


def worker(rank, world_size, batch_size, repeats, rendezvous, report_dir):
    torch.set_num_threads(4)
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        config = read_config()
        datamodule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup("fit")
        dataset = datamodule.train_datasets["lang"]
        indices = np.linspace(rank, len(dataset) - 1 - rank, batch_size, dtype=int)
        batch = default_collate([dataset[int(index)] for index in indices])
        for key in ("rgb_static", "rgb_gripper"):
            assert batch["rgb_obs"][key].shape == (batch_size, 1, 3, 224, 224)
        assert batch["actions"].shape == (batch_size, config.act_seq_len, 7)
        assert batch["robot_obs"].shape == (batch_size, 1, 13)
        assert batch["right_force_history"].shape == (batch_size, 1, 50, 6)
        columns = dataset.contract["robot_obs_columns"]
        assert columns == ["gripper", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz"]
        windows = len(dataset)
        del dataset, datamodule
        model = load_policy(config)
        # Match training_calvin.py's SyncBatchNorm and DDP settings.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(f"cuda:{rank}").train()
        batch = to_device(batch, model.device)
        distributed = DistributedDataParallel(PolicyLoss(model), device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            model._get_param_groups(), lr=config.model.optimizer.learning_rate,
            betas=tuple(config.model.optimizer.betas),
        )
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        # Allocate AdamW's lazy state without calling optimizer.step(). Reserving
        # it for unused parameters as well gives a small conservative overestimate.
        for param in parameters:
            optimizer.state[param] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
        ema_weights = [value.detach().clone() for value in model.state_dict().values()]
        # Foreach AdamW may need one parameter-sized temporary list; EMA's loop
        # uses one tensor-sized temporary. Keep both reserved during the peak.
        # Keep these live (not just cached) throughout the pass.
        workspaces = [torch.empty(parameter_bytes, dtype=torch.uint8, device=model.device),
                      torch.empty(max(value.numel() * value.element_size() for value in model.state_dict().values()),
                                  dtype=torch.uint8, device=model.device)]
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
        precision = config.trainer.precision
        if precision not in ("16-mixed", "bf16-mixed"):
            raise ValueError(f"Unsupported benchmark precision: {precision}")
        amp_dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[rank])
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        timings, losses, gradients_finite = [], [], []
        for iteration in range(repeats):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = distributed(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            losses.append(loss.item())
            finite = all(torch.isfinite(p.grad).all().item() for p in parameters if p.grad is not None)
            gradients_finite.append(finite)
            assert torch.isfinite(loss), "Non-finite flow loss"
            if finite:
                grad = model.force_history_encoder.gru.weight_ih_l0.grad
                assert grad is not None and grad.abs().sum() > 0
            scaler.update()
            print(json.dumps({"gpu": rank, "per_gpu_batch": batch_size, "pass": iteration + 1,
                              "seconds": timings[-1], "loss": losses[-1], "finite_gradients": finite}), flush=True)
            del loss
        total = torch.cuda.get_device_properties(rank).total_memory
        report = {
            "rank": rank, "device": torch.cuda.get_device_name(rank),
            "per_gpu_batch_size": batch_size, "global_batch_size": batch_size * world_size,
            "precision": precision, "action_horizon": config.act_seq_len,
            "training_windows": windows, "proprio_dim": 13, "proprio_contains_ft": False,
            "image_shape": [224, 224], "views": 2, "additional_image_crop": False,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "total_gib": total / GIB, "baseline_allocated_gib": baseline / GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "adamw_state_gib": 2 * parameter_bytes / GIB,
            "ema_gib": sum(value.numel() * value.element_size() for value in ema_weights) / GIB,
            "extra_update_workspace_gib": sum(value.numel() for value in workspaces) / GIB,
            "forward_backward_seconds": timings, "losses": losses,
            "finite_gradients": gradients_finite, "optimizer_steps": 0,
            "checkpoint_saved": False, "full_vlm": True,
            "all_vlm_weights_loaded_from_flower_checkpoint": True,
        }
        (Path(report_dir) / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        dist.barrier(device_ids=[rank])
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report", type=Path, default=REPO / ".benchmarks/default_sfm_batch_memory.json")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if min(args.batch_sizes) < 1 or args.repeats < 1:
        parser.error("Batch sizes and repeats must be positive")
    world_size = len(args.gpus.split(","))
    assert torch.cuda.device_count() == world_size
    summary = {"gpus": args.gpus, "optimizer_steps": 0, "cases": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        with tempfile.TemporaryDirectory(prefix="flower-batch-probe-") as directory:
            try:
                mp.spawn(worker, args=(world_size, batch_size, args.repeats,
                                      f"file://{directory}/rendezvous", directory), nprocs=world_size, join=True)
                reports = [json.loads((Path(directory) / f"rank{rank}.json").read_text()) for rank in range(world_size)]
                result = {"per_gpu_batch_size": batch_size, "status": "passed", "ranks": reports}
            except mp.ProcessRaisedException as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                result = {"per_gpu_batch_size": batch_size, "status": "out_of_memory", "error": str(exc)}
                print(json.dumps({"per_gpu_batch_size": batch_size, "status": "out_of_memory"}), flush=True)
        summary["cases"].append(result)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
