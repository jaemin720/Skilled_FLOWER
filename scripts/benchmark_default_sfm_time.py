#!/usr/bin/env python3
"""Compare DDP batch throughput without updating the FLOWER policy.

Real data loading, augmentation, transfer, forward/backward, AdamW, and EMA
are timed. AdamW owns independent copies of the parameters; only those
disposable copies and EMA buffers change. No training run or checkpoint is
created. Epoch estimates exclude checkpoint writes and external logging.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flower-mpl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/flower-hf-modules")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from lightning_fabric.utilities.apply_func import move_data_to_device

from scripts.benchmark_default_sfm import GIB, PolicyLoss, load_policy, read_config


class DiagnosticAdamW(torch.optim.AdamW):
    """Count actual AdamW calls on independent, disposable weight copies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_steps = 0

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self.completed_steps += 1
        return result


def worker(rank, world_size, batch_size, warmup, measured_steps, rendezvous, report_dir):
    torch.set_num_threads(4)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size)
    iterator = None
    try:
        config = read_config()
        config.batch_size = batch_size
        datamodule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup("fit")
        dataset = datamodule.train_datasets["lang"]
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=config.seed, drop_last=False)
        steps_per_epoch = math.ceil(len(sampler) / batch_size)
        if warmup + measured_steps > steps_per_epoch:
            raise ValueError("Diagnostic steps exceed one dataset pass")
        # Keep normal prefetch active throughout measurement, then drain a small
        # finite tail so workers exit with no pending tensor queue writes.
        tail_steps = 2 * dataset.num_workers + 1
        probe_indices = list(sampler)[:(warmup + measured_steps + tail_steps) * batch_size]
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=probe_indices,
            num_workers=dataset.num_workers, pin_memory=True,
            persistent_workers=dataset.num_workers > 0,
            prefetch_factor=2 if dataset.num_workers > 0 else None,
        )
        iterator = iter(loader)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(load_policy(config)).to(f"cuda:{rank}").train()
        distributed = DistributedDataParallel(PolicyLoss(model), device_ids=[rank], find_unused_parameters=True)
        parameter_versions = {name: value._version for name, value in model.named_parameters()}
        parameter_pairs, copied_groups = [], []
        for group in model._get_param_groups():
            copies = []
            for parameter in group["params"]:
                copied = nn.Parameter(parameter.detach().clone(), requires_grad=False)
                assert parameter.data_ptr() != copied.data_ptr()
                copies.append(copied)
                parameter_pairs.append((parameter, copied))
            copied_groups.append({**group, "params": copies})
        diagnostic_optimizer = DiagnosticAdamW(
            copied_groups, lr=config.model.optimizer.learning_rate,
            betas=tuple(config.model.optimizer.betas),
        )
        for _, parameter in parameter_pairs:
            diagnostic_optimizer.state[parameter] = {
                "step": torch.tensor(0.0), "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        ema = hydra.utils.instantiate(config.callbacks.ema)
        ema.log = lambda *args, **kwargs: None
        ema.on_train_start(None, model)
        precision = config.trainer.precision
        if precision not in ("16-mixed", "bf16-mixed"):
            raise ValueError(f"Unsupported diagnostic precision: {precision}")
        amp_dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0, enabled=precision == "16-mixed")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        records = []
        for iteration in range(warmup + measured_steps):
            if iteration == warmup:
                dist.barrier(device_ids=[rank])
                torch.cuda.synchronize()
                measured_started = time.perf_counter()
            started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            diagnostic_optimizer.zero_grad(set_to_none=True)
            load_started = time.perf_counter()
            batch = next(iterator)
            load_seconds = time.perf_counter() - load_started
            events[0].record()
            batch = move_data_to_device(batch, model.device)
            events[1].record()
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = distributed(batch)
            scaler.scale(loss).backward()
            for parameter, copied in parameter_pairs:
                copied.grad = parameter.grad
            scaler.unscale_(diagnostic_optimizer)
            events[2].record()
            scaler.step(diagnostic_optimizer)
            scaler.update()
            events[3].record()
            ema._cur_step = iteration + 1
            ema.ema(model)
            events[4].record()
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise ValueError("Non-finite loss in throughput diagnostic")
            if iteration >= warmup:
                records.append({
                    "step_wall_seconds": wall_seconds, "loader_wait_seconds": load_seconds,
                    "h2d_gpu_seconds": events[0].elapsed_time(events[1]) / 1000,
                    "forward_backward_gpu_seconds": events[1].elapsed_time(events[2]) / 1000,
                    "adamw_gpu_seconds": events[2].elapsed_time(events[3]) / 1000,
                    "ema_gpu_seconds": events[3].elapsed_time(events[4]) / 1000,
                    "loss": loss_value,
                })
            if rank == 0 and (iteration + 1 == warmup or (iteration + 1 - warmup) % 10 == 0):
                print(json.dumps({"per_gpu_batch": batch_size,
                                  "warmup_completed": min(iteration + 1, warmup),
                                  "measured_completed": max(0, iteration + 1 - warmup),
                                  "step_seconds": wall_seconds}), flush=True)
            del loss, batch
        measured_elapsed = time.perf_counter() - measured_started
        for remaining_batch in iterator:
            del remaining_batch
        assert parameter_versions == {name: value._version for name, value in model.named_parameters()}
        assert diagnostic_optimizer.completed_steps == warmup + measured_steps, "AMP skipped a diagnostic optimizer call"
        report = {
            "rank": rank, "device": torch.cuda.get_device_name(rank),
            "per_gpu_batch_size": batch_size, "global_batch_size": batch_size * world_size,
            "precision": precision, "image_shape": [224, 224], "views": 2,
            "action_horizon": config.act_seq_len, "training_windows": len(dataset),
            "steps_per_epoch": steps_per_epoch, "epochs": config.max_epochs,
            "num_workers_per_gpu": dataset.num_workers, "prefetch_factor": 2,
            "warmup_steps": warmup, "measured_steps": measured_steps,
            "measured_elapsed_seconds": measured_elapsed,
            "mean_step_seconds": measured_elapsed / measured_steps,
            "median_step_seconds": statistics.median(row["step_wall_seconds"] for row in records),
            "p90_step_seconds": float(np.percentile([row["step_wall_seconds"] for row in records], 90)),
            "mean_components_seconds": {key: statistics.mean(row[key] for row in records)
                                        for key in records[0] if key != "loss"},
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            "policy_parameter_versions_unchanged": True, "policy_optimizer_steps": 0,
            "diagnostic_optimizer_steps_on_copies": diagnostic_optimizer.completed_steps,
            "checkpoint_saved": False, "records": records,
        }
        (Path(report_dir) / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
        dist.barrier(device_ids=[rank])
    finally:
        try:
            if iterator is not None:
                iterator._shutdown_workers()
        finally:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--report", type=Path, default=REPO / ".benchmarks/default_sfm_batch_time.json")
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.steps < 2:
        parser.error("Positive batch sizes/warmup and at least two measured steps are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    world_size = len(args.gpus.split(","))
    assert torch.cuda.device_count() == world_size
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "gpus": args.gpus,
        "policy_optimizer_steps": 0,
        "includes": ["data_loader", "shared_color_jitter", "host_to_device", "full_vlm_and_dit",
                     "backward", "DDP", "AdamW_on_independent_weight_copies", "EMA"],
        "excludes": ["checkpoint_writes", "external_logging", "Lightning_loop_overhead"],
        "cases": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        with tempfile.TemporaryDirectory(prefix="flower-time-probe-") as directory:
            mp.spawn(worker, args=(world_size, batch_size, args.warmup, args.steps,
                                  f"file://{directory}/rendezvous", directory), nprocs=world_size, join=True)
            ranks = [json.loads((Path(directory) / f"rank{rank}.json").read_text()) for rank in range(world_size)]
        seconds_per_step = max(rank["measured_elapsed_seconds"] for rank in ranks) / args.steps
        epoch_seconds = seconds_per_step * ranks[0]["steps_per_epoch"]
        result = {
            "per_gpu_batch_size": batch_size, "global_batch_size": batch_size * world_size,
            "seconds_per_step": seconds_per_step,
            "samples_per_second": batch_size * world_size / seconds_per_step,
            "steps_per_epoch": ranks[0]["steps_per_epoch"],
            "estimated_seconds_per_epoch": epoch_seconds,
            "epochs": ranks[0]["epochs"],
            "estimated_total_seconds": epoch_seconds * ranks[0]["epochs"],
            "ranks": ranks,
        }
        summary["cases"].append(result)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "ranks"}), flush=True)
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
