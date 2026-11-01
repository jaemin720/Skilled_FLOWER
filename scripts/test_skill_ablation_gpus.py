#!/usr/bin/env python3
"""One-time batch-64 DDP checks on physical GPUs 2/3, using queued source."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def gpu_state():
    text = subprocess.check_output([
        "nvidia-smi", "-i", "2,3", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in text.splitlines():
        index, uuid, used, total, utilization = (part.strip() for part in line.split(","))
        result.append(dict(index=int(index), uuid=uuid, used_mib=int(used), total_mib=int(total),
                           utilization=int(utilization)))
    if sorted(g["index"] for g in result) != [2, 3]:
        raise RuntimeError("This one-time check requires physical GPUs 2 and 3")
    return result


def occupants(gpus):
    uuids = {g["uuid"] for g in gpus}
    text = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    return [line for line in text.splitlines() if line.split(",")[0].strip() in uuids]


def training_timing(run, warmup_steps, measured_steps, process_started_at, process_ended_at):
    """Time complete Lightning steps after warmup, including AdamW and EMA.

    TrainingStatus materializes CUDA loss before its monotonic timestamp, so
    the interval covers completed GPU work, loader waits and DDP synchronization.
    Keep its production 20-step logging frequency; never extrapolate startup.
    """
    records = [json.loads(line) for line in
               (Path(run) / "seed_42/training_progress.jsonl").read_text().splitlines()]
    by_step = {r["global_step"]: r for r in records if r["state"] == "running"}
    end_step = warmup_steps + measured_steps
    start, end = by_step[warmup_steps], by_step[end_step]
    seconds = end["elapsed_seconds"] - start["elapsed_seconds"]
    versions = end["gru_weight_version_change"] - start["gru_weight_version_change"]
    # The production AdamW performs two in-place updates per parameter. A
    # skipped AMP step would lower this delta and invalidate the speed estimate.
    if (seconds <= 0 or versions != 2 * measured_steps
            or start["amp_scale"] != end["amp_scale"]):
        raise RuntimeError("Timing window includes AMP skips or invalid timing; increase warmup")
    intervals = []
    for step in range(warmup_steps + 20, end_step + 1, 20):
        previous, current = by_step[step - 20], by_step[step]
        intervals.append(dict(start_step=step-20, end_step=step,
                              seconds_per_step=(current["elapsed_seconds"] - previous["elapsed_seconds"]) / 20))
    train_start = by_step[0]
    completed = next(r for r in reversed(records) if r["state"] == "completed")
    startup = (datetime.fromisoformat(train_start["updated_at"])
               - datetime.fromisoformat(process_started_at)).total_seconds()
    shutdown = (datetime.fromisoformat(process_ended_at)
                - datetime.fromisoformat(completed["updated_at"])).total_seconds()
    per_step = seconds / measured_steps
    per_epoch = per_step * end["steps_per_epoch"]
    return dict(warmup_steps=warmup_steps, measured_steps=measured_steps,
                measured_seconds=seconds, seconds_per_step=per_step,
                global_samples_per_second=128 / per_step,
                steps_per_epoch=end["steps_per_epoch"], epochs=30,
                estimated_epoch_seconds=per_epoch, estimated_30_epoch_seconds=30*per_epoch,
                startup_seconds=startup, shutdown_seconds=shutdown,
                warmup_seconds=start["elapsed_seconds"],
                gru_weight_version_change=versions, amp_scale=start["amp_scale"],
                amp_skipped_updates_in_measurement=False, intervals=intervals,
                excludes=["checkpoint serialization", "epoch-boundary overhead"],
                includes=["data loading", "augmentation", "forward/backward", "DDP", "AdamW", "EMA", "Lightning", "production logging frequency"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timing", action="store_true", help="Measure throughput at the production logging frequency")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.timing:
        if (not 20 <= args.warmup_steps <= 60 or not 20 <= args.measured_steps <= 120
                or args.warmup_steps % 20 or args.measured_steps % 20):
            raise ValueError("Timing requires 20..60 warmup and 20..120 measured steps, in multiples of 20")
        args.steps = args.warmup_steps + args.measured_steps
    elif not 2 <= args.steps <= 20:
        raise ValueError("Use 2..20 steps for a temporary diagnostic")
    plan = json.loads(args.plan.read_text())
    queue_dir = args.plan.resolve().parent
    spec = importlib.util.spec_from_file_location("queued_controller", queue_dir / "controller.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.verify_snapshot(plan, queue_dir)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        raise RuntimeError("Use a new output directory; existing test results are preserved")
    shutil.copy2(queue_dir / "source_manifest.json", output / "source_manifest.json")
    # Build an independent environment. The live queue and its plan stay on 0/1.
    env = queue.training_env(plan["snapshot"])
    env["CUDA_VISIBLE_DEVICES"] = "2,3"
    commands = []
    for job in plan["jobs"]:
        run = output / job["name"]
        cmd = [plan["python"], str(Path(plan["snapshot"]) / "flower/training_calvin.py"),
               "--config-name", job["config"], "batch_size=64", "devices=2",
               f"+trainer.max_steps={args.steps}", f"callbacks.status.every_n_steps={20 if args.timing else 1}",
               "callbacks.checkpoint.every_n_epochs=100000", "callbacks.checkpoint.save_last=false",
               f"hydra.run.dir={run}", f"logger.save_dir={run}", f"checkpoint_root={output}",
               f"callbacks.checkpoint.dirpath={run / 'checkpoints'}"]
        commands.append(dict(name=job["name"], config=job["config"], run_dir=str(run), command=cmd))
    request = dict(created_at=stamp(), physical_gpus=[2, 3], per_gpu_batch_size=64, global_batch_size=128,
                   steps=args.steps, production_plan=str(args.plan.resolve()),
                   timing=args.timing, warmup_steps=args.warmup_steps if args.timing else None,
                   measured_steps=args.measured_steps if args.timing else None,
                   production_gpus=plan["physical_gpus"], source_manifest=str(output / "source_manifest.json"),
                   commands=commands)
    queue.write_json(output / "request.json", request)
    if args.prepare_only:
        print(json.dumps(request, indent=2))
        return
    report = dict(request=request, state="running", started_at=stamp(), results=[])
    queue.write_json(report_path, report)
    child = None
    try:
        for job in commands:
            baseline = gpu_state()
            if occupants(baseline):
                raise RuntimeError("GPU 2/3 is occupied; refusing to overlap another job")
            queue.verify_snapshot(plan, queue_dir)
            run = Path(job["run_dir"])
            run.mkdir(parents=True, exist_ok=False)
            result = dict(name=job["name"], config=job["config"], state="running", started_at=stamp(),
                          gpu_baseline=baseline, peak_used_mib={str(g["index"]): g["used_mib"] for g in baseline})
            report["results"].append(result)
            queue.write_json(report_path, report)
            started = time.monotonic()
            with (run / "console.log").open("w") as log:
                child = subprocess.Popen(job["command"], cwd=plan["snapshot"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result["pid"] = child.pid
                while child.poll() is None:
                    for gpu in gpu_state():
                        key = str(gpu["index"])
                        result["peak_used_mib"][key] = max(result["peak_used_mib"][key], gpu["used_mib"])
                    progress = run / "seed_42/training_status.json"
                    if progress.exists():
                        result["progress"] = json.loads(progress.read_text())
                    queue.write_json(report_path, report)
                    if time.monotonic() - started > args.timeout_seconds:
                        raise TimeoutError(f"Diagnostic exceeded {args.timeout_seconds}s: {job['name']}")
                    time.sleep(2)
            result.update(exit_code=child.returncode, elapsed_seconds=time.monotonic() - started, ended_at=stamp())
            (run / "exit_code").write_text(str(child.returncode) + "\n")
            if child.returncode:
                raise RuntimeError(f"{job['name']} failed; see {run}/console.log")
            status = json.loads((run / "seed_42/training_status.json").read_text())
            if (status["state"] != "completed" or status["global_step"] != args.steps
                    or not math.isfinite(status["loss"]) or not status["gru_weight_version_change"]):
                raise RuntimeError(f"Incomplete/invalid training result for {job['name']}: {status}")
            if list(run.rglob("*.ckpt")):
                raise RuntimeError("A diagnostic checkpoint was unexpectedly saved")
            if args.timing:
                result["timing"] = training_timing(run, args.warmup_steps, args.measured_steps,
                                                    result["started_at"], result["ended_at"])
            result.update(state="passed", progress=status, checkpoints_saved=False)
            queue.write_json(report_path, report)
            print(json.dumps(result), flush=True)
            # Ensure both DDP ranks have released these two GPUs before the next model.
            for _ in range(30):
                if not occupants(gpu_state()):
                    break
                time.sleep(2)
            else:
                raise RuntimeError("GPU 2/3 remains occupied after the diagnostic exited")
        report.update(state="passed", ended_at=stamp(), final_gpu_state=gpu_state())
    except BaseException as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", ended_at=stamp())
        if report["results"] and report["results"][-1]["state"] == "running":
            report["results"][-1]["state"] = "failed"
        # Terminate only this script's own process group on failure/timeout.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        queue.write_json(report_path, report)


if __name__ == "__main__":
    main()
