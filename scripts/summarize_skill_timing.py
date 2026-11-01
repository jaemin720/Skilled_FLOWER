#!/usr/bin/env python3
"""Estimate the six-job queue from completed GPU throughput measurements."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = read(args.report)
    if report["state"] != "passed" or not report["request"]["timing"] or len(report["results"]) != 6:
        raise ValueError("Six completed timing experiments are required")
    plan_path = Path(report["request"]["production_plan"])
    plan = read(plan_path)
    if read(plan_path.parent / "source_manifest.json") != read(report["request"]["source_manifest"]):
        raise ValueError("Queued source differs from the measured source")
    run = Path(plan["dependency"]["run_dir"])
    status = read(run / "seed_42/training_status.json")
    lines = (run / "seed_42/training_progress.jsonl").read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.endswith("\n")]
    records = [r for r in records if r["state"] == "running"]
    intervals = [(a["global_step"], b["global_step"], b["elapsed_seconds"]-a["elapsed_seconds"])
                 for a, b in zip(records, records[1:]) if b["global_step"] > a["global_step"]]
    baseline_rate = statistics.median(seconds/(end-start) for start, end, seconds in intervals[-50:])
    save_delays = []
    for boundary in range(5*1041, records[-1]["global_step"], 5*1041):
        start, end, seconds = next(interval for interval in intervals if interval[0] <= boundary < interval[1])
        save_delays.append(max(0, seconds - baseline_rate*(end-start)))
    save_delay = statistics.median(save_delays)
    now = datetime.now(timezone.utc)
    kst = ZoneInfo("Asia/Seoul")
    remaining = max(0, plan["dependency"]["total_steps"] - status["global_step"])
    baseline_end = datetime.fromisoformat(status["updated_at"]) + timedelta(seconds=remaining*baseline_rate + (save_delay if remaining else 0))
    cursor = max(now, baseline_end)
    queue_start = cursor
    rows = []
    for job, result in zip(plan["jobs"], report["results"]):
        assert job["name"] == result["name"] and result["state"] == "passed"
        t = result["timing"]
        cold_extra = max(0, t["warmup_seconds"] - t["warmup_steps"]*t["seconds_per_step"])
        loading = t["startup_seconds"] + t["shutdown_seconds"] + cold_extra
        # A separate two-step preflight precedes each fresh 30-epoch model.
        preflight = loading + plan["smoke_steps"]*t["seconds_per_step"] + plan["poll_seconds"]/2
        duration = t["estimated_30_epoch_seconds"] + loading + 3*save_delay + plan["poll_seconds"]/2
        cursor += timedelta(seconds=preflight)
        training_start = cursor
        cursor += timedelta(seconds=duration)
        rows.append(dict(components=",".join(map(str, job["components"])),
                         seconds_per_step=t["seconds_per_step"], epoch_minutes=t["estimated_epoch_seconds"]/60,
                         training_30_epoch_hours=t["estimated_30_epoch_seconds"]/3600,
                         scheduled_hours_including_preflight=(duration+preflight)/3600,
                         estimated_training_start_kst=training_start.astimezone(kst).isoformat(),
                         estimated_training_end_kst=cursor.astimezone(kst).isoformat(),
                         gpu2_peak_mib=result["peak_used_mib"]["2"], gpu3_peak_mib=result["peak_used_mib"]["3"]))
    from_now = max(0, (cursor-now).total_seconds())
    summary = dict(created_at_kst=now.astimezone(kst).isoformat(), report=str(args.report.resolve()),
                   per_gpu_batch=64, global_batch=128, physical_test_gpus=[2,3], production_gpus=plan["physical_gpus"],
                   warmup_steps=report["request"]["warmup_steps"], measured_steps=report["request"]["measured_steps"], rows=rows,
                   baseline_status=status, baseline_recent_seconds_per_step=baseline_rate,
                   baseline_remaining_hours=max(0,(baseline_end-now).total_seconds())/3600,
                   baseline_estimated_end_kst=baseline_end.astimezone(kst).isoformat(),
                   observed_baseline_checkpoint_extra_seconds=save_delays,
                   assumed_seconds_per_future_checkpoint=save_delay,
                   six_models_training_hours=sum(r["training_30_epoch_hours"] for r in rows),
                   six_models_scheduled_hours=(cursor-queue_start).total_seconds()/3600,
                   all_remaining_hours=from_now/3600, estimated_queue_end_kst=cursor.astimezone(kst).isoformat(),
                   planning_margin_fraction=0.10,
                   planning_end_range_kst=[(now+timedelta(seconds=from_now*f)).astimezone(kst).isoformat() for f in (0.9,1.1)],
                   assumptions=["Current GPU 0/1 run completes successfully and all six jobs run serially without interruption.",
                                "GPU 2/3 throughput is representative of the same GPU model on 0/1.",
                                "Checkpoint overhead is approximated by the active baseline; Skill checkpoint writes are not directly timed.",
                                "The 10% planning margin is a practical allowance, not a statistical confidence interval."])
    output = args.report.parent / "timing_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
