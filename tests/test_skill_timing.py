"""Timing must exclude startup/warmup and include actual optimizer updates."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("gpu_test", Path(__file__).parents[1] / "scripts/test_skill_ablation_gpus.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def write_progress(tmp_path, skipped=False):
    records = []
    for step, elapsed, seconds in [(0, 0, "00:00:10"), (20, 200, "00:03:30"),
                                    (40, 224, "00:03:54"), (60, 252, "00:04:22")]:
        records.append(dict(state="running", global_step=step, elapsed_seconds=elapsed,
                            updated_at=f"2026-10-31T{seconds}+00:00", steps_per_epoch=1041,
                            gru_weight_version_change=2*step-(2 if skipped and step==60 else 0), amp_scale=1024))
    records.append(dict(records[-1], state="completed"))
    target = tmp_path / "seed_42/training_progress.jsonl"
    target.parent.mkdir()
    target.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_timing_excludes_loading_and_warmup(tmp_path):
    write_progress(tmp_path)
    result = probe.training_timing(tmp_path, 20, 40, "2026-10-31T00:00:00+00:00", "2026-10-31T00:04:35+00:00")
    assert result["seconds_per_step"] == pytest.approx(1.3)
    assert result["estimated_30_epoch_seconds"] == pytest.approx(1.3 * 1041 * 30)
    assert result["startup_seconds"] == 10
    assert result["shutdown_seconds"] == 13
    assert result["gru_weight_version_change"] == 80


def test_timing_rejects_skipped_optimizer_updates(tmp_path):
    write_progress(tmp_path, skipped=True)
    with pytest.raises(RuntimeError, match="AMP skips"):
        probe.training_timing(tmp_path, 20, 40, "2026-10-31T00:00:00+00:00", "2026-10-31T00:04:35+00:00")
