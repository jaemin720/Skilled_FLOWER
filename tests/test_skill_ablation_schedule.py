"""Exercise scheduling decisions without launching training or needing a GPU."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import hydra
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("skill_queue", ROOT / "scripts/schedule_skill_ablations.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize("components", queue.ORDER)
def test_six_configs_use_physical_no_xy_and_independent_pretrained_initialization(components):
    with hydra.initialize_config_dir(config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name=f"config_sfm_ablation_{components}")
    queue.validate_config(cfg, components)
    assert cfg.checkpoint_root == "/data/jack/Skilled_FLOWER/checkpoints"
    assert cfg.root_data_dir == str(ROOT / "datasets/default_sfm_crop_v2")
    assert cfg.model.pretrained_model_path == str(ROOT / "flower_vla_pret/360000_model_weights.pt")
    assert cfg.model.force_history_length == 50
    assert cfg.experiment_components == [int(c) for c in components]


def test_requested_order_and_no_inherited_rank_or_gpu_selection(monkeypatch):
    assert queue.ORDER == ("13", "123", "12", "3", "23", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    env = queue.training_env(Path("/snapshot"))
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert all(key not in env for key in queue.RANK_ENV)


def dependency(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (tmp_path / "seed_42").mkdir()
    return dict(pid=123, run_dir=str(tmp_path), checkpoint_dir=str(checkpoint), total_steps=31230)


def test_waits_for_exit_receipt_and_does_not_continue_on_dead_process(tmp_path):
    parent = dependency(tmp_path)
    with patch.object(queue, "process_matches", return_value=True):
        assert queue.dependency_state(parent) == "waiting_for_current_training"
    with patch.object(queue, "process_matches", return_value=False), pytest.raises(RuntimeError, match="disappeared"):
        queue.dependency_state(parent)


@pytest.mark.parametrize("exit_code,state,steps,final,ok", [
    (0, "completed", 31230, True, True), (1, "completed", 31230, True, False),
    (0, "failed", 31230, True, False), (0, "completed", 20000, True, False),
    (0, "completed", 31230, False, False),
])
def test_dependency_must_finish_all_epochs_and_save_final_checkpoint(tmp_path, exit_code, state, steps, final, ok):
    parent = dependency(tmp_path)
    (tmp_path / "exit_code").write_text(str(exit_code))
    (tmp_path / "seed_42/training_status.json").write_text(json.dumps(dict(state=state, global_step=steps, epoch=30)))
    if final:
        (tmp_path / "checkpoints/epoch029.ckpt").write_bytes(b"saved")
    if ok:
        assert queue.dependency_state(parent) == "completed"
    else:
        with pytest.raises(RuntimeError):
            queue.dependency_state(parent)


def test_only_gpu_zero_one_occupants_block_queue():
    with patch.object(queue.subprocess, "check_output", side_effect=[
        "0, GPU-a\n1, GPU-b\n4, GPU-c\n", "GPU-c, 100\nGPU-b, 200\n"]):
        assert queue.gpu_occupants() == ["GPU-b, 200"]


def test_preflight_and_training_have_distinct_paths_and_no_resume():
    plan = dict(python="python", snapshot="/source", smoke_steps=2)
    job = dict(config="config_sfm_ablation_123", run_dir="/runs/02", checkpoint_dir="/data/02/saved_models")
    full = queue.command(plan, job)
    smoke = queue.command(plan, job, smoke=True)
    assert not any("resume" in arg for arg in full + smoke)
    assert "hydra.run.dir=/runs/02" in full
    assert "hydra.run.dir=/runs/02/preflight" in smoke
    assert "+trainer.max_steps=2" in smoke
    assert "callbacks.checkpoint.dirpath=/data/02/saved_models" in full


@pytest.mark.parametrize("fail_job", [None, "123"])
def test_controller_runs_in_order_and_stops_on_failure(tmp_path, fail_job):
    folder = tmp_path / "logs/skill_ablation_queue/test"
    folder.mkdir(parents=True)
    source = folder / "source"
    source.mkdir()
    jobs = [dict(name=component, config=f"config_sfm_ablation_{component}",
                 run_dir=str(folder / "runs" / component), checkpoint_dir=str(tmp_path / "checkpoints" / component))
            for component in queue.ORDER]
    plan = dict(root=str(tmp_path), snapshot=str(source), python="python", jobs=jobs,
                dependency={}, smoke_steps=2, poll_seconds=0)
    queue.write_json(folder / "plan.json", plan)
    queue.write_json(folder / "source_manifest.json", {})
    queue.write_json(folder / "status.json", dict(state="prepared", jobs=[dict(name=j["name"], state="pending") for j in jobs]))
    calls = []

    class FakeTraining:
        pid = 999

        def __init__(self, argv, **kwargs):
            run_dir = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("hydra.run.dir=")))
            checkpoint_dir = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("callbacks.checkpoint.dirpath=")))
            component = argv[argv.index("--config-name") + 1].rsplit("_", 1)[-1]
            smoke = any(a.startswith("+trainer.max_steps=") for a in argv)
            calls.append((component, smoke))
            self.returncode = 1 if component == fail_job and not smoke else 0
            queue.write_json(run_dir / "seed_42/training_status.json", dict(
                state="completed", epoch=30, global_step=2 if smoke else 31230))
            if not smoke:
                for filename in queue.CHECKPOINTS:
                    (checkpoint_dir / filename).write_bytes(b"checkpoint")

        def poll(self):
            return self.returncode

    with patch.object(queue, "dependency_state", return_value="completed"), \
         patch.object(queue, "gpu_occupants", return_value=[]), \
         patch.object(queue.subprocess, "Popen", FakeTraining), \
         patch.object(queue.shutil, "disk_usage", return_value=type("Disk", (), dict(free=10**12))()):
        if fail_job:
            with pytest.raises(RuntimeError, match="training failed"):
                queue.run(folder / "plan.json")
        else:
            queue.run(folder / "plan.json")
    expected_order = queue.ORDER[:2] if fail_job else queue.ORDER
    assert calls == [(component, smoke) for component in expected_order for smoke in (True, False)]
    status = queue.read_json(folder / "status.json")
    assert status["state"] == ("failed" if fail_job else "completed")
    if fail_job:
        assert all(job["state"] == "pending" for job in status["jobs"][2:])
