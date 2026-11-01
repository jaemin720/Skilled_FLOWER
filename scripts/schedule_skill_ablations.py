#!/usr/bin/env python3
"""Persistent, serial GPU 0/1 queue for the six independent SFM ablations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/jack/miniforge3/envs/flower/bin/python"
ORDER = ("13", "123", "12", "3", "23", "2")
NAMES = ("faflow_history", "faflow_current_history", "faflow_current",
         "gru_history", "gru_current_history", "gru_current")
CHECKPOINTS = ("epoch009.ckpt", "epoch019.ckpt", "epoch029.ckpt")
RANK_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def process_matches(pid, run_dir):
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return any(str(run_dir).encode() in arg for arg in argv)
    except FileNotFoundError:
        return False


def dependency_state(dependency):
    run_dir = Path(dependency["run_dir"])
    receipt = run_dir / "exit_code"
    if not receipt.exists():
        if not process_matches(dependency["pid"], run_dir):
            raise RuntimeError("Previous training disappeared without an exit receipt; queue stopped")
        return "waiting_for_current_training"
    if int(receipt.read_text().strip()) != 0:
        raise RuntimeError("Previous training failed; queue stopped")
    status = read_json(run_dir / "seed_42/training_status.json")
    if (status["state"] != "completed" or status["epoch"] != 30
            or status["global_step"] != dependency["total_steps"]):
        raise RuntimeError("Previous training exited without completing all 30 epochs")
    final = Path(dependency["checkpoint_dir"]) / "epoch029.ckpt"
    if not final.is_file() or final.stat().st_size == 0:
        raise RuntimeError("Previous training's final checkpoint is missing")
    return "completed"


def gpu_occupants():
    # CUDA_VISIBLE_DEVICES cannot restrict nvidia-smi. Filter only physical 0/1;
    # processes on the other GPUs must never block or be touched by this queue.
    mapping = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    selected = {line.split(",")[1].strip() for line in mapping.splitlines()
                if line.split(",")[0].strip() in ("0", "1")}
    if len(selected) != 2:
        raise RuntimeError("Physical GPUs 0 and 1 are unavailable")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in apps.splitlines() if line.split(",")[0].strip() in selected]


def training_env(snapshot):
    env = dict(os.environ)
    for key in RANK_ENV:
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               MPLCONFIGDIR="/tmp/flower-mpl", HF_MODULES_CACHE="/tmp/flower-hf-modules",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline",
               WANDB_CONSOLE="off", PYTHONUNBUFFERED="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
               PYTHONPATH=str(snapshot))
    return env


def command(plan, job, smoke=False):
    run = Path(job["run_dir"]) / "preflight" if smoke else Path(job["run_dir"])
    ckpt = Path(job["checkpoint_dir"]) / "preflight" if smoke else Path(job["checkpoint_dir"])
    result = [plan["python"], str(Path(plan["snapshot"]) / "flower/training_calvin.py"),
              "--config-name", job["config"], f"hydra.run.dir={run}", f"logger.save_dir={run}",
              f"callbacks.checkpoint.dirpath={ckpt}"]
    if smoke:
        result += [f"+trainer.max_steps={plan['smoke_steps']}", "callbacks.checkpoint.every_n_epochs=100000"]
    return result


def validate_config(cfg, components):
    expected_source = "both" if "2" in components and "3" in components else "current" if "2" in components else "history"
    if (cfg.model.skill_source != expected_source or cfg.max_epochs != 30 or cfg.devices != 2
            or cfg.batch_size != 64 or cfg.proprio_dims != 11 or cfg.act_seq_len != 10
            or cfg.model.lowdim_obs_dim != 11 or not cfg.model.use_force_history_adaln
            or cfg.datamodule.datasets.lang_dataset.proprio_source != "physical"
            or list(map(list, cfg.datamodule.proprioception_dims.keep_indices)) != [[0, 1], [3, 13]]):
        raise ValueError(f"Incorrect common baseline for {components}")
    if (cfg.model.force_history_encoder_type != ("fafmp_gru" if "1" in components else "gru")
            or cfg.model.adaln_conditioning_type != ("concat_mlp" if "1" in components else "sum")
            or cfg.callbacks.checkpoint.every_n_epochs != 10 or cfg.callbacks.checkpoint.save_last
            or cfg.callbacks.checkpoint.save_top_k != -1 or not cfg.model.load_pretrained
            or cfg.model.lr_scheduler.lr_scheduler.total_steps != 31230 or cfg.get("resume_from_checkpoint")):
        raise ValueError(f"Incorrect conditioning/training schedule for {components}")
    ds = cfg.datamodule.datasets.lang_dataset
    if ds.load_skill_history != ("3" in components) or ds.load_quest_observations != ("2" in components):
        raise ValueError(f"Incorrect Skill inputs for {components}")


def prepare(args):
    import hydra
    from omegaconf import OmegaConf
    parent = read_json(args.after_run)
    queue = ROOT / "logs/skill_ablation_queue" / args.queue_id
    queue.mkdir(parents=True, exist_ok=False)
    snapshot = queue / "source"
    snapshot.mkdir()
    for directory in ("flower", "conf"):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Freeze the controller as well as the experiment source.
    shutil.copy2(Path(__file__), queue / "controller.py")
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    write_json(queue / "source_manifest.json", hashes)
    jobs = []
    with hydra.initialize_config_dir(config_dir=str(snapshot / "conf")):
        for index, (components, name) in enumerate(zip(ORDER, NAMES), 1):
            config = f"config_sfm_ablation_{components}"
            cfg = hydra.compose(config_name=config)
            validate_config(cfg, components)
            job_name = f"{index:02d}_{components}_{name}"
            run = queue / "runs" / job_name
            output = Path(cfg.checkpoint_root) / "skill_ablation_queue" / args.queue_id / job_name / "seed_42/saved_models"
            jobs.append(dict(name=job_name, components=[int(i) for i in components], config=config,
                             run_dir=str(run), checkpoint_dir=str(output)))
            (queue / f"{job_name}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    plan = dict(schema=1, queue_id=args.queue_id, created_at=now(), python=PYTHON,
                root=str(ROOT), snapshot=str(snapshot), controller_sha256=digest(queue / "controller.py"),
                physical_gpus=[0, 1], poll_seconds=15, smoke_steps=2,
                dependency=dict(run_dir=parent["run_dir"], checkpoint_dir=parent["checkpoint_dir"],
                                pid=args.wait_pid, total_steps=parent["total_steps"]),
                initialization="Each experiment starts from the same base FLOWER checkpoint; no chained resume",
                absent_component_1="Original GRU F/T retained", jobs=jobs)
    write_json(queue / "plan.json", plan)
    write_json(queue / "status.json", dict(state="prepared", updated_at=now(),
                                          jobs=[dict(name=j["name"], state="pending") for j in jobs]))
    print(queue / "plan.json")


def verify_snapshot(plan, queue):
    for relative, expected in read_json(queue / "source_manifest.json").items():
        if digest(Path(plan["snapshot"]) / relative) != expected:
            raise RuntimeError(f"Scheduled source changed: {relative}")


def run(plan_path):
    queue = Path(plan_path).resolve().parent
    plan = read_json(plan_path)
    # One active queue across all manifests; no accidental duplicate launches.
    lock = (Path(plan["root"]) / "logs/skill_ablation_queue/active.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read_json(queue / "status.json")
    if state["state"] != "prepared":
        raise RuntimeError("Only a prepared queue can start; existing state requires inspection")

    def update(phase, **extra):
        state.update(state=phase, updated_at=now(), controller_pid=os.getpid(), **extra)
        write_json(queue / "status.json", state)

    def wait_for_gpu():
        while True:
            occupied = gpu_occupants()
            if not occupied:
                return
            update("waiting_for_gpu_0_1", gpu_occupants=occupied)
            time.sleep(plan["poll_seconds"])

    def launch(job, smoke):
        run_dir = Path(job["run_dir"]) / "preflight" if smoke else Path(job["run_dir"])
        status_path = run_dir / "seed_42/training_status.json"
        if status_path.exists():
            raise RuntimeError(f"Refusing to reuse an already-started experiment: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        env = training_env(plan["snapshot"])
        argv = command(plan, job, smoke)
        write_json(run_dir / "launch.json", dict(command=argv, created_at=now(),
                   physical_gpus=[0, 1], source_manifest=str(queue / "source_manifest.json"),
                   checkpoint_dir=job["checkpoint_dir"], smoke=smoke))
        with (run_dir / "console.log").open("a") as log:
            child = subprocess.Popen(argv, cwd=plan["snapshot"], env=env, stdout=log, stderr=subprocess.STDOUT)
            update("preflight" if smoke else "training", active_run_dir=str(run_dir), training_pid=child.pid)
            while child.poll() is None:
                if status_path.is_file():
                    update("preflight" if smoke else "training", progress=read_json(status_path))
                time.sleep(plan["poll_seconds"])
            (run_dir / "exit_code").write_text(str(child.returncode) + "\n")
            if child.returncode != 0:
                raise RuntimeError(f"{job['name']} {'preflight' if smoke else 'training'} failed with {child.returncode}; see {run_dir}/console.log")
        progress = read_json(status_path)
        expected_steps = plan["smoke_steps"] if smoke else 31230
        if progress["state"] != "completed" or progress["global_step"] != expected_steps:
            raise RuntimeError(f"Incomplete training: {run_dir}")
        if not smoke:
            for filename in CHECKPOINTS:
                file = Path(job["checkpoint_dir"]) / filename
                if not file.is_file() or file.stat().st_size == 0:
                    raise RuntimeError(f"Expected checkpoint missing: {file}")

    try:
        verify_snapshot(plan, queue)
        while dependency_state(plan["dependency"]) != "completed":
            update("waiting_for_current_training", dependency=plan["dependency"])
            time.sleep(plan["poll_seconds"])
        for index, job in enumerate(plan["jobs"]):
            update("waiting_for_gpu_0_1", active_job=job["name"], active_index=index)
            wait_for_gpu()
            verify_snapshot(plan, queue)
            output = Path(job["checkpoint_dir"])
            output.mkdir(parents=True, exist_ok=True)
            if list(output.glob("*.ckpt")):
                raise RuntimeError(f"Refusing to reuse checkpoints: {output}")
            if shutil.disk_usage(output).free < 50 * 1024**3:
                raise RuntimeError("Less than 50 GiB remains for this experiment's three checkpoints")
            shutil.copy2(queue / "plan.json", output / "experiment_queue.json")
            shutil.copy2(queue / "source_manifest.json", output / "source_manifest.json")
            state["jobs"][index].update(state="preflight", started_at=now())
            launch(job, smoke=True)
            wait_for_gpu()
            state["jobs"][index]["state"] = "training"
            launch(job, smoke=False)
            state["jobs"][index].update(state="completed", completed_at=now(),
                                       checkpoint_dir=job["checkpoint_dir"])
            update("experiment_completed")
        update("completed", completed_at=now())
    except BaseException as error:
        if "active_index" in state:
            state["jobs"][state["active_index"]]["state"] = "failed"
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--after-run", required=True, type=Path)
    prep.add_argument("--wait-pid", required=True, type=int)
    prep.add_argument("--queue-id", required=True)
    for name in ("start", "run", "status"):
        sub.add_parser(name).add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args.plan)
    elif args.action == "status":
        print(json.dumps(read_json(args.plan.resolve().parent / "status.json"), indent=2, ensure_ascii=False))
    else:
        plan = read_json(args.plan)
        queue = args.plan.resolve().parent
        controller = queue / "controller.py"
        if digest(controller) != plan["controller_sha256"]:
            raise RuntimeError("Frozen controller source changed")
        with (queue / "controller.log").open("a") as log:
            child = subprocess.Popen([plan["python"], str(controller), "run", str(args.plan.resolve())],
                cwd=plan["root"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=training_env(plan["snapshot"]))
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError(f"Queue did not stay running; see {queue}/controller.log")
        print(json.dumps(dict(controller_pid=child.pid, plan=str(args.plan.resolve()),
                              status=read_json(queue / "status.json")), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
