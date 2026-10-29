#!/usr/bin/env python3
"""Receive completed Kingston task archives, unpack, and verify FLOWER loading."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/media/csilab/robot_data/preprocessed/flower_default_sfm_crop_v2_20261028"
TASKS = ("audio_jack", "barlatch", "bnc", "hdmi", "knob", "lan", "outlet", "rotary_switch", "usb")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "Kingston"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=REPO / "datasets/default_sfm_crop_v2_archives")
    parser.add_argument("--output", type=Path, default=REPO / "datasets/default_sfm_crop_v2")
    parser.add_argument("--detach", action="store_true", help="Continue without an active assistant or terminal")
    args = parser.parse_args()
    if args.detach:
        log_dir = REPO / "datasets/default_sfm_crop_v2_transfer"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "transfer.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--archives", str(args.archives.resolve()), "--output", str(args.output.resolve())],
                cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        (log_dir / "transfer.pid").write_text(str(process.pid) + "\n")
        print(json.dumps(dict(detached=True, pid=process.pid, log=str(log_dir / "transfer.log"))))
        return
    args.archives.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Dataset already exists: {args.output}")
    received = set()
    failures = 0
    query = ("python3 -c \"import json; from pathlib import Path; "
             f"p=Path('{REMOTE}'); "
             "print(json.dumps(dict(status=json.loads((p/'status.json').read_text()), "
             "ready=[f.stem for f in (p/'archives').glob('*.json')])))\"")

    def receive(name):
        subprocess.run([
            "rsync", "-a", "--partial", "--info=stats2", "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10",
            f"Kingston:{REMOTE}/archives/{name}", str(args.archives) + "/",
        ], check=True)

    while True:
        result = subprocess.run(SSH + [query], capture_output=True, text=True, timeout=45)
        if result.returncode:
            failures += 1
            if failures >= 3:
                raise RuntimeError(result.stderr)
            time.sleep(15)
            continue
        failures = 0
        remote = json.loads(result.stdout)
        print(json.dumps(remote["status"]), flush=True)
        if remote["status"]["phase"] == "failed":
            raise RuntimeError(remote["status"])
        for task in TASKS:
            if task in remote["ready"] and task not in received:
                receive(task + ".tar")
                receive(task + ".json")
                received.add(task)
                print(json.dumps(dict(phase="received_task", task=task, received=len(received))), flush=True)
        if "packages" in remote["ready"]:
            if len(received) != len(TASKS):
                raise ValueError("Incomplete production task set")
            receive("metadata.tar")
            receive("packages.json")
            break
        time.sleep(30)
    subprocess.run([sys.executable, str(REPO / "scripts/unpack_default_sfm.py"),
                    "--archives", str(args.archives), "--output", str(args.output)], check=True)
    subprocess.run([sys.executable, str(REPO / "scripts/verify_default_sfm.py"),
                    "--dataset", str(args.output)], check=True)
    print(json.dumps(dict(phase="complete", dataset=str(args.output), training_started=False)), flush=True)


if __name__ == "__main__":
    main()
