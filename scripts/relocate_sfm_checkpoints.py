#!/usr/bin/env python3
"""Apply an explicit checkpoint relocation plan, verifying bytes before relinking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time


def signature(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def emit(**message):
    print(json.dumps(message), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    destination_root = Path(plan["destination_root"])
    assert destination_root == Path("/data/jack/Skilled_FLOWER/checkpoints")
    for item in plan["checkpoints"]:
        source, destination = Path(item["source"]), Path(item["destination"])
        assert source.is_relative_to(Path(plan["source_root"]) / "logs")
        assert destination.is_relative_to(destination_root)
        assert not source.is_symlink() and signature(source) == item["source_signature"]
        assert not destination.exists() and not destination.is_symlink(), destination
    if not args.apply:
        emit(state="plan_valid",files=len(plan["checkpoints"]),bytes=plan["total_bytes"])
        return
    assert shutil.disk_usage("/data/jack").free > plan["total_bytes"] + 5 * 1024**3
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / "relocation_manifest.json"
    assert not manifest_path.exists(), "A relocation manifest already exists; inspect it before retrying"
    manifest = {**plan, "state": "copying", "verified_checkpoints": [], "metadata": []}
    write_json(manifest_path, manifest)
    for item in plan["metadata"]:
        source, destination = Path(item["source"]), Path(item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists(), destination
        shutil.copy2(source, destination)
        checksum = digest(source)
        assert digest(destination) == checksum
        manifest["metadata"].append({**item, "sha256": checksum})
    write_json(manifest_path, manifest)
    for index, item in enumerate(plan["checkpoints"], 1):
        source, destination = Path(item["source"]), Path(item["destination"])
        assert signature(source) == item["source_signature"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + ".partial")
        emit(state="copying", index=index, total=len(plan["checkpoints"]), source=str(source))
        checksum = hashlib.sha256()
        written, last_report = 0, time.monotonic()
        with source.open("rb") as reader, pending.open("xb") as writer:
            for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                checksum.update(block)
                writer.write(block)
                written += len(block)
                if time.monotonic() - last_report >= 15:
                    emit(state="copying", index=index, copied_gib=round(written / 1024**3, 2))
                    last_report = time.monotonic()
            writer.flush()
            os.fsync(writer.fileno())
        assert written == item["source_signature"]["size"]
        assert signature(source) == item["source_signature"], "Source changed while copying"
        emit(state="verifying", index=index, destination=str(destination))
        expected = checksum.hexdigest()
        assert digest(pending) == expected, "Destination checksum mismatch; original retained"
        shutil.copystat(source, pending)
        pending.replace(destination)
        manifest["verified_checkpoints"].append({**item, "sha256": expected, "linked": False})
        write_json(manifest_path, manifest)
        emit(state="verified", index=index, sha256=expected)
    # Every checkpoint is verified before replacing any source with a link.
    manifest["state"] = "linking"
    write_json(manifest_path, manifest)
    for item in manifest["verified_checkpoints"]:
        source, destination = Path(item["source"]), Path(item["destination"])
        assert not source.is_symlink() and signature(source) == item["source_signature"]
        assert destination.stat().st_size == item["source_signature"]["size"]
        link = source.with_name(source.name + ".relocation-link")
        link.symlink_to(destination)
        os.replace(link, source)
        assert source.is_symlink() and source.resolve() == destination
        item["linked"] = True
        write_json(manifest_path, manifest)
    manifest["state"] = "complete"
    write_json(manifest_path, manifest)
    emit(state="complete", files=len(plan["checkpoints"]), released_gib=plan["total_bytes"] / 1024**3,
         manifest=str(manifest_path))


if __name__ == "__main__":
    main()
