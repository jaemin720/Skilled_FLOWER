#!/usr/bin/env python3
"""Verify and safely unpack the per-task archives produced on Kingston."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    staging = output.with_name(output.name + ".incoming")
    if output.exists() or staging.exists():
        raise FileExistsError("Select an unused output directory; existing datasets are never overwritten")
    package_list = json.loads((args.archives / "packages.json").read_text())
    assert package_list["complete"]
    assert len({row["file"] for row in package_list["packages"]}) == len(package_list["packages"])
    staging.mkdir(parents=True)
    seen = set()
    for row in package_list["packages"]:
        assert Path(row["file"]).name == row["file"]
        path = args.archives / row["file"]
        assert path.stat().st_size == row["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path}")
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                destination = (staging / member.name).resolve()
                if not member.isfile() or not destination.is_relative_to(staging) or destination in seen:
                    raise ValueError(f"Unsafe or repeated archive member: {member.name}")
                seen.add(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("xb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
        print(json.dumps(dict(archive=row["file"], checksum_verified=True, extracted_files=len(seen))), flush=True)
    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["complete"] and manifest["frames"] == package_list["frames"]
    assert len(list((staging / "training").glob("episode_*.npz"))) == manifest["frames"]
    (staging / "transfer_verification.json").write_text(json.dumps(package_list, indent=2) + "\n")
    staging.rename(output)
    print(json.dumps(dict(complete=True, output=str(output), frames=manifest["frames"], episodes=manifest["episodes"])), flush=True)


if __name__ == "__main__":
    main()
