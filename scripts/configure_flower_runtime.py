"""Configure the legacy simulator packages without changing their pinned versions.

Run with the flower environment's Python after installing requirements.flower.txt:
    python scripts/configure_flower_runtime.py /path/to/mujoco210-linux-x86_64.tar.gz

Official MuJoCo archive: https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
All installed files stay inside this Conda environment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import tarfile
import textwrap


COMPATIBILITY_MODULE = '''"""Legacy NetworkX/URDF/TACTO compatibility for the flower environment."""
import fractions
import math
import numpy as np

# NetworkX 2.2 imports the function removed from fractions in Python 3.9.
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

# These deprecated NumPy aliases were identical to Python's built-in types.
# NetworkX 2.2, urdfpy 0.0.22, and TACTO 0.0.3 still use them.
np.__dict__.setdefault("int", int)
np.__dict__.setdefault("float", float)
'''

ACTIVATE = r'''
_FLOWER_SAVED_MUJOCO_PATH_SET="${MUJOCO_PY_MUJOCO_PATH+x}"
_FLOWER_SAVED_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH-}"
_FLOWER_SAVED_LD_LIBRARY_PATH_SET="${LD_LIBRARY_PATH+x}"
_FLOWER_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export MUJOCO_PY_MUJOCO_PATH="${CONDA_PREFIX}/share/mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
'''

DEACTIVATE = r'''
if [ "${_FLOWER_SAVED_MUJOCO_PATH_SET-}" = x ]; then
    export MUJOCO_PY_MUJOCO_PATH="${_FLOWER_SAVED_MUJOCO_PATH}"
else
    unset MUJOCO_PY_MUJOCO_PATH
fi
if [ "${_FLOWER_SAVED_LD_LIBRARY_PATH_SET-}" = x ]; then
    export LD_LIBRARY_PATH="${_FLOWER_SAVED_LD_LIBRARY_PATH}"
else
    unset LD_LIBRARY_PATH
fi
unset _FLOWER_SAVED_MUJOCO_PATH_SET _FLOWER_SAVED_MUJOCO_PATH
unset _FLOWER_SAVED_LD_LIBRARY_PATH_SET _FLOWER_SAVED_LD_LIBRARY_PATH
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mujoco_archive", type=Path)
    args = parser.parse_args()
    prefix = Path(sys.prefix)
    if prefix.name != "flower" or not (prefix / "conda-meta").is_dir():
        parser.error("Run this script with the flower Conda environment's Python.")
    if sys.version_info[:2] != (3, 9):
        parser.error("This compatibility configuration is for Python 3.9.")

    native_root = prefix / "share" / "mujoco"
    native_path = native_root / "mujoco210"
    if not (native_path / "bin" / "libmujoco210.so").is_file():
        native_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.mujoco_archive, "r:gz") as archive:
            if any(Path(member.name).parts[0] != "mujoco210" for member in archive.getmembers()):
                raise ValueError("Expected the official mujoco210 archive layout.")
            archive.extractall(native_root, filter="data")
    if not (native_path / "bin" / "libmujoco210.so").is_file():
        raise RuntimeError("MuJoCo 2.1.0 shared library was not extracted.")

    site_packages = Path(sysconfig.get_path("purelib"))
    (site_packages / "flower_legacy_compat.py").write_text(COMPATIBILITY_MODULE)
    (site_packages / "flower_legacy_compat.pth").write_text("import flower_legacy_compat\n")
    for directory, content in [("activate.d", ACTIVATE), ("deactivate.d", DEACTIVATE)]:
        hooks = prefix / "etc" / "conda" / directory
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "flower-mujoco.sh").write_text(textwrap.dedent(content).lstrip())

    record = {
        "mujoco_version": "2.1.0",
        "mujoco_url": "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz",
        "mujoco_sha256": hashlib.sha256(args.mujoco_archive.read_bytes()).hexdigest(),
        "compatibility_module": str(site_packages / "flower_legacy_compat.py"),
        "aliases": ["fractions.gcd = math.gcd", "numpy.int = int", "numpy.float = float"],
    }
    (native_root / "flower-runtime.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"MuJoCo 2.1.0: {native_path}")
    print(f"Legacy package compatibility: {site_packages / 'flower_legacy_compat.pth'}")
    print("Reactivate flower to load the MuJoCo library paths.")


if __name__ == "__main__":
    main()
