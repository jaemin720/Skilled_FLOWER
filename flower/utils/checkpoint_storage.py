"""Keep SFM checkpoints and their evaluation config outside the log filesystem."""

import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def prepare_checkpoint_storage(cfg, run_dir: Path, *, write_metadata: bool = True):
    """Resolve the output before ModelCheckpoint is constructed on every rank.

    SFM configs opt in with checkpoint_root and dirpath=null. An explicit
    dirpath remains an override. Only rank zero writes metadata/compatibility
    links; all ranks compute the same path from the actual Hydra run directory.
    """
    root = cfg.get("checkpoint_root")
    if root is None:
        return None

    run_dir = Path(run_dir).resolve()
    explicit = cfg.callbacks.checkpoint.get("dirpath")
    if explicit is not None:
        checkpoint_dir = Path(str(explicit)).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = run_dir / checkpoint_dir
        checkpoint_dir = checkpoint_dir.resolve()
    else:
        root = Path(str(root)).expanduser().resolve()
        log_root = Path(str(cfg.log_dir)).expanduser().resolve()
        try:
            run_path = run_dir.relative_to(log_root)
        except ValueError:
            # External Hydra run directories can share a basename. Keep their
            # storage separate without placing an absolute path under root.
            suffix = hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
            run_path = Path(f"{run_dir.name}-{suffix}")
        checkpoint_dir = root / log_root.name / run_path / f"seed_{cfg.seed}" / "saved_models"

    cfg.callbacks.checkpoint.dirpath = str(checkpoint_dir)
    if not write_metadata:
        return checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A config beside the checkpoint also works for custom output layouts;
    # the RPC loader searches every parent directory for .hydra/config.yaml.
    metadata_dir = checkpoint_dir / ".hydra"
    metadata_dir.mkdir(exist_ok=True)
    _atomic_write(metadata_dir / "config.yaml", OmegaConf.to_yaml(cfg, resolve=True))
    for name in ("hydra.yaml", "overrides.yaml"):
        source = run_dir / ".hydra" / name
        if source.is_file():
            _atomic_write(metadata_dir / name, source.read_text())

    # Preserve the conventional log path used by replay/resume commands, while
    # leaving pre-existing directories and symlinks completely intact.
    local_path = run_dir / f"seed_{cfg.seed}" / "saved_models"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path != checkpoint_dir and not local_path.exists() and not local_path.is_symlink():
        try:
            local_path.symlink_to(checkpoint_dir, target_is_directory=True)
        except FileExistsError:
            pass

    record = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "training_config": str(metadata_dir / "config.yaml"),
        "local_checkpoint_path": str(local_path),
        "local_path_points_to_storage": local_path.resolve() == checkpoint_dir,
        "explicit_dirpath": explicit is not None,
    }
    _atomic_write(run_dir / "checkpoint_storage.json", json.dumps(record, indent=2) + "\n")
    return checkpoint_dir
