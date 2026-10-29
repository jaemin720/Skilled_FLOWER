"""Write local training progress independently of the experiment logger."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

from pytorch_lightning import Callback
import torch

logger = logging.getLogger(__name__)


class TrainingStatus(Callback):
    def __init__(self, every_n_steps=20):
        self.every_n_steps = int(every_n_steps)
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.started = None
        self.initial_weight_version = None
        self.last_loss = None

    def _write(self, trainer, module, state, **extra):
        if not trainer.is_global_zero:
            return
        device = module.device
        encoder = getattr(module, "force_history_encoder", None)
        gru = getattr(encoder, "gru", None)
        changed = None
        if gru is not None and self.initial_weight_version is not None:
            changed = gru.weight_ih_l0._version - self.initial_weight_version
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        record = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epoch": min(trainer.current_epoch + 1, trainer.max_epochs) if trainer.max_epochs > 0 else trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "steps_per_epoch": trainer.num_training_batches,
            "max_epochs": trainer.max_epochs,
            "loss": self.last_loss,
            "learning_rates": [group["lr"] for optimizer in trainer.optimizers for group in optimizer.param_groups],
            "amp_scale": scaler.get_scale() if scaler is not None else None,
            "gru_weight_version_change": changed,
            "elapsed_seconds": time.monotonic() - self.started if self.started is not None else None,
            "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3 if device.type == "cuda" else 0,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0,
            **extra,
        }
        root = Path(trainer.default_root_dir)
        root.mkdir(parents=True, exist_ok=True)
        pending = root / "training_status.json.tmp"
        pending.write_text(json.dumps(record, indent=2) + "\n")
        pending.replace(root / "training_status.json")
        with (root / "training_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logger.info("Training status: %s", json.dumps(record))

    def on_train_start(self, trainer, pl_module):
        self.started = time.monotonic()
        gru = getattr(getattr(pl_module, "force_history_encoder", None), "gru", None)
        if gru is not None:
            self.initial_weight_version = gru.weight_ih_l0._version
        self._write(trainer, pl_module, "running")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step != 1 and trainer.global_step % self.every_n_steps:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if torch.is_tensor(loss):
            self.last_loss = float(loss.detach())
        self._write(trainer, pl_module, "running", batch_index=batch_idx)

    def on_train_end(self, trainer, pl_module):
        self._write(trainer, pl_module, "completed")

    def on_exception(self, trainer, pl_module, exception):
        self._write(trainer, pl_module, "failed", error=f"{type(exception).__name__}: {exception}")
