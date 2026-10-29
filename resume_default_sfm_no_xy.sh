#!/usr/bin/env bash
# Restore the saved five-epoch state of the XY-removed experiment.
# Training resumes at epoch 6 and ends at the requested total of 40 epochs.
set -euo pipefail
cd /home/jack/Skilled_FLOWER

flower_source_run=/home/jack/Skilled_FLOWER/logs/default_sfm_no_xy/runs/flower_sfm_no_xy_b64_20261028_133635
flower_resume_checkpoint="$flower_source_run/seed_42/saved_models/epoch004.ckpt"
test -f "$flower_resume_checkpoint"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR=/tmp/flower-mpl
export HF_MODULES_CACHE=/tmp/flower-hf-modules
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
# Lightning 2.0 omits weights_only when loading full training checkpoints.
# This run restores our own trusted checkpoint, including OmegaConf metadata.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Hydra inspection flags, e.g. --cfg job --resolve, do not start training.
exec /home/jack/miniforge3/envs/flower/bin/python -u flower/training_calvin.py \
  --config-path "$flower_source_run/.hydra" --config-name config \
  "+resume_from_checkpoint=$flower_resume_checkpoint" \
  'max_epochs=40' \
  'trainer.max_epochs=40' \
  'model.lr_scheduler.lr_scheduler.total_steps=41640' \
  '+model.lr_scheduler.lr_scheduler.extend_on_resume=true' \
  'hydra.run.dir=/home/jack/Skilled_FLOWER/logs/default_sfm_no_xy/resumes/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  'logger.save_dir=${hydra:run.dir}' \
  'callbacks.checkpoint.dirpath=${hydra:run.dir}/seed_42/saved_models' \
  "$@"
