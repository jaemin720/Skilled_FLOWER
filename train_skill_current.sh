#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec /home/jack/miniforge3/envs/flower/bin/python flower/training_calvin.py \
  --config-name config_default_sfm_skill_current "$@"
