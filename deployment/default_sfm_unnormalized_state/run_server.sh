#!/usr/bin/env bash
# Physical proprioception without X/Y, epoch 40, EMA weights.
set -euo pipefail
FLOWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FLOWER_RUN="$FLOWER_ROOT/logs/default_sfm_unnormalized_state/runs/flower_sfm_no_xy_unnorm_e40_b64_20261029_200954"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec bash "$FLOWER_ROOT/deployment/default_sfm/run_server.sh" \
  --checkpoint "$FLOWER_RUN/seed_42/saved_models/epoch029.ckpt" \
  --training-config "$FLOWER_RUN/.hydra/config.yaml" \
  --preprocess-bundle "$FLOWER_ROOT/datasets/default_sfm_crop_v2/rpc_bundle" \
  --weights ema \
  --port "${FLOWER_SERVER_PORT:-45589}" \
  --audit-file "$FLOWER_ROOT/logs/default_sfm_unnormalized_state_rpc/model_audit.json" \
  "$@"
