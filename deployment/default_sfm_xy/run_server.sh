#!/usr/bin/env bash
# X/Y-included FLOWER, epoch 35, EMA, stored on /data/jack.
set -euo pipefail
FLOWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FLOWER_RUN="/data/jack/Skilled_FLOWER/checkpoints/default_sfm/flower_sfm_xy_b64_e40_resume_20261029_123336"
exec bash "$FLOWER_ROOT/deployment/default_sfm/run_server.sh" \
  --checkpoint "$FLOWER_RUN/seed_42/saved_models/epoch034.ckpt" \
  --training-config "$FLOWER_RUN/.hydra/config.yaml" \
  --weights ema \
  --port "${FLOWER_SERVER_PORT:-45587}" \
  --audit-file "$FLOWER_ROOT/logs/default_sfm_xy_rpc/model_audit.json" \
  "$@"
