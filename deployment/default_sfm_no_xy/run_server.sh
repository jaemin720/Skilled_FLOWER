#!/usr/bin/env bash
# X/Y-excluded FLOWER, epoch 40, EMA; use a separate port from the XY model.
set -euo pipefail
FLOWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FLOWER_RUN="/data/jack/Skilled_FLOWER/checkpoints/default_sfm_no_xy/flower_sfm_no_xy_b64_e40_resume_20261028_215541"
exec bash "$FLOWER_ROOT/deployment/default_sfm/run_server.sh" \
  --checkpoint "$FLOWER_RUN/seed_42/saved_models/epoch029.ckpt" \
  --training-config "$FLOWER_RUN/.hydra/config.yaml" \
  --weights ema \
  --port "${FLOWER_SERVER_PORT:-45588}" \
  --audit-file "$FLOWER_ROOT/logs/default_sfm_no_xy_rpc/model_audit.json" \
  "$@"
