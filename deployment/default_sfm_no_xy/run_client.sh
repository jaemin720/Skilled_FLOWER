#!/usr/bin/env bash
# Run on Kingston; reuse the original 13-channel preprocessing bundle.
# The inference server selects the trained 11 channels after normalization.
set -euo pipefail
export FLOWER_SERVER_IP="${FLOWER_SERVER_IP:-10.206.178.31}"
export FLOWER_SERVER_PORT="${FLOWER_SERVER_PORT:-45588}"
export FLOWER_REQUEST_STEPS="${FLOWER_REQUEST_STEPS:-5}"
exec bash /home/csilab/share_projects/clean_share/team_codes/flower_default_sfm_rpc_20261028/run_client.sh "$@"
