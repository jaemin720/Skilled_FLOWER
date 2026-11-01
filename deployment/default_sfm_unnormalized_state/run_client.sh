#!/usr/bin/env bash
# Run on Kingston. The server restores physical state before removing X/Y.
set -euo pipefail
if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: bash run_client.sh TASK [existing env_rpc_client options]"
  echo "Tasks: audio_jack barlatch bnc hdmi knob lan outlet rotary_switch usb"
  echo "Defaults: FLOWER 10.206.178.31:45589, request_steps=5, monitor port=8125"
  echo "Environment: FLOWER_SERVER_IP, FLOWER_SERVER_PORT, FLOWER_REQUEST_STEPS, FLOWER_CLIENT_PYTHON, FLOWER_BUNDLE"
  exit 0
fi
TASK="$1"
shift
CLEAN_SHARE_ROOT="${CLEAN_SHARE_ROOT:-/home/csilab/share_projects/clean_share}"
FLOWER_CLIENT_PYTHON="${FLOWER_CLIENT_PYTHON:-/home/csilab/miniforge3/envs/hilserl/bin/python}"
FLOWER_BUNDLE="${FLOWER_BUNDLE:-$CLEAN_SHARE_ROOT/team_codes/flower_default_sfm_rpc_20261028/rpc_bundle}"
PROMPT="$("$FLOWER_CLIENT_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); task=sys.argv[2]; print(m[task])' "$FLOWER_BUNDLE/instructions.json" "$TASK")"
cd "$CLEAN_SHARE_ROOT"
exec "$FLOWER_CLIENT_PYTHON" -m Real_Robo.env_rpc_client \
  --trainer_ip "${FLOWER_SERVER_IP:-10.206.178.31}" \
  --trainer_port "${FLOWER_SERVER_PORT:-45589}" \
  --env_name default_sfm \
  --preprocess-bundle "$FLOWER_BUNDLE" \
  --prompt "$PROMPT" \
  --request_steps "${FLOWER_REQUEST_STEPS:-5}" \
  --monitor-port 8125 "$@"
