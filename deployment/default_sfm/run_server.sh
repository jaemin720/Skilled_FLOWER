#!/usr/bin/env bash
set -euo pipefail
FLOWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FLOWER_ROOT"
exec "${FLOWER_PYTHON:-/home/jack/miniforge3/envs/flower/bin/python}" \
  server_flower_rpc.py "$@"
