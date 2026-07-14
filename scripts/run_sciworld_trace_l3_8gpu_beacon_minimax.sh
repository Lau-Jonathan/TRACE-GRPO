#!/usr/bin/env bash
# Backward-compatible wrapper.
# The judge backend can be configured via env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[compat] run_sciworld_trace_l3_8gpu_beacon_minimax.sh is deprecated; forwarding to run_sciworld_trace_l3_8gpu_beacon_llmjudge.sh" >&2
exec bash "$SCRIPT_DIR/run_sciworld_trace_l3_8gpu_beacon_llmjudge.sh" "$@"
