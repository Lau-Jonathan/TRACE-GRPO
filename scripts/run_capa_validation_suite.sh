#!/usr/bin/env bash
# One-click CAPA correctness validation suite.
#
# Usage:
#   bash trace_grpo/scripts/run_capa_validation_suite.sh
#   MODE=quick bash trace_grpo/scripts/run_capa_validation_suite.sh
#   TEST_GPU=1 bash trace_grpo/scripts/run_capa_validation_suite.sh
#   CONDA_ENV=trace_grpo bash trace_grpo/scripts/run_capa_validation_suite.sh
#
# Modes:
#   quick: CPU + packed-forward equivalence (fast)
#   full : quick + FA2 bf16 drift + actor RPC smoke (default)
#
# Notes for 4x H200:
#   - This suite is compatible with 4 GPUs, but these unit tests are mostly
#     single-process and use one GPU at a time.
#   - Set TEST_GPU to pick which card runs GPU tests.
set -euo pipefail

MODE=${MODE:-full}
CONDA_ENV=${CONDA_ENV:-trace_grpo}
TEST_GPU=${TEST_GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/..:$REPO_ROOT/verl"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-$REPO_ROOT/outputs/capa_validation/$STAMP}
mkdir -p "$LOG_DIR"

if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ] && command -v conda >/dev/null 2>&1; then
  RUNNER=(conda run --no-capture-output -n "$CONDA_ENV")
else
  RUNNER=()
fi

run_python_check() {
  if [ ${#RUNNER[@]} -gt 0 ]; then
    "${RUNNER[@]}" "$PYTHON_BIN" - <<'PY'
import importlib
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")

def has_mod(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

print("flash_attn_installed:", has_mod("flash_attn"))
PY
  else
    "$PYTHON_BIN" - <<'PY'
import importlib
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]={torch.cuda.get_device_name(i)}")

def has_mod(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

print("flash_attn_installed:", has_mod("flash_attn"))
PY
  fi
}

run_case() {
  local name="$1"
  local kind="$2"  # cpu | gpu
  local test_expr="$3"
  local log_file="$LOG_DIR/${name}.log"
  echo
  echo "[$(date +%H:%M:%S)] case=$name kind=$kind"
  echo "  expr: $test_expr"
  echo "  log : $log_file"

  set +e
  if [ "$kind" = "gpu" ]; then
    if [ ${#RUNNER[@]} -gt 0 ]; then
      CUDA_VISIBLE_DEVICES="$TEST_GPU" "${RUNNER[@]}" "$PYTHON_BIN" -m pytest -q "$test_expr" \
        >"$log_file" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$TEST_GPU" "$PYTHON_BIN" -m pytest -q "$test_expr" \
        >"$log_file" 2>&1
    fi
  else
    if [ ${#RUNNER[@]} -gt 0 ]; then
      "${RUNNER[@]}" "$PYTHON_BIN" -m pytest -q "$test_expr" \
        >"$log_file" 2>&1
    else
      "$PYTHON_BIN" -m pytest -q "$test_expr" >"$log_file" 2>&1
    fi
  fi
  local rc=$?
  set -e

  if [ $rc -eq 0 ]; then
    echo "  result: PASS"
    PASSED_CASES+=("$name")
  else
    echo "  result: FAIL (rc=$rc)"
    FAILED_CASES+=("$name")
  fi
}

echo "Repo: $REPO_ROOT"
echo "Mode: $MODE"
echo "Conda runner: ${RUNNER[*]:-(current shell)}"
echo "TEST_GPU=$TEST_GPU"
echo "LOG_DIR=$LOG_DIR"

echo
echo "== Preflight =="
run_python_check | tee "$LOG_DIR/preflight.log"

declare -a PASSED_CASES=()
declare -a FAILED_CASES=()

# Always-run core equivalence tests.
run_case "capa_prefix_kv_equivalence" "cpu" \
  "tests/test_prefix_kv_equivalence.py::test_capa_equivalence"
run_case "capa_prefix_kv_equivalence_gqa" "cpu" \
  "tests/test_prefix_kv_equivalence.py::test_capa_equivalence_gqa"
run_case "capa_qwen2_packed_vs_reference" "cpu" \
  "tests/test_capa_forward.py::test_qwen2_capa_forward_uses_single_packed_path_and_matches_reference"

if [ "$MODE" = "full" ]; then
  run_case "capa_fa2_bf16_drift" "gpu" \
    "tests/test_capa_forward.py::test_qwen2_capa_forward_fa2_bf16_matches_reference_with_drift_bound"
  run_case "actor_l3_rpc_smoke" "gpu" \
    "tests/test_actor_l3_rpc_smoke.py::test_fsdp_worker_l3_rpc_matches_reference_provider"
elif [ "$MODE" != "quick" ]; then
  echo "Unsupported MODE=$MODE (use quick or full)" >&2
  exit 2
fi

echo
echo "== Summary =="
echo "PASS: ${#PASSED_CASES[@]}"
for c in "${PASSED_CASES[@]}"; do
  echo "  - $c"
done
echo "FAIL: ${#FAILED_CASES[@]}"
for c in "${FAILED_CASES[@]}"; do
  echo "  - $c"
done
echo "Logs: $LOG_DIR"

if [ ${#FAILED_CASES[@]} -ne 0 ]; then
  exit 1
fi

echo "All CAPA validation cases passed."
