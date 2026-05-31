#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/nightly_ueba_pipeline.sh [options]

Options:
  --data-dir PATH       CERT R4.2 directory with logon/device/file/email/http CSV files.
  --labels PATH         Optional labels CSV/JSON/JSONL with user_id,date,risk_label.
  --output-root PATH    Output root directory. Default: outputs/nightly.
  --backend NAME        model_bakeoff backend: vllm or transformers. Default: vllm.
  --limit N             Dev/test examples per bake-off run. Default: 200.
  --max-rows N          Max rows per CERT CSV file for dataset preparation.
  --few-shot-k N        Few-shot examples for second bake-off run. Default: 3.
  --models "A B C"      Registry model keys. Default: qwen3/qwen2.5/smollm3/phi4.
  --setup-env           Run scripts/setup_uv_env.sh local cuda before experiments.
  --synthetic-smoke     Use tiny synthetic dataset instead of CERT data.
  --skip-baselines      Prepare data and run bake-off only.
  --skip-bakeoff        Prepare data and baselines only.
  --help                Show help.

Recommended tmux run:
  tmux new -s nirs
  scripts/nightly_ueba_pipeline.sh --data-dir data/cert-r4.2 --setup-env

Detach from tmux:
  Ctrl+B, then D
EOF
}

DATA_DIR=""
LABELS_PATH=""
OUTPUT_ROOT="outputs/nightly"
BACKEND="vllm"
LIMIT="200"
MAX_ROWS=""
FEW_SHOT_K="3"
MODELS="qwen3_4b_instruct_2507 qwen2_5_3b_instruct smollm3_3b phi4_mini_instruct"
SETUP_ENV=0
SYNTHETIC_SMOKE=0
SKIP_BAKEOFF=0
SKIP_BASELINES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --labels)
      LABELS_PATH="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --max-rows)
      MAX_ROWS="$2"
      shift 2
      ;;
    --few-shot-k)
      FEW_SHOT_K="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --setup-env)
      SETUP_ENV=1
      shift
      ;;
    --synthetic-smoke)
      SYNTHETIC_SMOKE=1
      shift
      ;;
    --skip-bakeoff)
      SKIP_BAKEOFF=1
      shift
      ;;
    --skip-baselines)
      SKIP_BASELINES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$SYNTHETIC_SMOKE" -eq 0 && -z "$DATA_DIR" ]]; then
  echo "ERROR: --data-dir is required unless --synthetic-smoke is set." >&2
  usage
  exit 2
fi

RUN_ID="nightly_ueba_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
DATASET_DIR="$RUN_DIR/dataset"
mkdir -p "$LOG_DIR" "$DATASET_DIR"

MASTER_LOG="$LOG_DIR/nightly.log"
GPU_LOG="$LOG_DIR/gpu.csv"
GPU_MONITOR_PID=""

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  {
    echo "### $name"
    echo "+ $*"
    "$@"
  } 2>&1 | tee "$LOG_DIR/${name}.log"
  log "DONE  $name"
}

start_gpu_monitor() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "timestamp,name,utilization_gpu,memory_used_mb,memory_total_mb,power_w" > "$GPU_LOG"
    (
      while true; do
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
          --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null || true
        sleep 30
      done
    ) &
    GPU_MONITOR_PID="$!"
    log "GPU monitor started: pid=$GPU_MONITOR_PID, log=$GPU_LOG"
  else
    log "nvidia-smi not found; GPU monitor skipped"
  fi
}

stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
    log "GPU monitor stopped"
  fi
}

write_run_manifest() {
  cat > "$RUN_DIR/run_manifest.md" <<EOF
# $RUN_ID

- started_at: $(date -Iseconds)
- repo: $(git rev-parse --show-toplevel 2>/dev/null || pwd)
- branch: $(git branch --show-current 2>/dev/null || echo unknown)
- commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)
- backend: $BACKEND
- limit: $LIMIT
- few_shot_k: $FEW_SHOT_K
- models: $MODELS
- synthetic_smoke: $SYNTHETIC_SMOKE
- skip_baselines: $SKIP_BASELINES
- skip_bakeoff: $SKIP_BAKEOFF
- data_dir: ${DATA_DIR:-none}
- labels: ${LABELS_PATH:-none}
EOF
}

write_summary() {
  {
    echo "# Nightly UEBA Pipeline Summary"
    echo
    echo "- run_dir: \`$RUN_DIR\`"
    echo "- backend: \`$BACKEND\`"
    echo "- models: \`$MODELS\`"
    echo
    echo "## Dataset"
    if [[ -f "$DATASET_DIR/summary.json" ]]; then
      echo
      echo '```json'
      cat "$DATASET_DIR/summary.json"
      echo
      echo '```'
    fi
    echo
    echo "## Baselines"
    for path in "$RUN_DIR"/baseline_*/metrics.json; do
      [[ -f "$path" ]] || continue
      echo
      echo "### $(basename "$(dirname "$path")")"
      echo '```json'
      cat "$path"
      echo
      echo '```'
    done
    echo
    echo "## Bake-off"
    for path in "$RUN_DIR"/bakeoff_*/summary.md; do
      [[ -f "$path" ]] || continue
      echo
      cat "$path"
    done
    echo
    echo "## Logs"
    echo "- master: \`$MASTER_LOG\`"
    echo "- gpu: \`$GPU_LOG\`"
  } > "$RUN_DIR/SUMMARY.md"
}

archive_results() {
  tar -czf "$RUN_DIR.tar.gz" -C "$OUTPUT_ROOT" "$RUN_ID"
  log "Archive written: $RUN_DIR.tar.gz"
}

trap 'stop_gpu_monitor; write_summary; archive_results' EXIT

log "Nightly UEBA pipeline run dir: $RUN_DIR"
write_run_manifest
start_gpu_monitor

if [[ "$SETUP_ENV" -eq 1 ]]; then
  run_step "00_setup_uv_env" scripts/setup_uv_env.sh local cuda
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Activated .venv"
fi

run_step "01_python_cuda_check" python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception as exc:
    print("torch_check_error", repr(exc))
PY

PREPARE_ARGS=(scripts/prepare_cert_dataset.py --output-dir "$DATASET_DIR" --seed 42)
if [[ "$SYNTHETIC_SMOKE" -eq 1 ]]; then
  PREPARE_ARGS+=(--synthetic-smoke)
else
  PREPARE_ARGS+=(--data-dir "$DATA_DIR")
fi
if [[ -n "$LABELS_PATH" ]]; then
  PREPARE_ARGS+=(--labels "$LABELS_PATH")
fi
if [[ -n "$MAX_ROWS" ]]; then
  PREPARE_ARGS+=(--max-rows-per-file "$MAX_ROWS")
fi
run_step "02_prepare_dataset" python "${PREPARE_ARGS[@]}"

if [[ "$SKIP_BASELINES" -eq 0 ]]; then
  run_step "03_baseline_logreg" python scripts/run_ueba_baseline.py \
    --train-jsonl "$DATASET_DIR/train.jsonl" \
    --test-jsonl "$DATASET_DIR/test.jsonl" \
    --output-dir "$RUN_DIR/baseline_logreg" \
    --model logreg

  run_step "04_baseline_rf" python scripts/run_ueba_baseline.py \
    --train-jsonl "$DATASET_DIR/train.jsonl" \
    --test-jsonl "$DATASET_DIR/test.jsonl" \
    --output-dir "$RUN_DIR/baseline_rf" \
    --model rf
else
  log "Baselines skipped by --skip-baselines"
fi

if [[ "$SKIP_BAKEOFF" -eq 0 ]]; then
  # Word-splitting of MODELS is intentional: model registry keys are simple tokens.
  # shellcheck disable=SC2206
  MODEL_ARRAY=($MODELS)
  run_step "05_bakeoff_zero_shot" python scripts/model_bakeoff.py \
    --dataset-jsonl "$DATASET_DIR/dev.jsonl" \
    --registry configs/model_registry.yaml \
    --output-dir "$RUN_DIR/bakeoff_zero_shot" \
    --backend "$BACKEND" \
    --limit "$LIMIT" \
    --models "${MODEL_ARRAY[@]}"

  run_step "06_bakeoff_few_shot" python scripts/model_bakeoff.py \
    --dataset-jsonl "$DATASET_DIR/dev.jsonl" \
    --registry configs/model_registry.yaml \
    --output-dir "$RUN_DIR/bakeoff_few_shot" \
    --backend "$BACKEND" \
    --few-shot-k "$FEW_SHOT_K" \
    --limit "$LIMIT" \
    --models "${MODEL_ARRAY[@]}"
else
  log "Bake-off skipped by --skip-bakeoff"
fi

write_summary
log "Nightly pipeline completed. Summary: $RUN_DIR/SUMMARY.md"
