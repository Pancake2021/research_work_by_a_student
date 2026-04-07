#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/.nightly"
LOG_DIR="$ROOT_DIR/logs/nightly"
mkdir -p "$STATE_DIR" "$LOG_DIR"

PID_FILE="$STATE_DIR/runner.pid"
LOG_FILE="$STATE_DIR/runner.log"
CMD_FILE="$STATE_DIR/runner.cmd"

usage() {
  cat <<'EOF'
Usage:
  scripts/macos_nightly_runner.sh start [baseline|full] [extra args for run_full_pipeline]
  scripts/macos_nightly_runner.sh status
  scripts/macos_nightly_runner.sh stop

Examples:
  scripts/macos_nightly_runner.sh start baseline
  scripts/macos_nightly_runner.sh start baseline --dataset synthetic --train-size 200 --test-size 50

Notes:
- Uses `caffeinate -is` to prevent idle/system sleep while running.
- Display may turn off and process continues.
- If you CLOSE THE LID, Mac usually sleeps anyway (unless clamshell mode: power + external display + keyboard/mouse).
EOF
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

build_command() {
  local profile="${1:-baseline}"
  shift || true

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  if [[ "$profile" == "baseline" ]]; then
    echo "python scripts/run_full_pipeline.py --mode baseline --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts} --json-metrics $*"
    return 0
  fi

  if [[ "$profile" == "full" ]]; then
    if [[ "${FORCE_LOCAL_RL:-0}" != "1" ]]; then
      cat >&2 <<'EOF'
Refusing `full` on local Mac by default.
Reason: this project targets CUDA/Colab GPU stack (Unsloth/TRL/bitsandbytes);
on Apple Silicon results are typically unstable/slow and 7 experiments are unlikely to finish overnight.
If you still want to try, rerun with FORCE_LOCAL_RL=1.
EOF
      return 2
    fi

    # Sequential chain for local attempt (still risky on Apple Silicon).
    echo "bash -lc 'set -e; \
      python scripts/run_full_pipeline.py --mode baseline --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp01 --json-metrics; \
      python scripts/run_full_pipeline.py --mode grpo --reward accuracy --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp02 --json-metrics; \
      python scripts/run_full_pipeline.py --mode grpo --reward reasoning --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp03 --json-metrics; \
      python scripts/run_full_pipeline.py --mode grpo --reward binary --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp04 --json-metrics; \
      python scripts/run_full_pipeline.py --mode ppo --reward reasoning --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp05 --json-metrics; \
      python scripts/run_full_pipeline.py --mode dapo --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp06 --json-metrics; \
      python scripts/run_full_pipeline.py --mode lambda_grpo --dataset synthetic --train-size 300 --test-size 100 --output-dir ./outputs/nightly_${ts}_exp07 --json-metrics'"
    return 0
  fi

  echo "Unknown profile: $profile" >&2
  return 2
}

start_runner() {
  local profile="${1:-baseline}"
  shift || true

  if is_running; then
    echo "Runner already active (pid=$(cat "$PID_FILE"))"
    exit 1
  fi

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script is intended for macOS." >&2
    exit 1
  fi

  local run_cmd
  run_cmd="$(build_command "$profile" "$@")"

  {
    echo "[$(date -Iseconds)] profile=$profile"
    echo "cmd=$run_cmd"
  } > "$CMD_FILE"

  local full_cmd
  full_cmd="cd '$ROOT_DIR' && $run_cmd"

  nohup caffeinate -is bash -lc "$full_cmd" > "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  echo "Started nightly runner."
  echo "  pid: $pid"
  echo "  log: $LOG_FILE"
  echo "  cmd: $CMD_FILE"
}

status_runner() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "Runner is active (pid=$pid)"
    echo "Log tail:"
    tail -n 30 "$LOG_FILE" || true
  else
    echo "Runner is not active."
    [[ -f "$LOG_FILE" ]] && { echo "Last log tail:"; tail -n 30 "$LOG_FILE" || true; }
  fi
}

stop_runner() {
  if ! is_running; then
    echo "Runner is not active."
    rm -f "$PID_FILE"
    exit 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" || true
  fi

  rm -f "$PID_FILE"
  echo "Runner stopped."
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start)
      shift
      start_runner "$@"
      ;;
    status)
      status_runner
      ;;
    stop)
      stop_runner
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
