#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

DEFAULT_PYTHON="3.12"
PYTHON_VERSION="$DEFAULT_PYTHON"
DEP_GROUPS=()
KNOWN_GROUPS=("local" "colab" "cuda" "dev")

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_uv_env.sh [--python 3.12] [group...]

Groups:
  local   Local notebook/dev runtime (default if no groups provided)
  colab   Colab MCP/runtime extras (Flask/ngrok/tools)
  cuda    CUDA stack for Linux GPU boxes (unsloth/bitsandbytes)
  dev     Extra developer tooling (pytest/jupyterlab)

Examples:
  scripts/setup_uv_env.sh
  scripts/setup_uv_env.sh local dev
  scripts/setup_uv_env.sh --python 3.12 local colab
EOF
}

contains_group() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --python" >&2
        usage
        exit 2
      fi
      PYTHON_VERSION="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if ! contains_group "$1" "${KNOWN_GROUPS[@]}"; then
        echo "Unknown group: $1" >&2
        echo "Allowed groups: ${KNOWN_GROUPS[*]}" >&2
        exit 2
      fi
      DEP_GROUPS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#DEP_GROUPS[@]} -eq 0 ]]; then
  DEP_GROUPS=("local")
fi

echo "Creating/updating .venv with Python $PYTHON_VERSION..."
uv venv --python "$PYTHON_VERSION" .venv

SYNC_ARGS=()
for g in "${DEP_GROUPS[@]}"; do
  SYNC_ARGS+=("--group" "$g")
done

echo "Syncing groups: ${DEP_GROUPS[*]}"
uv sync "${SYNC_ARGS[@]}"

echo "Done."
echo "Python: $ROOT_DIR/.venv/bin/python"
echo "Activate: source $ROOT_DIR/.venv/bin/activate"
