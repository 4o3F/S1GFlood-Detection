#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_DATASET_DIR="/mnt/6D437734319B5084/lhx/S1GFloods"

usage() {
    printf 'Usage: bash %s [DATASET_DIR] [TRAIN_OPTIONS...]\n' "$(basename "$0")"
    printf '\n'
    printf 'Default dataset: %s\n' "$DEFAULT_DATASET_DIR"
    printf 'Environment override: S1GFLOODS_DATASET=/path/to/S1GFloods\n'
    printf '\n'
    printf 'Examples:\n'
    printf '  bash %s\n' "$(basename "$0")"
    printf '  bash %s /path/to/S1GFloods --backbone vitae --mode rsp\n' "$(basename "$0")"
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
    usage
    exit 0
fi

dataset_dir="${S1GFLOODS_DATASET:-$DEFAULT_DATASET_DIR}"
if [[ -n "${1:-}" && "${1:-}" != -* ]]; then
    dataset_dir="$1"
    shift
fi
dataset_dir="${dataset_dir%/}"

command -v uv >/dev/null 2>&1 || {
    printf 'Error: uv is not installed or not available in PATH.\n' >&2
    exit 1
}

for relative_dir in train/A train/B train/GT val/A val/B val/GT; do
    if [[ ! -d "$dataset_dir/$relative_dir" ]]; then
        printf 'Error: required dataset directory is missing: %s\n' \
            "$dataset_dir/$relative_dir" >&2
        exit 1
    fi
done

cd "$PROJECT_DIR"
uv sync --locked

mkdir -p .tmp/log
log_file=".tmp/log/train_$(date '+%Y%m%d-%H%M%S').log"

printf 'Project: %s\n' "$PROJECT_DIR"
printf 'Dataset: %s\n' "$dataset_dir"
printf 'Log: %s\n' "$PROJECT_DIR/$log_file"
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

uv run python train.py --dataset-dir "$dataset_dir/" "$@" 2>&1 | tee "$log_file"
