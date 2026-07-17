#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# User configuration: edit values in this section before running the script.
# =============================================================================
readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DATA_ROOT="/home/ubuntu/lhx/Sentinel1-SAR/restored_grd"
readonly OUTPUT_DIR="/home/ubuntu/lhx/Sentinel1-SAR/restored_grd/predicted"
readonly WORK_DIR="${HOME}/scratch/damnet-safe"
readonly SNAP_CACHE_DIR="${WORK_DIR}/snap-cache"
readonly SNAP_GPT="/usr/local/esa-snap/bin/gpt"
readonly CHECKPOINT_PATH="/home/ubuntu/lhx/DAM-Net/.tmp/S1GFloods_vitae_rsp/checkpoint_epoch_60.pth"

readonly PRE_SAFE_NAME="S1A_IW_GRDH_1SDV_20240402T141444_20240402T141510_053256_06745E_5249.SAFE"
readonly POST_SAFE_NAME="S1A_IW_GRDH_1SDV_20240414T141444_20240414T141509_053431_067B51_75FD.SAFE"
readonly OUTPUT_NAME="kulsary_flood_20240414.tif"

readonly TARGET_CRS="EPSG:32639"
readonly PIXEL_SPACING="10"
readonly STRIDE="128"
readonly BATCH_SIZE="4"
readonly DEVICE="cuda:0"
readonly DB_MIN="-25"
readonly DB_MAX="0"
readonly THRESHOLD="0.5"

# Set to true for the first diagnostic run or when replacing existing outputs.
readonly KEEP_INTERMEDIATE="false"
readonly OVERWRITE="false"
readonly SYNC_ENVIRONMENT="false"
readonly USE_SNAP_CACHE="true"
readonly REFRESH_SNAP_CACHE="false"
# =============================================================================

readonly PRE_SAFE_PATH="${DATA_ROOT}/${PRE_SAFE_NAME}"
readonly POST_SAFE_PATH="${DATA_ROOT}/${POST_SAFE_NAME}"
readonly OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"
readonly LOG_PATH="${OUTPUT_DIR}/${OUTPUT_NAME%.*}.log"

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

validate_boolean() {
    local name="$1"
    local value="$2"
    if [[ "$value" != "true" && "$value" != "false" ]]; then
        fail "${name} must be true or false, found: ${value}"
    fi
}

command -v uv >/dev/null 2>&1 || fail "uv is not installed or not in PATH."
[[ -x "$SNAP_GPT" ]] || fail "SNAP gpt is not executable: ${SNAP_GPT}"
[[ -f "$PROJECT_DIR/infer_safe.py" ]] || fail "infer_safe.py is missing from ${PROJECT_DIR}"
[[ -d "$PRE_SAFE_PATH" ]] || fail "pre-event SAFE directory is missing: ${PRE_SAFE_PATH}"
[[ -d "$POST_SAFE_PATH" ]] || fail "post-event SAFE directory is missing: ${POST_SAFE_PATH}"
[[ -f "$CHECKPOINT_PATH" ]] || fail "checkpoint is missing: ${CHECKPOINT_PATH}"

validate_boolean "KEEP_INTERMEDIATE" "$KEEP_INTERMEDIATE"
validate_boolean "OVERWRITE" "$OVERWRITE"
validate_boolean "SYNC_ENVIRONMENT" "$SYNC_ENVIRONMENT"
validate_boolean "USE_SNAP_CACHE" "$USE_SNAP_CACHE"
validate_boolean "REFRESH_SNAP_CACHE" "$REFRESH_SNAP_CACHE"
if [[ "$REFRESH_SNAP_CACHE" == "true" && "$USE_SNAP_CACHE" != "true" ]]; then
    fail "REFRESH_SNAP_CACHE requires USE_SNAP_CACHE=true."
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
if [[ "$USE_SNAP_CACHE" == "true" ]]; then
    mkdir -p "$SNAP_CACHE_DIR"
fi
cd "$PROJECT_DIR"

if [[ "$SYNC_ENVIRONMENT" == "true" ]]; then
    printf 'Synchronizing the locked Python environment...\n'
    uv sync --locked
fi

snap_cache_display="disabled"
if [[ "$USE_SNAP_CACHE" == "true" ]]; then
    snap_cache_display="$SNAP_CACHE_DIR"
fi

printf '\nDAM-Net SAFE inference configuration\n'
printf '  Project:       %s\n' "$PROJECT_DIR"
printf '  Pre-event:     %s\n' "$PRE_SAFE_PATH"
printf '  Post-event:    %s\n' "$POST_SAFE_PATH"
printf '  Checkpoint:    %s\n' "$CHECKPOINT_PATH"
printf '  Flood map:     %s\n' "$OUTPUT_PATH"
printf '  Probability:   %s\n' "${OUTPUT_PATH%.*}_probability.${OUTPUT_PATH##*.}"
printf '  Work dir:      %s\n' "$WORK_DIR"
printf '  SNAP cache:    %s\n' "$snap_cache_display"
printf '  Cache refresh: %s\n' "$REFRESH_SNAP_CACHE"
printf '  SNAP gpt:      %s\n' "$SNAP_GPT"
printf '  Device:        %s\n' "$DEVICE"
printf '  Log:           %s\n\n' "$LOG_PATH"

command=(
    uv run python infer_safe.py
    "$PRE_SAFE_PATH"
    "$POST_SAFE_PATH"
    --checkpoint "$CHECKPOINT_PATH"
    --output "$OUTPUT_PATH"
    --gpt "$SNAP_GPT"
    --target-crs "$TARGET_CRS"
    --pixel-spacing "$PIXEL_SPACING"
    --stride "$STRIDE"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --db-min "$DB_MIN"
    --db-max "$DB_MAX"
    --threshold "$THRESHOLD"
    --work-dir "$WORK_DIR"
    --trust-checkpoint
)

if [[ "$USE_SNAP_CACHE" == "true" ]]; then
    command+=(--snap-cache-dir "$SNAP_CACHE_DIR")
else
    command+=(--no-snap-cache)
fi
if [[ "$REFRESH_SNAP_CACHE" == "true" ]]; then
    command+=(--refresh-snap-cache)
fi
if [[ "$KEEP_INTERMEDIATE" == "true" ]]; then
    command+=(--keep-intermediate)
fi
if [[ "$OVERWRITE" == "true" ]]; then
    command+=(--overwrite)
fi

printf 'Starting inference...\n'
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n\n'

"${command[@]}" 2>&1 | tee "$LOG_PATH"

printf '\nInference complete.\n'
printf 'Flood map: %s\n' "$OUTPUT_PATH"
printf 'Probability map: %s\n' "${OUTPUT_PATH%.*}_probability.${OUTPUT_PATH##*.}"
printf 'Log: %s\n' "$LOG_PATH"
