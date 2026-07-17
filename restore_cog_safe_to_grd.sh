#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# =============================================================================
# User configuration: edit values in this section before running the script.
# =============================================================================
readonly DATA_ROOT="/mnt/6D437734319B5084/lhx/Sentinel1-SAR"
readonly RESTORED_DATA_ROOT="${DATA_ROOT}/restored_grd"
readonly CONVERSION_WORK_ROOT="${DATA_ROOT}/.cog2grd-work"
readonly COG2GRD_IMAGE="cdse_utilities"
readonly MIN_FREE_GIB="50"
readonly KEEP_WORK_DIR="false"
readonly REUSE_COMPLETED="true"

readonly -a COG_SAFE_NAMES=(
    "S1A_IW_GRDH_1SDV_20240402T141444_20240402T141510_053256_06745E_D77F_COG.SAFE"
    "S1A_IW_GRDH_1SDV_20240414T141444_20240414T141509_053431_067B51_2A05_COG.SAFE"
    "S1A_IW_GRDH_1SDV_20240426T141445_20240426T141510_053606_068232_9DE6_COG.SAFE"
)

# These names are used only to print the follow-up predict_safe_pair.sh values.
readonly PRE_COG_SAFE_NAME="${COG_SAFE_NAMES[0]}"
readonly POST_COG_SAFE_NAME="${COG_SAFE_NAMES[1]}"
# =============================================================================

readonly PRODUCTS_DIR="${RESTORED_DATA_ROOT}/products"

RUN_DIR=""
LOG_PATH=""
SUCCESS="false"
RESOLVED_SAFE_PATH=""
declare -a PARTIAL_DIRS=()
declare -A RESTORED_SAFE_PATHS=()
declare -A RESTORED_SAFE_NAMES=()

usage() {
    printf 'Usage: bash %s\n' "$(basename "$0")"
    printf '\n'
    printf 'Restore the configured unpacked Sentinel-1 *_COG.SAFE products to\n'
    printf 'standard GRD SAFE products with the official CDSE Docker utility.\n'
    printf 'This script does not run SNAP or model inference.\n'
    printf '\n'
    printf 'Edit the configuration block at the top of the script before running.\n'
    printf 'Required image build command:\n'
    printf '  docker build "https://ghfast.top/github.com/eu-cdse/utilities.git#main" -t %s\n' \
        "$COG2GRD_IMAGE"
}

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

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || \
        fail "required command is not installed or not in PATH: ${command_name}"
}

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

validate_cog_safe() {
    local safe_name="$1"
    local safe_path="${DATA_ROOT}/${safe_name}"
    local match=""

    if [[ "$safe_name" != "$(basename -- "$safe_name")" || \
          "$safe_name" == "." || "$safe_name" == ".." || \
          "$safe_name" != *_COG.SAFE ]]; then
        fail "COG SAFE entry must be a basename ending in _COG.SAFE: ${safe_name}"
    fi
    [[ -d "$safe_path" ]] || fail "COG SAFE directory is missing: ${safe_path}"
    [[ -f "$safe_path/manifest.safe" ]] || \
        fail "manifest.safe is missing: ${safe_path}/manifest.safe"
    [[ -d "$safe_path/measurement" ]] || \
        fail "measurement directory is missing: ${safe_path}/measurement"
    [[ -d "$safe_path/annotation/calibration" ]] || \
        fail "calibration directory is missing: ${safe_path}/annotation/calibration"

    match="$(find "$safe_path/measurement" -maxdepth 1 -type f \
        \( -iname '*-cog.tif' -o -iname '*-cog.tiff' \) -print -quit)"
    [[ -n "$match" ]] || fail "no COG measurement TIFF was found in ${safe_path}"

    match="$(find "$safe_path/measurement" -maxdepth 1 -type f \
        \( -iname '*-vv-*-cog.tif' -o -iname '*-vv-*-cog.tiff' \) \
        -print -quit)"
    [[ -n "$match" ]] || fail "no VV COG measurement TIFF was found in ${safe_path}"

    match="$(find "$safe_path/annotation/calibration" -maxdepth 1 -type f \
        -iname 'noise-*-vv-*-cog.xml' -print -quit)"
    [[ -n "$match" ]] || fail "no VV COG noise XML was found in ${safe_path}"

    match="$(find "$safe_path/annotation/calibration" -maxdepth 1 -type f \
        -iname 'calibration-*-vv-*-cog.xml' -print -quit)"
    [[ -n "$match" ]] || fail "no VV COG calibration XML was found in ${safe_path}"
}

resolve_standard_safe() {
    local cache_dir="$1"
    local require_marker="$2"
    local safe_path=""
    local match=""
    local -a safe_dirs=()

    RESOLVED_SAFE_PATH=""
    [[ -d "$cache_dir" ]] || return 1
    if [[ "$require_marker" == "true" && ! -f "$cache_dir/.complete" ]]; then
        return 1
    fi

    mapfile -d '' -t safe_dirs < <(
        find "$cache_dir" -mindepth 1 -maxdepth 1 -type d \
            -name '*.SAFE' -print0
    )
    (( ${#safe_dirs[@]} == 1 )) || return 1
    safe_path="${safe_dirs[0]}"

    [[ "${safe_path^^}" != *"_COG.SAFE" ]] || return 1
    [[ -f "$safe_path/manifest.safe" ]] || return 1
    [[ -d "$safe_path/measurement" ]] || return 1
    [[ -d "$safe_path/annotation/calibration" ]] || return 1

    match="$(find "$safe_path/measurement" -maxdepth 1 -type f \
        \( -iname '*-vv-*.tif' -o -iname '*-vv-*.tiff' \) -print -quit)"
    [[ -n "$match" ]] || return 1

    match="$(find "$safe_path/annotation/calibration" -maxdepth 1 -type f \
        -iname 'noise-*-vv-*.xml' -print -quit)"
    [[ -n "$match" ]] || return 1

    match="$(find "$safe_path/annotation/calibration" -maxdepth 1 -type f \
        -iname 'calibration-*-vv-*.xml' -print -quit)"
    [[ -n "$match" ]] || return 1

    match="$(find "$safe_path/measurement" "$safe_path/annotation/calibration" \
        -maxdepth 1 -type f -iname '*-cog.*' -print -quit)"
    [[ -z "$match" ]] || return 1

    RESOLVED_SAFE_PATH="$safe_path"
    return 0
}

check_free_space() {
    local path="$1"
    local available_kib=""
    local required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    local -a lines=()

    mapfile -t lines < <(df -Pk --output=avail "$path")
    (( ${#lines[@]} >= 2 )) || fail "could not determine free space for ${path}"
    available_kib="${lines[1]//[[:space:]]/}"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || \
        fail "invalid free-space value for ${path}: ${available_kib}"

    printf 'Available conversion space: %s GiB\n' \
        "$((available_kib / 1024 / 1024))"
    if (( available_kib < required_kib )); then
        fail "conversion filesystem needs at least ${MIN_FREE_GIB} GiB free: ${path}"
    fi
}

create_product_link() {
    local source_name="$1"
    local safe_path="$2"
    local product_key="${source_name%.SAFE}"
    local safe_name="$(basename -- "$safe_path")"
    local link_path="${PRODUCTS_DIR}/${safe_name}"
    local target="../${product_key}/${safe_name}"

    if [[ -L "$link_path" ]]; then
        [[ "$(readlink -- "$link_path")" == "$target" ]] || \
            fail "product link already points elsewhere: ${link_path}"
    elif [[ -e "$link_path" ]]; then
        fail "product link path already exists and is not managed by this script: ${link_path}"
    else
        ln -s -- "$target" "$link_path"
    fi
}

restore_product() {
    local source_name="$1"
    local product_key="${source_name%.SAFE}"
    local source_path="${DATA_ROOT}/${source_name}"
    local cache_dir="${RESTORED_DATA_ROOT}/${product_key}"
    local input_zip="${RUN_DIR}/input/${product_key}.zip"
    local converted_dir="${RUN_DIR}/converted/${product_key}"
    local conversion_log="${RUN_DIR}/logs/${product_key}.log"
    local partial_dir=""
    local output_zip=""
    local safe_path=""
    local -a output_zips=()
    local -a docker_command=()

    if [[ -e "$cache_dir" || -L "$cache_dir" ]]; then
        if [[ "$REUSE_COMPLETED" == "true" ]] && \
           resolve_standard_safe "$cache_dir" "true"; then
            safe_path="$RESOLVED_SAFE_PATH"
            printf '\nReusing restored product:\n  %s\n  -> %s\n' \
                "$source_path" "$safe_path"
            RESTORED_SAFE_PATHS["$source_name"]="$safe_path"
            RESTORED_SAFE_NAMES["$source_name"]="$(basename -- "$safe_path")"
            create_product_link "$source_name" "$safe_path"
            return
        fi
        fail "restored cache exists but is incomplete or reuse is disabled: ${cache_dir}"
    fi

    printf '\nStaging COG SAFE: %s\n' "$source_path"
    (
        cd "$DATA_ROOT"
        zip -0 -q -r "$input_zip" "$source_name"
    )
    unzip -tq "$input_zip" >/dev/null || \
        fail "staged COG archive failed validation: ${input_zip}"
    unzip -Z1 "$input_zip" | grep -Fqx "${source_name}/" || \
        fail "staged archive does not contain the expected SAFE root: ${source_name}"

    mkdir -p "$converted_dir"
    docker_command=(
        docker run --rm
        --user "$(id -u):$(id -g)"
        --mount "type=bind,source=${RUN_DIR},target=/work"
        "$COG2GRD_IMAGE"
        COG2GRD.sh
        -i "/work/input/${product_key}.zip"
        -o "/work/converted/${product_key}"
    )

    printf 'Restoring standard GRD SAFE...\n'
    print_command "${docker_command[@]}"
    if ! "${docker_command[@]}" 2>&1 | tee "$conversion_log"; then
        fail "COG2GRD failed for ${source_name}; log: ${conversion_log}"
    fi

    mapfile -d '' -t output_zips < <(
        find "$converted_dir" -maxdepth 1 -type f -name '*.zip' -print0
    )
    (( ${#output_zips[@]} == 1 )) || \
        fail "COG2GRD must produce exactly one ZIP in ${converted_dir}, found ${#output_zips[@]}"
    output_zip="${output_zips[0]}"
    unzip -tq "$output_zip" >/dev/null || \
        fail "restored GRD archive failed validation: ${output_zip}"

    partial_dir="$(mktemp -d "${RESTORED_DATA_ROOT}/.partial-${product_key}.XXXXXX")"
    PARTIAL_DIRS+=("$partial_dir")
    unzip -q "$output_zip" -d "$partial_dir"

    resolve_standard_safe "$partial_dir" "false" || \
        fail "restored GRD SAFE structure is invalid in ${partial_dir}"
    safe_path="$RESOLVED_SAFE_PATH"

    cp -- "$conversion_log" "$partial_dir/conversion.log"
    printf 'source=%s\nimage=%s\n' "$source_path" "$COG2GRD_IMAGE" \
        > "$partial_dir/source.txt"
    touch "$partial_dir/.complete"
    mv -- "$partial_dir" "$cache_dir"
    safe_path="${cache_dir}/$(basename -- "$safe_path")"

    resolve_standard_safe "$cache_dir" "true" || \
        fail "restored cache failed validation after installation: ${cache_dir}"
    safe_path="$RESOLVED_SAFE_PATH"

    RESTORED_SAFE_PATHS["$source_name"]="$safe_path"
    RESTORED_SAFE_NAMES["$source_name"]="$(basename -- "$safe_path")"
    create_product_link "$source_name" "$safe_path"

    printf 'Restored product:\n  %s\n  -> %s\n' "$source_path" "$safe_path"
}

cleanup() {
    local status=$?
    local partial_dir=""

    if [[ "$SUCCESS" == "true" && "$KEEP_WORK_DIR" == "false" && \
          -n "$RUN_DIR" && -d "$RUN_DIR" && \
          "$RUN_DIR" == "${CONVERSION_WORK_ROOT%/}"/cog2grd-* ]]; then
        rm -rf -- "$RUN_DIR"
    elif [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        printf 'Conversion work directory preserved in: %s\n' "$RUN_DIR" >&2
    fi

    if [[ "$SUCCESS" != "true" ]]; then
        for partial_dir in "${PARTIAL_DIRS[@]}"; do
            if [[ -d "$partial_dir" ]]; then
                printf 'Partial restored cache preserved in: %s\n' "$partial_dir" >&2
            fi
        done
    fi

    return "$status"
}

main() {
    local source_name=""
    local safe_name=""
    local duplicate_key=""
    local -A seen_names=()

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        return 0
    fi
    (( $# == 0 )) || fail "unexpected arguments; use --help for usage"

    validate_boolean "KEEP_WORK_DIR" "$KEEP_WORK_DIR"
    validate_boolean "REUSE_COMPLETED" "$REUSE_COMPLETED"
    [[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || \
        fail "MIN_FREE_GIB must be a non-negative integer: ${MIN_FREE_GIB}"
    (( ${#COG_SAFE_NAMES[@]} > 0 )) || fail "COG_SAFE_NAMES must not be empty"

    require_command docker
    require_command zip
    require_command unzip
    require_command find
    require_command df
    require_command tee
    require_command readlink

    [[ -d "$DATA_ROOT" ]] || fail "data root does not exist: ${DATA_ROOT}"
    docker info >/dev/null 2>&1 || \
        fail "Docker daemon is unavailable or the current user lacks permission"
    if ! docker image inspect "$COG2GRD_IMAGE" >/dev/null 2>&1; then
        fail "Docker image '${COG2GRD_IMAGE}' is missing. Build it with: docker build \"https://ghfast.top/github.com/eu-cdse/utilities.git#main\" -t ${COG2GRD_IMAGE}"
    fi
    docker run --rm "$COG2GRD_IMAGE" COG2GRD.sh -v >/dev/null 2>&1 || \
        fail "Docker image does not provide a working COG2GRD.sh: ${COG2GRD_IMAGE}"

    for source_name in "${COG_SAFE_NAMES[@]}"; do
        duplicate_key="${source_name,,}"
        [[ -z "${seen_names[$duplicate_key]:-}" ]] || \
            fail "duplicate COG SAFE entry: ${source_name}"
        seen_names["$duplicate_key"]="true"
        validate_cog_safe "$source_name"
    done
    [[ -n "${seen_names[${PRE_COG_SAFE_NAME,,}]:-}" ]] || \
        fail "PRE_COG_SAFE_NAME is not listed in COG_SAFE_NAMES: ${PRE_COG_SAFE_NAME}"
    [[ -n "${seen_names[${POST_COG_SAFE_NAME,,}]:-}" ]] || \
        fail "POST_COG_SAFE_NAME is not listed in COG_SAFE_NAMES: ${POST_COG_SAFE_NAME}"
    [[ "$PRE_COG_SAFE_NAME" != "$POST_COG_SAFE_NAME" ]] || \
        fail "PRE_COG_SAFE_NAME and POST_COG_SAFE_NAME must differ"

    mkdir -p "$RESTORED_DATA_ROOT" "$PRODUCTS_DIR" "$CONVERSION_WORK_ROOT"
    check_free_space "$CONVERSION_WORK_ROOT"

    LOG_PATH="${RESTORED_DATA_ROOT}/restore_$(date '+%Y%m%d-%H%M%S').log"
    exec > >(tee -a "$LOG_PATH") 2>&1

    RUN_DIR="$(mktemp -d "${CONVERSION_WORK_ROOT%/}/cog2grd-XXXXXX")"
    mkdir -p "$RUN_DIR/input" "$RUN_DIR/converted" "$RUN_DIR/logs"

    printf '\nSentinel-1 COG SAFE restoration\n'
    printf '  Input root:      %s\n' "$DATA_ROOT"
    printf '  Restored root:   %s\n' "$RESTORED_DATA_ROOT"
    printf '  Product links:   %s\n' "$PRODUCTS_DIR"
    printf '  Work directory:  %s\n' "$RUN_DIR"
    printf '  Docker image:    %s\n' "$COG2GRD_IMAGE"
    printf '  Log:             %s\n' "$LOG_PATH"

    for source_name in "${COG_SAFE_NAMES[@]}"; do
        restore_product "$source_name"
    done

    printf '\nRestored product mappings\n'
    for source_name in "${COG_SAFE_NAMES[@]}"; do
        printf '  %s\n    -> %s\n' \
            "$source_name" "${RESTORED_SAFE_PATHS[$source_name]}"
    done

    printf '\nUse these values in predict_safe_pair.sh:\n'
    printf 'readonly DATA_ROOT="%s"\n' "$PRODUCTS_DIR"
    printf 'readonly PRE_SAFE_NAME="%s"\n' \
        "${RESTORED_SAFE_NAMES[$PRE_COG_SAFE_NAME]}"
    printf 'readonly POST_SAFE_NAME="%s"\n' \
        "${RESTORED_SAFE_NAMES[$POST_COG_SAFE_NAME]}"

    printf '\nAvailable standard GRD SAFE names:\n'
    for source_name in "${COG_SAFE_NAMES[@]}"; do
        safe_name="${RESTORED_SAFE_NAMES[$source_name]}"
        printf '  %s\n' "$safe_name"
    done

    SUCCESS="true"
    printf '\nRestoration complete. No SNAP processing or prediction was run.\n'
    printf 'Log: %s\n' "$LOG_PATH"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

main "$@"
