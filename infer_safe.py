import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Iterator
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
import torch
import torch.nn as nn
from tqdm import tqdm


PATCH_SIZE = 256
MAX_BATCH_SIZE = 64
FLOOD_CLASS_INDEX = 1
PROBABILITY_NODATA = -9999.0
SNAP_CACHE_SCHEMA_VERSION = 2
SNAP_CACHE_RASTER_NAME = "sigma0_vv.tif"
SNAP_CACHE_META_NAME = "meta.json"
SNAP_CACHE_COMPLETE_NAME = ".complete"
SNAP_CACHE_CURRENT_NAME = "current"
SNAP_CACHE_GENERATIONS_NAME = "generations"
PERCENT_RE = re.compile(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?%")
PRODUCT_NAME_RE = re.compile(
    r"^(?P<platform>S1[AB])_(?P<mode>[A-Z0-9]+)_"
    r"(?P<product>[A-Z0-9]{4})_[A-Z0-9]{4}_"
    r"(?P<start>\d{8}T\d{6})_"
)


class InferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafeProduct:
    root: Path
    manifest: Path
    identifier: str
    platform: str | None
    product_type: str | None
    acquisition_mode: str | None
    start_time: datetime | None
    stop_time: datetime | None
    orbit_direction: str | None
    relative_orbit: int | None
    polarizations: frozenset[str]


class AlignedRasterPair:
    """Windowed pre/post reader on the pre-event raster grid."""

    def __init__(self, pre_path: Path, post_path: Path):
        self.pre = rasterio.open(pre_path)
        self.post = rasterio.open(post_path)
        self.post_vrt = None

        try:
            self._validate_source(self.pre, "pre-event")
            self._validate_source(self.post, "post-event")

            post_bounds = transform_bounds(
                self.post.crs,
                self.pre.crs,
                *self.post.bounds,
                densify_pts=21,
            )
            left = max(self.pre.bounds.left, post_bounds[0])
            bottom = max(self.pre.bounds.bottom, post_bounds[1])
            right = min(self.pre.bounds.right, post_bounds[2])
            top = min(self.pre.bounds.top, post_bounds[3])
            if left >= right or bottom >= top:
                raise InferenceError("The terrain-corrected products do not overlap.")

            floating_window = from_bounds(
                left,
                bottom,
                right,
                top,
                transform=self.pre.transform,
            )
            col_start = max(0, math.ceil(floating_window.col_off - 1e-6))
            row_start = max(0, math.ceil(floating_window.row_off - 1e-6))
            col_stop = min(
                self.pre.width,
                math.floor(floating_window.col_off + floating_window.width + 1e-6),
            )
            row_stop = min(
                self.pre.height,
                math.floor(floating_window.row_off + floating_window.height + 1e-6),
            )
            if col_stop <= col_start or row_stop <= row_start:
                raise InferenceError("The common raster grid is empty after alignment.")

            self.pre_window = Window(
                col_start,
                row_start,
                col_stop - col_start,
                row_stop - row_start,
            )
            self.width = int(self.pre_window.width)
            self.height = int(self.pre_window.height)
            self.crs = self.pre.crs
            self.transform = self.pre.window_transform(self.pre_window)
            self.post_vrt = WarpedVRT(
                self.post,
                crs=self.crs,
                transform=self.transform,
                width=self.width,
                height=self.height,
                resampling=Resampling.bilinear,
                dtype="float32",
                nodata=np.nan,
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_source(dataset, label: str) -> None:
        if dataset.count != 1:
            raise InferenceError(
                f"Expected one Sigma0_VV band in the {label} raster, "
                f"found {dataset.count}."
            )
        if dataset.crs is None:
            raise InferenceError(f"The {label} raster has no CRS.")
        if dataset.width <= 0 or dataset.height <= 0:
            raise InferenceError(f"The {label} raster is empty.")
        if abs(dataset.transform.a) <= 0 or abs(dataset.transform.e) <= 0:
            raise InferenceError(f"The {label} raster has an invalid transform.")

    def read(self, window: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.post_vrt is None:
            raise InferenceError("The aligned raster pair is closed.")

        absolute_pre_window = Window(
            self.pre_window.col_off + window.col_off,
            self.pre_window.row_off + window.row_off,
            window.width,
            window.height,
        )
        pre = self.pre.read(
            1,
            window=absolute_pre_window,
            out_dtype="float32",
        )
        pre_mask = self.pre.read_masks(1, window=absolute_pre_window) > 0
        post = self.post_vrt.read(
            1,
            window=window,
            out_dtype="float32",
        )
        post_mask = self.post_vrt.read_masks(1, window=window) > 0
        valid = (
            pre_mask
            & post_mask
            & np.isfinite(pre)
            & np.isfinite(post)
            & (pre > 0)
            & (post > 0)
        )
        return pre, post, valid

    def close(self) -> None:
        if self.post_vrt is not None:
            self.post_vrt.close()
            self.post_vrt = None
        if getattr(self, "post", None) is not None:
            self.post.close()
        if getattr(self, "pre", None) is not None:
            self.pre.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess two Sentinel-1 GRD SAFE products with ESA SNAP and "
            "run georeferenced DAM-Net sliding-window flood inference."
        )
    )
    parser.add_argument("pre_product", help="Pre-event .SAFE directory or manifest.safe")
    parser.add_argument("post_product", help="Post-event .SAFE directory or manifest.safe")
    parser.add_argument("--checkpoint", required=True, help="Trusted DAM-Net checkpoint")
    parser.add_argument("--output", required=True, help="Output flood-mask GeoTIFF")
    parser.add_argument(
        "--probability-output",
        help="Output flood-probability GeoTIFF (default: <output>_probability.tif)",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Acknowledge that the full-model checkpoint is trusted before unpickling",
    )
    parser.add_argument(
        "--gpt",
        default=os.environ.get("SNAP_GPT", "gpt"),
        help="ESA SNAP gpt executable or path (default: SNAP_GPT or gpt)",
    )
    parser.add_argument(
        "--orbit-type",
        default="Sentinel Precise (Auto Download)",
        help="SNAP orbit file type",
    )
    parser.add_argument(
        "--dem-name",
        default="Copernicus 30m Global DEM",
        help="SNAP terrain-correction DEM",
    )
    parser.add_argument(
        "--target-crs",
        default="AUTO:42001",
        help="SNAP target CRS (default: automatic UTM)",
    )
    parser.add_argument("--pixel-spacing", type=float, default=10.0)
    parser.add_argument("--db-min", type=float, default=-25.0)
    parser.add_argument("--db-max", type=float, default=0.0)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--work-dir",
        help="Parent directory for SNAP intermediates and disk accumulators",
    )
    parser.add_argument(
        "--snap-cache-dir",
        help=(
            "Persistent SNAP Sigma0 cache root (default: SNAP_CACHE_DIR or "
            "<work-dir>/snap-cache)"
        ),
    )
    parser.add_argument(
        "--no-snap-cache",
        action="store_true",
        help="Disable persistent SNAP preprocessing cache",
    )
    parser.add_argument(
        "--refresh-snap-cache",
        action="store_true",
        help="Recompute and replace matching SNAP cache entries",
    )
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if not args.trust_checkpoint:
        raise InferenceError(
            "Full-model checkpoints are Python pickles. Pass --trust-checkpoint "
            "only for a checkpoint from a trusted source."
        )
    if not 1 <= args.stride <= PATCH_SIZE:
        raise InferenceError(f"--stride must be between 1 and {PATCH_SIZE}.")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise InferenceError(
            f"--batch-size must be between 1 and {MAX_BATCH_SIZE}."
        )
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise InferenceError("--threshold must be finite and between 0 and 1.")
    if not math.isfinite(args.db_min) or not math.isfinite(args.db_max):
        raise InferenceError("--db-min and --db-max must be finite.")
    if args.db_min >= args.db_max:
        raise InferenceError("--db-min must be smaller than --db-max.")
    if not math.isfinite(args.pixel_spacing) or args.pixel_spacing <= 0:
        raise InferenceError("--pixel-spacing must be finite and positive.")

    snap_cache_dir = getattr(args, "snap_cache_dir", None)
    no_snap_cache = getattr(args, "no_snap_cache", False)
    refresh_snap_cache = getattr(args, "refresh_snap_cache", False)
    work_dir = getattr(args, "work_dir", None)
    env_cache_dir = os.environ.get("SNAP_CACHE_DIR")
    if no_snap_cache and snap_cache_dir:
        raise InferenceError(
            "--no-snap-cache cannot be combined with --snap-cache-dir."
        )
    if no_snap_cache and refresh_snap_cache:
        raise InferenceError(
            "--no-snap-cache cannot be combined with --refresh-snap-cache."
        )
    if refresh_snap_cache and not (snap_cache_dir or env_cache_dir or work_dir):
        raise InferenceError(
            "--refresh-snap-cache requires --snap-cache-dir, SNAP_CACHE_DIR, "
            "or --work-dir."
        )


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    mask_path = Path(args.output).expanduser().resolve()
    if mask_path.suffix.lower() not in {".tif", ".tiff"}:
        raise InferenceError("--output must use a .tif or .tiff suffix.")

    if args.probability_output:
        probability_path = Path(args.probability_output).expanduser().resolve()
    else:
        probability_path = mask_path.with_name(
            f"{mask_path.stem}_probability{mask_path.suffix}"
        )
    if probability_path.suffix.lower() not in {".tif", ".tiff"}:
        raise InferenceError("--probability-output must use a .tif or .tiff suffix.")
    if probability_path == mask_path:
        raise InferenceError("Flood-mask and probability outputs must be different files.")

    for path in (mask_path, probability_path):
        if path.exists() and not path.is_file():
            raise InferenceError(f"Output path exists and is not a file: {path}")
        if path.exists() and not args.overwrite:
            raise InferenceError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
    return mask_path, probability_path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _texts(root: ET.Element, local_name: str) -> list[str]:
    values = []
    for element in root.iter():
        if _local_name(element.tag) == local_name and element.text:
            value = element.text.strip()
            if value:
                values.append(value)
    return values


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace(" ", "T")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _is_polarization_measurement(path: Path, polarization: str) -> bool:
    name = path.name.lower()
    polarization = str(polarization).strip().lower()
    if polarization not in {"vv", "vh", "hh", "hv"}:
        raise ValueError(f"unsupported SAR polarization: {polarization}")
    return (
        f"-{polarization}-" in name
        or f"_{polarization}_" in name
        or name.endswith(f"-{polarization}.tif")
        or name.endswith(f"-{polarization}.tiff")
        or name.endswith(f"_{polarization}.tif")
        or name.endswith(f"_{polarization}.tiff")
    )


def _is_vv_measurement(path: Path) -> bool:
    return _is_polarization_measurement(path, "VV")


def _expected_polarizations(args=None) -> tuple[str, ...]:
    values = getattr(args, "polarizations", ("VV",))
    if isinstance(values, str):
        values = values.split(",")
    normalized = tuple(str(value).strip().upper() for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("polarizations must be a non-empty unique sequence")
    unsupported = sorted(set(normalized) - {"VV", "VH", "HH", "HV"})
    if unsupported:
        raise ValueError(f"unsupported SAR polarizations: {unsupported}")
    return normalized


def _cache_polarizations(inputs: dict[str, object]) -> tuple[str, ...]:
    return tuple(inputs.get("polarizations", ("VV",)))


def _cache_raster_name(inputs: dict[str, object]) -> str:
    polarizations = _cache_polarizations(inputs)
    if polarizations == ("VV",):
        return SNAP_CACHE_RASTER_NAME
    suffix = "_".join(value.lower() for value in polarizations)
    return f"sigma0_{suffix}.tif"


def resolve_safe_product(path_value: str) -> SafeProduct:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        manifest = path / "manifest.safe"
        root = path
    elif path.is_file() and path.name.lower() == "manifest.safe":
        manifest = path
        root = path.parent
    else:
        raise InferenceError(
            f"SAFE input must be an unpacked .SAFE directory or manifest.safe: {path}"
        )
    if not manifest.is_file():
        raise InferenceError(f"Missing manifest.safe in {root}")

    try:
        xml_root = ET.parse(manifest).getroot()
    except ET.ParseError as exc:
        raise InferenceError(f"Invalid SAFE manifest {manifest}: {exc}") from exc

    identifier = root.name
    name_match = PRODUCT_NAME_RE.match(identifier)
    platform = name_match.group("platform") if name_match else None
    acquisition_mode = name_match.group("mode") if name_match else None
    product_type = name_match.group("product")[:3] if name_match else None
    start_time = _parse_datetime(name_match.group("start")) if name_match else None

    family_names = _texts(xml_root, "familyName")
    platform_numbers = _texts(xml_root, "number")
    if family_names and platform_numbers:
        family = family_names[0].upper().replace("SENTINEL-", "S1")
        number = platform_numbers[0].upper()
        if family == "S1" and number in {"A", "B"}:
            platform = f"S1{number}"

    product_types = _texts(xml_root, "productType")
    modes = _texts(xml_root, "mode")
    start_times = _texts(xml_root, "startTime")
    stop_times = _texts(xml_root, "stopTime")
    passes = _texts(xml_root, "pass")
    polarizations = {
        value.upper() for value in _texts(xml_root, "transmitterReceiverPolarisation")
    }
    if product_types:
        product_type = product_types[0].upper()
    if modes:
        acquisition_mode = modes[0].upper()
    if start_times:
        start_time = _parse_datetime(start_times[0])
    stop_time = _parse_datetime(stop_times[0]) if stop_times else None
    orbit_direction = passes[0].upper() if passes else None

    relative_orbit = None
    relative_elements = [
        element
        for element in xml_root.iter()
        if _local_name(element.tag) == "relativeOrbitNumber" and element.text
    ]
    preferred = next(
        (
            element
            for element in relative_elements
            if element.attrib.get("type", "").lower() == "start"
        ),
        relative_elements[0] if relative_elements else None,
    )
    if preferred is not None:
        try:
            relative_orbit = int(preferred.text.strip())
        except (TypeError, ValueError):
            pass

    measurement_dir = root / "measurement"
    if not measurement_dir.is_dir():
        raise InferenceError(f"Missing SAFE measurement directory: {measurement_dir}")
    measurement_files = [item for item in measurement_dir.iterdir() if item.is_file()]
    vv_measurements = [item for item in measurement_files if _is_vv_measurement(item)]
    if not vv_measurements:
        raise InferenceError(
            f"No VV measurement GeoTIFF was found in {measurement_dir}."
        )
    polarizations.add("VV")

    safe = SafeProduct(
        root=root,
        manifest=manifest,
        identifier=identifier,
        platform=platform,
        product_type=product_type,
        acquisition_mode=acquisition_mode,
        start_time=start_time,
        stop_time=stop_time,
        orbit_direction=orbit_direction,
        relative_orbit=relative_orbit,
        polarizations=frozenset(polarizations),
    )
    validate_safe_product(safe)
    return safe


def validate_safe_product(product: SafeProduct) -> None:
    if not product.identifier.upper().endswith(".SAFE"):
        raise InferenceError(f"Not an unpacked SAFE product: {product.root}")
    if not product.product_type or not product.product_type.upper().startswith("GRD"):
        raise InferenceError(
            f"Only Sentinel-1 GRD products are supported: {product.identifier}"
        )
    if product.acquisition_mode and product.acquisition_mode.upper() != "IW":
        raise InferenceError(
            f"Only IW acquisition mode is supported, found "
            f"{product.acquisition_mode} in {product.identifier}."
        )
    if "VV" not in product.polarizations:
        raise InferenceError(f"VV polarization is missing from {product.identifier}.")


def validate_safe_pair(pre: SafeProduct, post: SafeProduct) -> None:
    if pre.start_time and post.start_time and pre.start_time >= post.start_time:
        raise InferenceError(
            "The pre-event product must have an earlier acquisition time than "
            "the post-event product."
        )
    if (
        pre.acquisition_mode
        and post.acquisition_mode
        and pre.acquisition_mode != post.acquisition_mode
    ):
        raise InferenceError(
            f"Acquisition modes differ: {pre.acquisition_mode} vs "
            f"{post.acquisition_mode}."
        )
    if (
        pre.orbit_direction
        and post.orbit_direction
        and pre.orbit_direction != post.orbit_direction
    ):
        raise InferenceError(
            f"Orbit directions differ: {pre.orbit_direction} vs "
            f"{post.orbit_direction}."
        )
    if (
        pre.relative_orbit is not None
        and post.relative_orbit is not None
        and pre.relative_orbit != post.relative_orbit
    ):
        raise InferenceError(
            f"Relative orbits differ: {pre.relative_orbit} vs "
            f"{post.relative_orbit}."
        )


def resolve_gpt(gpt_value: str) -> str:
    expanded = Path(gpt_value).expanduser()
    if expanded.parent != Path(".") or expanded.is_absolute():
        if expanded.is_file() and os.access(expanded, os.X_OK):
            return str(expanded.resolve())
        raise InferenceError(f"SNAP gpt executable is not runnable: {expanded}")
    resolved = shutil.which(gpt_value)
    if not resolved:
        raise InferenceError(
            "ESA SNAP gpt was not found. Install SNAP, add gpt to PATH, set "
            "SNAP_GPT, or pass --gpt /path/to/gpt."
        )
    return resolved


def build_gpt_command(
    gpt: str,
    graph: Path,
    product: SafeProduct,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        gpt,
        str(graph),
        f"-Pinput={product.manifest}",
        f"-Poutput={output}",
        f"-PorbitType={args.orbit_type}",
        f"-PdemName={args.dem_name}",
        f"-PtargetCrs={args.target_crs}",
        f"-PpixelSpacing={args.pixel_spacing}",
    ]


def run_gpt(command: list[str], log_path: Path, description: str) -> None:
    last_lines: deque[str] = deque(maxlen=40)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
    except OSError as exc:
        raise InferenceError(f"Failed to start SNAP gpt: {exc}") from exc

    latest_percent = 0
    with log_path.open("w", encoding="utf-8") as log_file, tqdm(
        total=100,
        desc=description,
        unit="%",
        leave=True,
    ) as progress:
        assert process.stdout is not None
        for raw_line in process.stdout:
            log_file.write(raw_line)
            line = raw_line.rstrip()
            if line:
                last_lines.append(line)
            match = PERCENT_RE.search(line)
            if match:
                percent = min(100, int(match.group(1)))
                if percent > latest_percent:
                    progress.update(percent - latest_percent)
                    latest_percent = percent
        return_code = process.wait()
        if return_code == 0 and latest_percent < 100:
            progress.update(100 - latest_percent)

    if return_code != 0:
        excerpt = "\n".join(last_lines)
        raise InferenceError(
            f"SNAP gpt failed with exit code {return_code}. Log: {log_path}\n"
            f"Last output lines:\n{excerpt}"
        )


def validate_sigma0_raster(
    path: Path,
    polarizations: tuple[str, ...] = ("VV",),
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise InferenceError(f"SNAP did not create the expected raster: {path}")

    with rasterio.open(path) as dataset:
        expected = tuple(str(value).upper() for value in polarizations)
        if dataset.count != len(expected):
            raise InferenceError(
                f"Unexpected SNAP output band count {dataset.count}; expected "
                f"{len(expected)} linear Sigma0 bands in order {expected}."
            )
        if dataset.crs is None:
            raise InferenceError("The SNAP output raster has no CRS.")
        if dataset.width <= 0 or dataset.height <= 0:
            raise InferenceError("The SNAP output raster is empty.")
        if abs(dataset.transform.a) <= 0 or abs(dataset.transform.e) <= 0:
            raise InferenceError("The SNAP output raster has an invalid transform.")
        for band_index, polarization in enumerate(expected):
            raw_description = dataset.descriptions[band_index]
            description = (raw_description or "").upper().replace("-", "_")
            expected_description = f"SIGMA0_{polarization}"
            if description and expected_description not in description:
                raise InferenceError(
                    f"Unexpected SNAP output band {band_index + 1} "
                    f"'{raw_description}'; expected {expected_description}."
                )
        sample_height = min(dataset.height, 1024)
        sample_width = min(dataset.width, 1024)
        sample = dataset.read(
            list(range(1, len(expected) + 1)),
            out_shape=(sample_height, sample_width),
            out_dtype="float32",
            resampling=Resampling.nearest,
        )
        valid_by_band = np.any(
            np.isfinite(sample) & (sample > 0),
            axis=(1, 2),
        )
        if not bool(valid_by_band.all()):
            raise InferenceError(
                "SNAP output has no finite positive Sigma0 samples for every "
                f"expected band {expected}: {path}"
            )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_sha256_file(path: Path, sample_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    offsets = {
        0,
        max(0, (size - sample_size) // 2),
        max(0, size - sample_size),
    }
    with path.open("rb") as source:
        for offset in sorted(offsets):
            source.seek(offset)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(source.read(sample_size))
    return digest.hexdigest()


def _safe_source_inventory(
    product: SafeProduct,
    polarizations: tuple[str, ...] = ("VV",),
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    measurement_dir = product.root / "measurement"
    annotation_dir = product.root / "annotation"

    if measurement_dir.is_dir():
        for path in sorted(measurement_dir.rglob("*")):
            if path.is_file() and any(
                _is_polarization_measurement(path, polarization)
                for polarization in polarizations
            ):
                inventory.append(
                    {
                        "path": path.relative_to(product.root).as_posix(),
                        "size": path.stat().st_size,
                        "sample_sha256": _sampled_sha256_file(path),
                    }
                )

    if annotation_dir.is_dir():
        for path in sorted(annotation_dir.rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "path": path.relative_to(product.root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    return inventory


def _sanitize_cache_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "product"


def build_snap_cache_key(
    product: SafeProduct,
    graph: Path,
    gpt: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, object]]:
    gpt_path = Path(gpt).resolve()
    polarizations = _expected_polarizations(args)
    inputs: dict[str, object] = {
        "schema_version": SNAP_CACHE_SCHEMA_VERSION,
        "product_identifier": product.identifier,
        "manifest_sha256": _sha256_file(product.manifest),
        "safe_source_inventory": _safe_source_inventory(
            product,
            polarizations,
        ),
        "graph_sha256": _sha256_file(graph),
        "gpt_path": str(gpt_path),
        "gpt_sha256": _sha256_file(gpt_path),
        "orbit_type": args.orbit_type,
        "dem_name": args.dem_name,
        "target_crs": args.target_crs,
        "pixel_spacing": format(float(args.pixel_spacing), ".12g"),
    }
    # Preserve existing VV-only cache keys. Dual-polarization entries carry an
    # explicit contract and use a semantic cache filename.
    if polarizations != ("VV",):
        inputs["polarizations"] = list(polarizations)
    serialized = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), inputs


def resolve_snap_cache_root(
    args: argparse.Namespace,
    work_parent: Path,
) -> Path | None:
    if getattr(args, "no_snap_cache", False):
        return None
    configured = getattr(args, "snap_cache_dir", None) or os.environ.get(
        "SNAP_CACHE_DIR"
    )
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(args, "work_dir", None):
        return (work_parent / "snap-cache").resolve()
    return None


def _snap_cache_entry_dir(
    cache_root: Path,
    product: SafeProduct,
    cache_key: str,
) -> Path:
    product_id = _sanitize_cache_component(product.identifier)
    return cache_root / "entries" / product_id / cache_key


@contextmanager
def snap_cache_lock(cache_root: Path, cache_key: str):
    lock_dir = cache_root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_key}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_snap_cache_generation(entry_dir: Path) -> Path | None:
    current_path = entry_dir / SNAP_CACHE_CURRENT_NAME
    if current_path.is_symlink():
        try:
            generation = current_path.resolve(strict=True)
            generations_dir = (
                entry_dir / SNAP_CACHE_GENERATIONS_NAME
            ).resolve(strict=True)
            generation.relative_to(generations_dir)
        except (OSError, RuntimeError, ValueError):
            return None
        return generation if generation.is_dir() else None
    if current_path.exists():
        return None
    return entry_dir


def load_snap_cache_entry(
    entry_dir: Path,
    expected_key: str,
    expected_inputs: dict[str, object],
) -> Path | None:
    generation = _resolve_snap_cache_generation(entry_dir)
    if generation is None:
        return None

    complete_path = generation / SNAP_CACHE_COMPLETE_NAME
    meta_path = generation / SNAP_CACHE_META_NAME
    raster_path = generation / _cache_raster_name(expected_inputs)
    if not complete_path.is_file() or not meta_path.is_file():
        return None
    if not raster_path.is_file() or raster_path.stat().st_size == 0:
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        if metadata.get("schema_version") != SNAP_CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("cache_key") != expected_key:
            return None
        if metadata.get("inputs") != expected_inputs:
            return None
        if metadata.get("raster_size") != raster_path.stat().st_size:
            return None
        validate_sigma0_raster(
            raster_path,
            _cache_polarizations(expected_inputs),
        )
    except (
        InferenceError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        rasterio.errors.RasterioError,
    ):
        return None
    return raster_path


def _unused_path(parent: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=parent)
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _quarantine_cache_path(path: Path, prefix: str) -> None:
    quarantine = _unused_path(path.parent, prefix)
    os.replace(path, quarantine)


def install_snap_cache_entry(
    cache_root: Path,
    entry_dir: Path,
    cache_key: str,
    inputs: dict[str, object],
    build: Callable[[Path], None],
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    if entry_dir.is_symlink() or (
        entry_dir.exists() and not entry_dir.is_dir()
    ):
        _quarantine_cache_path(entry_dir, f".invalid-{cache_key}-")
    entry_dir.mkdir(parents=True, exist_ok=True)

    generations_dir = entry_dir / SNAP_CACHE_GENERATIONS_NAME
    generations_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = Path(
        tempfile.mkdtemp(
            prefix=f".partial-{cache_key}-",
            dir=generations_dir,
        )
    )
    current_temp: Path | None = None
    try:
        raster_path = partial_dir / _cache_raster_name(inputs)
        build(raster_path)
        validate_sigma0_raster(raster_path, _cache_polarizations(inputs))
        metadata = {
            "schema_version": SNAP_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "inputs": inputs,
            "raster_size": raster_path.stat().st_size,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (partial_dir / SNAP_CACHE_META_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (partial_dir / SNAP_CACHE_COMPLETE_NAME).touch()

        generation_dir = generations_dir / partial_dir.name.replace(
            ".partial-", "generation-", 1
        )
        os.replace(partial_dir, generation_dir)

        current_path = entry_dir / SNAP_CACHE_CURRENT_NAME
        if current_path.exists() and not current_path.is_symlink():
            _quarantine_cache_path(current_path, ".invalid-current-")
        current_temp = _unused_path(entry_dir, ".current-")
        target = Path(SNAP_CACHE_GENERATIONS_NAME) / generation_dir.name
        os.symlink(target, current_temp)
        os.replace(current_temp, current_path)
        current_temp = None
        return generation_dir / _cache_raster_name(inputs)
    finally:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        if current_temp is not None and current_temp.is_symlink():
            current_temp.unlink()


def get_or_create_sigma0(
    gpt: str,
    graph: Path,
    product: SafeProduct,
    output: Path,
    args: argparse.Namespace,
    label: str,
    cache_root: Path | None,
    refresh: bool,
    build: Callable[[Path], None] | None = None,
) -> Path:
    if build is None:
        def builder(path: Path) -> None:
            try:
                preprocess_safe(gpt, graph, product, path, args, label)
            except Exception as exc:
                cache_log = path.with_suffix(".snap.log")
                run_log = output.with_suffix(".snap.log")
                if cache_log.is_file() and cache_log != run_log:
                    try:
                        run_log.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cache_log, run_log)
                    except OSError as copy_error:
                        raise InferenceError(
                            f"{exc}\nFailed to preserve SNAP log at {run_log}: "
                            f"{copy_error}"
                        ) from exc
                    raise InferenceError(
                        f"{exc}\nSNAP failure log preserved at: {run_log}"
                    ) from exc
                raise
    else:
        builder = build

    if cache_root is None:
        builder(output)
        return output

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key, inputs = build_snap_cache_key(product, graph, gpt, args)
    entry_dir = _snap_cache_entry_dir(cache_root, product, cache_key)
    if not refresh:
        cached = load_snap_cache_entry(entry_dir, cache_key, inputs)
        if cached is not None:
            print(f"SNAP cache hit ({label}): {cached}")
            return cached
        print(f"SNAP cache miss ({label}): {entry_dir}")
    else:
        print(f"SNAP cache refresh ({label}): {entry_dir}")

    with snap_cache_lock(cache_root, cache_key):
        if not refresh:
            cached = load_snap_cache_entry(entry_dir, cache_key, inputs)
            if cached is not None:
                print(f"SNAP cache hit after wait ({label}): {cached}")
                return cached
        cached = install_snap_cache_entry(
            cache_root,
            entry_dir,
            cache_key,
            inputs,
            builder,
        )
        print(f"SNAP cache installed ({label}): {cached}")
        return cached


def preprocess_safe(
    gpt: str,
    graph: Path,
    product: SafeProduct,
    output: Path,
    args: argparse.Namespace,
    label: str,
) -> None:
    is_cog_safe = "_COG.SAFE" in product.identifier.upper()
    if is_cog_safe:
        print(
            f"Warning: {product.identifier} is COG_SAFE. Direct SNAP support "
            "for its ZSTD-compressed measurement files depends on the installed "
            "SNAP/GDAL version."
        )
    command = build_gpt_command(gpt, graph, product, output, args)
    try:
        run_gpt(command, output.with_suffix(".snap.log"), f"SNAP {label}")
    except InferenceError as exc:
        if is_cog_safe:
            raise InferenceError(
                f"{exc}\nThe input is COG_SAFE. If the SNAP log reports an "
                "unsupported ZSTD/TIFF compression or reader error, convert it "
                "to original GRD SAFE with the official CDSE COG2GRD utility, "
                "or obtain the original SAFE product, then rerun inference."
            ) from exc
        raise
    validate_sigma0_raster(output, _expected_polarizations(args))


def sigma0_to_model_intensity(
    sigma0: np.ndarray,
    valid: np.ndarray,
    db_min: float,
    db_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(sigma0, dtype=np.float32)
    final_valid = np.asarray(valid, dtype=bool) & np.isfinite(sigma) & (sigma > 0)
    intensity = np.zeros(sigma.shape, dtype=np.float32)
    if np.any(final_valid):
        db = 10.0 * np.log10(sigma[final_valid])
        scaled = (db - db_min) / (db_max - db_min)
        intensity[final_valid] = np.clip(scaled, 0.0, 1.0) * 255.0
    return intensity, final_valid


def prepare_model_tile(
    sigma0: np.ndarray,
    valid: np.ndarray,
    db_min: float,
    db_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = sigma0.shape
    if height > PATCH_SIZE or width > PATCH_SIZE:
        raise InferenceError(
            f"Window exceeds the fixed {PATCH_SIZE}x{PATCH_SIZE} model input."
        )
    intensity, final_valid = sigma0_to_model_intensity(
        sigma0,
        valid,
        db_min,
        db_max,
    )
    padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    padded_valid = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    padded[:height, :width] = intensity
    padded_valid[:height, :width] = final_valid
    return np.repeat(padded[None, :, :], 3, axis=0), padded_valid


def axis_starts(length: int, stride: int, patch_size: int = PATCH_SIZE) -> list[int]:
    if length <= 0:
        raise ValueError("Axis length must be positive.")
    if not 1 <= stride <= patch_size:
        raise ValueError("Stride must be within the patch size.")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def iter_windows(width: int, height: int, stride: int) -> Iterator[Window]:
    for row in axis_starts(height, stride):
        for col in axis_starts(width, stride):
            yield Window(
                col,
                row,
                min(PATCH_SIZE, width - col),
                min(PATCH_SIZE, height - row),
            )


def make_blend_kernel(patch_size: int = PATCH_SIZE) -> np.ndarray:
    axis = np.hanning(patch_size).astype(np.float32)
    kernel = np.outer(axis, axis).astype(np.float32)
    np.maximum(kernel, np.float32(1e-3), out=kernel)
    kernel /= kernel.max()
    return kernel


def resolve_device(value: str) -> torch.device:
    normalized = value.lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but is not available.")
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise InferenceError(f"Invalid CUDA device: {value}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise InferenceError(
                f"CUDA device {index} does not exist; found "
                f"{torch.cuda.device_count()} device(s)."
            )
        return torch.device(normalized)
    raise InferenceError(f"Unsupported device value: {value}")


def validate_checkpoint_compatibility(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        model = model.module
    required = (
        "TACE_pre.proj_q.weight",
        "TACE_pre.proj_k.weight",
        "TACE_pre.proj_v.weight",
        "TACE_post.proj_q.weight",
        "TACE_post.proj_k.weight",
        "TACE_post.proj_v.weight",
    )
    state_keys = set(model.state_dict())
    missing = [name for name in required if name not in state_keys]
    if missing:
        raise InferenceError(
            "This checkpoint predates commit 11c309a and has no registered "
            "TACE Q/K/V weights. It cannot be used for reproducible inference; "
            "retrain and save a new checkpoint with the current code. Missing: "
            + ", ".join(missing)
        )
    return model


def load_trusted_model(checkpoint: Path, device: torch.device) -> nn.Module:
    if not checkpoint.is_file():
        raise InferenceError(f"Checkpoint does not exist: {checkpoint}")
    try:
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise InferenceError(f"Failed to load checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise InferenceError("Checkpoint does not contain a complete torch.nn.Module.")
    model = validate_checkpoint_compatibility(model)
    model.to(device)
    model.eval()
    return model


def _run_model_batch(
    model: nn.Module,
    pre_tiles: list[np.ndarray],
    post_tiles: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    try:
        pre_tensor = torch.from_numpy(np.stack(pre_tiles)).to(
            device,
            non_blocking=device.type == "cuda",
        )
        post_tensor = torch.from_numpy(np.stack(post_tiles)).to(
            device,
            non_blocking=device.type == "cuda",
        )
        with torch.inference_mode():
            logits = model(pre_tensor, post_tensor)
            expected = (len(pre_tiles), 2, PATCH_SIZE, PATCH_SIZE)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                shape = tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits)
                raise InferenceError(
                    f"Unexpected model output {shape}; expected {expected}."
                )
            probabilities = torch.softmax(logits.float(), dim=1)[:, FLOOD_CLASS_INDEX]
            if not torch.isfinite(probabilities).all():
                raise InferenceError("The model produced non-finite flood probabilities.")
        return probabilities.cpu().numpy()
    except MemoryError as exc:
        raise InferenceError(
            "Host memory was exhausted while assembling an inference batch. "
            "Rerun with a smaller --batch-size."
        ) from exc
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise InferenceError(
            "CUDA ran out of memory during 256x256 DAM-Net inference. "
            "Rerun with --batch-size 1, select another CUDA device, or use "
            "--device cpu."
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise InferenceError(
                f"{device.type.upper()} memory was exhausted during inference. "
                "Rerun with a smaller --batch-size or select another device."
            ) from exc
        raise


def check_accumulator_space(work_dir: Path, width: int, height: int) -> int:
    required = width * height * np.dtype(np.float32).itemsize * 2
    available = shutil.disk_usage(work_dir).free
    reserve = max(int(required * 1.2), required + 64 * 1024 * 1024)
    if available < reserve:
        raise InferenceError(
            f"Insufficient free space in {work_dir}: need about "
            f"{reserve / (1024 ** 3):.2f} GiB for accumulators, have "
            f"{available / (1024 ** 3):.2f} GiB."
        )
    return required


def check_output_space(
    mask_path: Path,
    probability_path: Path,
    width: int,
    height: int,
) -> None:
    pixels = width * height
    requirements = (
        (mask_path.parent, pixels * 2),
        (probability_path.parent, pixels * 5),
    )
    by_device: dict[int, tuple[Path, int]] = {}
    for directory, required in requirements:
        device = directory.stat().st_dev
        previous_directory, previous_required = by_device.get(
            device,
            (directory, 0),
        )
        by_device[device] = (previous_directory, previous_required + required)

    for directory, required in by_device.values():
        reserve = max(int(required * 1.3), required + 64 * 1024 * 1024)
        available = shutil.disk_usage(directory).free
        if available < reserve:
            raise InferenceError(
                f"Insufficient free space in output filesystem {directory}: "
                f"need about {reserve / (1024 ** 3):.2f} GiB, have "
                f"{available / (1024 ** 3):.2f} GiB."
            )


def run_sliding_inference(
    pair: AlignedRasterPair,
    model: nn.Module,
    device: torch.device,
    stride: int,
    batch_size: int,
    db_min: float,
    db_max: float,
    work_dir: Path,
) -> tuple[np.memmap, np.memmap]:
    check_accumulator_space(work_dir, pair.width, pair.height)
    probability_sum = np.memmap(
        work_dir / "probability_sum.dat",
        dtype="float32",
        mode="w+",
        shape=(pair.height, pair.width),
    )
    weight_sum = np.memmap(
        work_dir / "weight_sum.dat",
        dtype="float32",
        mode="w+",
        shape=(pair.height, pair.width),
    )
    probability_sum[:] = 0
    weight_sum[:] = 0

    blend = make_blend_kernel()
    total_windows = len(axis_starts(pair.width, stride)) * len(
        axis_starts(pair.height, stride)
    )
    queued_pre: list[np.ndarray] = []
    queued_post: list[np.ndarray] = []
    queued_records: list[tuple[Window, np.ndarray]] = []
    any_valid = False

    def flush(progress: tqdm) -> None:
        nonlocal any_valid
        if not queued_records:
            return
        probabilities = _run_model_batch(model, queued_pre, queued_post, device)
        for probability, (window, valid) in zip(probabilities, queued_records):
            height = int(window.height)
            width = int(window.width)
            row = int(window.row_off)
            col = int(window.col_off)
            tile_valid = valid[:height, :width]
            weights = blend[:height, :width] * tile_valid
            probability_sum[row:row + height, col:col + width] += (
                probability[:height, :width] * weights
            )
            weight_sum[row:row + height, col:col + width] += weights
            any_valid = any_valid or bool(np.any(tile_valid))
        progress.update(len(queued_records))
        queued_pre.clear()
        queued_post.clear()
        queued_records.clear()

    try:
        with tqdm(
            total=total_windows,
            desc="DAM-Net inference",
            unit="window",
            postfix={"device": str(device), "batch": batch_size},
        ) as progress:
            for window in iter_windows(pair.width, pair.height, stride):
                pre_sigma, post_sigma, common_valid = pair.read(window)
                pre_tile, pre_valid = prepare_model_tile(
                    pre_sigma,
                    common_valid,
                    db_min,
                    db_max,
                )
                post_tile, post_valid = prepare_model_tile(
                    post_sigma,
                    common_valid,
                    db_min,
                    db_max,
                )
                tile_valid = pre_valid & post_valid
                if not np.any(tile_valid):
                    progress.update(1)
                    continue
                queued_pre.append(pre_tile)
                queued_post.append(post_tile)
                queued_records.append((window, tile_valid))
                if len(queued_records) >= batch_size:
                    flush(progress)
            flush(progress)
    except MemoryError as exc:
        raise InferenceError(
            "Host memory was exhausted while preparing inference tiles. "
            "Rerun with a smaller --batch-size."
        ) from exc

    probability_sum.flush()
    weight_sum.flush()
    if not any_valid:
        raise InferenceError("The aligned products have no common valid Sigma0 pixels.")
    return probability_sum, weight_sum


def _temporary_tiff(final_path: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{final_path.stem}.",
        suffix=".partial.tif",
        dir=final_path.parent,
    )
    os.close(descriptor)
    os.unlink(temporary)
    return Path(temporary)


def _replace_output_pair(
    mask_temp: Path,
    mask_path: Path,
    probability_temp: Path,
    probability_path: Path,
) -> None:
    pairs = (
        (probability_temp, probability_path),
        (mask_temp, mask_path),
    )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _, final_path in pairs:
            if final_path.exists():
                descriptor, backup_value = tempfile.mkstemp(
                    prefix=f".{final_path.stem}.backup.",
                    suffix=final_path.suffix,
                    dir=final_path.parent,
                )
                os.close(descriptor)
                os.unlink(backup_value)
                backup_path = Path(backup_value)
                os.replace(final_path, backup_path)
                backups[final_path] = backup_path

        for temporary, final_path in pairs:
            os.replace(temporary, final_path)
            installed.append(final_path)
    except Exception:
        for final_path in installed:
            if final_path.is_file():
                final_path.unlink()
        for final_path, backup_path in backups.items():
            if backup_path.exists():
                os.replace(backup_path, final_path)
        raise
    else:
        for backup_path in backups.values():
            if backup_path.exists():
                backup_path.unlink()


def _output_windows(width: int, height: int) -> Iterator[Window]:
    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):
            yield Window(
                col,
                row,
                min(PATCH_SIZE, width - col),
                min(PATCH_SIZE, height - row),
            )


def write_outputs(
    mask_path: Path,
    probability_path: Path,
    pair: AlignedRasterPair,
    probability_sum: np.ndarray,
    weight_sum: np.ndarray,
    threshold: float,
    tags: dict[str, str],
) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    check_output_space(mask_path, probability_path, pair.width, pair.height)
    mask_temp = _temporary_tiff(mask_path)
    probability_temp = _temporary_tiff(probability_path)

    common_profile = {
        "driver": "GTiff",
        "width": pair.width,
        "height": pair.height,
        "count": 1,
        "crs": pair.crs,
        "transform": pair.transform,
        "tiled": True,
        "blockxsize": PATCH_SIZE,
        "blockysize": PATCH_SIZE,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    mask_profile = {
        **common_profile,
        "dtype": "uint8",
        "predictor": 2,
    }
    probability_profile = {
        **common_profile,
        "dtype": "float32",
        "nodata": PROBABILITY_NODATA,
        "predictor": 3,
    }

    windows = list(_output_windows(pair.width, pair.height))
    try:
        with rasterio.open(mask_temp, "w", **mask_profile) as mask_dst, rasterio.open(
            probability_temp,
            "w",
            **probability_profile,
        ) as probability_dst, tqdm(
            total=len(windows),
            desc="Writing outputs",
            unit="block",
        ) as progress:
            mask_dst.update_tags(**tags)
            probability_dst.update_tags(**tags)
            for window in windows:
                row = int(window.row_off)
                col = int(window.col_off)
                height = int(window.height)
                width = int(window.width)
                weights = np.asarray(
                    weight_sum[row:row + height, col:col + width],
                    dtype=np.float32,
                )
                sums = np.asarray(
                    probability_sum[row:row + height, col:col + width],
                    dtype=np.float32,
                )
                if not np.isfinite(weights).all() or not np.isfinite(sums).all():
                    raise InferenceError(
                        "Non-finite values were found in the stitched probability accumulators."
                    )
                valid = weights > 0
                probability = np.full(
                    (height, width),
                    PROBABILITY_NODATA,
                    dtype=np.float32,
                )
                np.divide(sums, weights, out=probability, where=valid)
                np.clip(probability, 0.0, 1.0, out=probability, where=valid)
                flood_mask = np.zeros((height, width), dtype=np.uint8)
                flood_mask[valid & (probability >= threshold)] = 255
                validity_mask = valid.astype(np.uint8) * 255

                mask_dst.write(flood_mask, 1, window=window)
                mask_dst.write_mask(validity_mask, window=window)
                probability_dst.write(probability, 1, window=window)
                probability_dst.write_mask(validity_mask, window=window)
                progress.update(1)

        _replace_output_pair(
            mask_temp,
            mask_path,
            probability_temp,
            probability_path,
        )
    except Exception:
        for temporary in (mask_temp, probability_temp):
            if temporary.exists():
                temporary.unlink()
        raise


def build_output_tags(
    pre: SafeProduct,
    post: SafeProduct,
    checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    return {
        "pre_product": pre.identifier,
        "post_product": post.identifier,
        "pre_acquisition": pre.start_time.isoformat() if pre.start_time else "unknown",
        "post_acquisition": post.start_time.isoformat() if post.start_time else "unknown",
        "checkpoint": checkpoint.name,
        "flood_class": str(FLOOD_CLASS_INDEX),
        "threshold": str(args.threshold),
        "db_min": str(args.db_min),
        "db_max": str(args.db_max),
        "patch_size": str(PATCH_SIZE),
        "stride": str(args.stride),
        "dem": args.dem_name,
        "requested_crs": args.target_crs,
        "pixel_spacing": str(args.pixel_spacing),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = None
    success = False

    try:
        validate_cli_args(args)
        mask_path, probability_path = resolve_output_paths(args)
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        graph = Path(__file__).resolve().parent / "snap" / "s1_grd_preprocess.xml"
        if not graph.is_file():
            raise InferenceError(f"SNAP graph does not exist: {graph}")

        pre = resolve_safe_product(args.pre_product)
        post = resolve_safe_product(args.post_product)
        validate_safe_pair(pre, post)
        gpt = resolve_gpt(args.gpt)
        device = resolve_device(args.device)

        work_parent = (
            Path(args.work_dir).expanduser().resolve()
            if args.work_dir
            else Path(tempfile.gettempdir())
        )
        work_parent.mkdir(parents=True, exist_ok=True)
        cache_root = resolve_snap_cache_root(args, work_parent)
        if cache_root is None:
            print("SNAP cache: disabled")
        else:
            cache_root.mkdir(parents=True, exist_ok=True)
            print(f"SNAP cache: {cache_root}")

        run_dir = Path(tempfile.mkdtemp(prefix="damnet-safe-", dir=work_parent))
        print(f"Working directory: {run_dir}")

        pre_raster = get_or_create_sigma0(
            gpt,
            graph,
            pre,
            run_dir / "pre_sigma0_vv.tif",
            args,
            "pre-event",
            cache_root,
            args.refresh_snap_cache,
        )
        post_raster = get_or_create_sigma0(
            gpt,
            graph,
            post,
            run_dir / "post_sigma0_vv.tif",
            args,
            "post-event",
            cache_root,
            args.refresh_snap_cache,
        )

        model = load_trusted_model(checkpoint, device)
        with AlignedRasterPair(pre_raster, post_raster) as pair:
            probability_sum, weight_sum = run_sliding_inference(
                pair,
                model,
                device,
                args.stride,
                args.batch_size,
                args.db_min,
                args.db_max,
                run_dir,
            )
            tags = build_output_tags(pre, post, checkpoint, args)
            write_outputs(
                mask_path,
                probability_path,
                pair,
                probability_sum,
                weight_sum,
                args.threshold,
                tags,
            )
            del probability_sum
            del weight_sum

        success = True
        print(f"Flood map: {mask_path}")
        print(f"Flood probability: {probability_path}")
        return 0
    except (InferenceError, OSError, rasterio.errors.RasterioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if run_dir is not None:
            print(f"Intermediates preserved in: {run_dir}", file=sys.stderr)
        return 1
    finally:
        if success and run_dir is not None and not args.keep_intermediate:
            shutil.rmtree(run_dir)
        elif success and run_dir is not None:
            print(f"Intermediates kept in: {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
