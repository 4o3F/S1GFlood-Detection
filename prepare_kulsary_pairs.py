"""Prepare Kulsary Orbit 159 as S1GFloods-compatible temporal pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from tqdm import tqdm

from infer_safe import (
    InferenceError,
    PATCH_SIZE,
    _snap_cache_entry_dir,
    build_snap_cache_key,
    get_or_create_sigma0,
    load_snap_cache_entry,
    resolve_gpt,
    resolve_safe_product,
    resolve_snap_cache_root,
    sigma0_to_model_intensity,
    validate_safe_product,
)
from utils.kulsary_temporal import (
    MASK_CRS,
    OUTPUT_SPLITS,
    PAIR_VARIANTS,
    ROLE_DATES,
    AssignedPair,
    MaskRef,
    TileKey,
    assign_spatial_blocks,
    compose_flood_mask,
    discover_mask_refs,
    expand_pair_variants,
    iter_full_windows,
    load_binary_water_mask,
    spatial_block_key,
)


CONVERTER_VERSION = "1.1.0"
OUTPUT_SUBDIRS = ("A", "B", "GT", "WATER_GT_A", "WATER_GT_B")
MASK_SUBDIRS = ("GT", "WATER_GT_A", "WATER_GT_B")
GT_SEMANTICS = "peak_water AND NOT variant baseline water"
WATER_GT_FORMULA = (
    "WATER_GT_A = full water[a_role]; WATER_GT_B = full water[b_role]"
)
MASK_ENCODING = "single-channel PNG mode L with values {0,255}"
DEFAULT_SAFE_ROOT = Path("/home/ubuntu/lhx/Sentinel1-SAR/restored_grd")
DEFAULT_WORK_DIR = Path.home() / "scratch" / "damnet-safe"
DEFAULT_GRAPH = Path(__file__).resolve().parent / "snap" / "s1_grd_preprocess.xml"
DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
DEFAULT_DEM_NAME = "Copernicus 30m Global DEM"
SPLIT_STRATEGY = (
    "spatial super-block hash split; block={block_tiles}x{block_tiles} tiles; "
    "both pair variants at one coordinate share a split"
)
PAIR_MANIFEST_COLUMNS = (
    "split",
    "filename",
    "pair_id",
    "variant",
    "chronological",
    "tile_row",
    "tile_col",
    "block_row",
    "block_col",
    "a_role",
    "b_role",
    "a_date",
    "b_date",
    "a_safe_id",
    "b_safe_id",
    "gt_formula",
    "water_gt_a_pixels",
    "water_gt_b_pixels",
    "water_gt_formula",
    "gt_positive_pixels",
    "gt_fraction",
    "valid_fraction",
)


@dataclass(frozen=True)
class CommonGrid:
    crs: object
    transform: object
    width: int
    height: int
    peak_window: Window
    bounds: tuple[float, float, float, float]


class Sigma0Stack:
    """Read three Sigma0 rasters on a mask-clipped peak-date grid."""

    def __init__(self, paths: dict[str, Path], mask_ref: MaskRef):
        self.datasets = {}
        self.vrts = {}
        try:
            self.datasets = {
                role: rasterio.open(paths[role]) for role in ROLE_DATES
            }
            for role, dataset in self.datasets.items():
                self._validate_source(dataset, role)
            self.grid = self._build_common_grid(mask_ref)
            for role in ("before", "after"):
                dataset = self.datasets[role]
                self.vrts[role] = WarpedVRT(
                    dataset,
                    crs=self.grid.crs,
                    transform=self.grid.transform,
                    width=self.grid.width,
                    height=self.grid.height,
                    resampling=Resampling.bilinear,
                    dtype="float32",
                    src_nodata=dataset.nodata,
                    nodata=np.nan,
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_source(dataset, role: str) -> None:
        if dataset.count != 1:
            raise InferenceError(
                f"expected one Sigma0_VV band for {role}, found {dataset.count}"
            )
        if dataset.crs is None:
            raise InferenceError(f"Sigma0 raster has no CRS for {role}")
        if dataset.width <= 0 or dataset.height <= 0:
            raise InferenceError(f"Sigma0 raster is empty for {role}")
        if abs(dataset.transform.a) <= 0 or abs(dataset.transform.e) <= 0:
            raise InferenceError(f"Sigma0 raster has an invalid transform for {role}")

    def _build_common_grid(self, mask_ref: MaskRef) -> CommonGrid:
        peak = self.datasets["peak"]
        left, bottom, right, top = peak.bounds

        for role in ("before", "after"):
            dataset = self.datasets[role]
            transformed = transform_bounds(
                dataset.crs,
                peak.crs,
                *dataset.bounds,
                densify_pts=21,
            )
            left = max(left, transformed[0])
            bottom = max(bottom, transformed[1])
            right = min(right, transformed[2])
            top = min(top, transformed[3])

        mask_bounds = array_bounds(
            mask_ref.size[1],
            mask_ref.size[0],
            mask_ref.transform,
        )
        transformed_mask_bounds = transform_bounds(
            MASK_CRS,
            peak.crs,
            *mask_bounds,
            densify_pts=21,
        )
        left = max(left, transformed_mask_bounds[0])
        bottom = max(bottom, transformed_mask_bounds[1])
        right = min(right, transformed_mask_bounds[2])
        top = min(top, transformed_mask_bounds[3])

        if left >= right or bottom >= top:
            source_bounds = {
                role: tuple(dataset.bounds)
                for role, dataset in self.datasets.items()
            }
            raise InferenceError(
                "the three terrain-corrected products and water-mask extent do "
                f"not overlap: sigma0={source_bounds}, mask={mask_bounds}"
            )

        floating = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=peak.transform,
        )
        col_start = max(0, math.ceil(floating.col_off - 1e-6))
        row_start = max(0, math.ceil(floating.row_off - 1e-6))
        col_stop = min(
            peak.width,
            math.floor(floating.col_off + floating.width + 1e-6),
        )
        row_stop = min(
            peak.height,
            math.floor(floating.row_off + floating.height + 1e-6),
        )
        if col_stop <= col_start or row_stop <= row_start:
            raise InferenceError("the common mask-clipped peak-grid window is empty")

        peak_window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            row_stop - row_start,
        )
        transform = peak.window_transform(peak_window)
        bounds = window_bounds(peak_window, peak.transform)
        return CommonGrid(
            crs=peak.crs,
            transform=transform,
            width=int(peak_window.width),
            height=int(peak_window.height),
            peak_window=peak_window,
            bounds=tuple(float(value) for value in bounds),
        )

    def read(self, window: Window) -> tuple[dict[str, np.ndarray], np.ndarray]:
        peak_window = Window(
            self.grid.peak_window.col_off + window.col_off,
            self.grid.peak_window.row_off + window.row_off,
            window.width,
            window.height,
        )
        arrays = {
            "peak": self.datasets["peak"].read(
                1,
                window=peak_window,
                out_dtype="float32",
            ),
            "before": self.vrts["before"].read(
                1,
                window=window,
                out_dtype="float32",
            ),
            "after": self.vrts["after"].read(
                1,
                window=window,
                out_dtype="float32",
            ),
        }
        masks = {
            "peak": self.datasets["peak"].read_masks(
                1,
                window=peak_window,
            )
            > 0,
            "before": self.vrts["before"].read_masks(1, window=window) > 0,
            "after": self.vrts["after"].read_masks(1, window=window) > 0,
        }
        valid = np.ones(arrays["peak"].shape, dtype=bool)
        for role in ROLE_DATES:
            valid &= masks[role]
            valid &= np.isfinite(arrays[role])
            valid &= arrays[role] > 0
        return arrays, valid

    def close(self) -> None:
        for vrt in self.vrts.values():
            vrt.close()
        self.vrts.clear()
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert three Kulsary Orbit 159 Sentinel-1 GRD scenes and "
            "PNG+PGW water masks into S1GFloods temporal pairs."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="mask root")
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=DEFAULT_SAFE_ROOT,
        help=f"restored standard GRD SAFE root (default: {DEFAULT_SAFE_ROOT})",
    )
    parser.add_argument("--output", type=Path, required=True, help="output root")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--snap-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--gpt",
        default=os.environ.get("SNAP_GPT", "gpt"),
        help="ESA SNAP gpt executable or path",
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--orbit-type", default=DEFAULT_ORBIT_TYPE)
    parser.add_argument("--dem-name", default=DEFAULT_DEM_NAME)
    parser.add_argument("--target-crs", default="EPSG:32639")
    parser.add_argument("--pixel-spacing", type=float, default=10.0)
    parser.add_argument("--db-min", type=float, default=-25.0)
    parser.add_argument("--db-max", type=float, default=0.0)
    parser.add_argument("--block-tiles", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-snap-cache", action="store_true")
    parser.add_argument("--refresh-snap-cache", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("split ratios must be finite and positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--train-ratio, --val-ratio, and --test-ratio must sum to 1")
    if args.block_tiles <= 0:
        raise ValueError("--block-tiles must be positive")
    if not math.isfinite(args.pixel_spacing) or args.pixel_spacing <= 0:
        raise ValueError("--pixel-spacing must be finite and positive")
    if not math.isfinite(args.db_min) or not math.isfinite(args.db_max):
        raise ValueError("--db-min and --db-max must be finite")
    if args.db_min >= args.db_max:
        raise ValueError("--db-min must be smaller than --db-max")
    if args.no_snap_cache and args.snap_cache_dir:
        raise ValueError("--no-snap-cache cannot be combined with --snap-cache-dir")
    if args.no_snap_cache and args.refresh_snap_cache:
        raise ValueError(
            "--no-snap-cache cannot be combined with --refresh-snap-cache"
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_paths(args: argparse.Namespace):
    source = args.source.expanduser().resolve()
    safe_root = args.safe_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    graph = args.graph.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"mask source directory is missing: {source}")
    if not safe_root.is_dir():
        raise FileNotFoundError(f"SAFE root directory is missing: {safe_root}")
    if not graph.is_file():
        raise FileNotFoundError(f"SNAP graph is missing: {graph}")
    if output == source or output == safe_root:
        raise ValueError("output must be different from source and --safe-root")

    cache_root = resolve_snap_cache_root(args, work_dir)
    staging = output.with_name(f".{output.name}.partial")
    writable_paths = [("output", output), ("work directory", work_dir)]
    if cache_root is not None:
        writable_paths.append(("SNAP cache", cache_root))
    for label, path in writable_paths:
        if _is_within(path, source) or _is_within(path, safe_root):
            raise ValueError(
                f"{label} must be outside the read-only mask and SAFE roots: {path}"
            )

    support_paths = [("work directory", work_dir)]
    if cache_root is not None:
        support_paths.append(("SNAP cache", cache_root))
    for label, path in support_paths:
        if (
            _is_within(path, output)
            or _is_within(output, path)
            or _is_within(path, staging)
            or _is_within(staging, path)
        ):
            raise ValueError(
                f"{label} must not overlap the output or staging tree: {path}"
            )
    return source, safe_root, output, work_dir, graph, cache_root


def _discover_products(safe_root: Path):
    safe_paths_by_target = {}
    for search_root in (safe_root, safe_root / "products"):
        if not search_root.is_dir():
            continue
        for path in search_root.iterdir():
            if path.is_dir() and path.name.upper().endswith(".SAFE"):
                target = path.resolve()
                safe_paths_by_target[target] = target

    safe_paths = sorted(safe_paths_by_target, key=str)
    if not safe_paths:
        raise FileNotFoundError(
            "no unpacked SAFE products found directly under "
            f"{safe_root} or in {safe_root / 'products'}"
        )

    by_date = defaultdict(list)
    for path in safe_paths:
        product = resolve_safe_product(str(path))
        if product.start_time is not None and product.start_time.date() in ROLE_DATES.values():
            by_date[product.start_time.date()].append(product)

    products = {}
    for role, acquisition_date in ROLE_DATES.items():
        matches = by_date.get(acquisition_date, [])
        if len(matches) != 1:
            raise InferenceError(
                f"expected exactly one SAFE acquired on {acquisition_date.isoformat()} "
                f"for role {role}, found {len(matches)}"
            )
        validate_safe_product(matches[0])
        products[role] = matches[0]
    return products


def _validate_product_geometry(products) -> None:
    checks = (
        ("platform", {product.platform for product in products.values()}, {"S1A"}),
        ("product type", {product.product_type for product in products.values()}, {"GRD"}),
        ("acquisition mode", {product.acquisition_mode for product in products.values()}, {"IW"}),
        ("orbit direction", {product.orbit_direction for product in products.values()}, {"ASCENDING"}),
        ("relative orbit", {product.relative_orbit for product in products.values()}, {159}),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise InferenceError(
                f"Kulsary products have unexpected {label}: "
                f"{sorted(map(str, actual))}; expected {sorted(map(str, expected))}"
            )
    missing_vv = [
        product.identifier
        for product in products.values()
        if "VV" not in {value.upper() for value in product.polarizations}
    ]
    if missing_vv:
        raise InferenceError(
            f"Kulsary products are missing VV polarization: {', '.join(missing_vv)}"
        )


def _reject_cog_products(products) -> None:
    cog = [
        product.identifier
        for product in products.values()
        if "_COG.SAFE" in product.identifier.upper()
    ]
    if cog:
        raise InferenceError(
            "full conversion requires restored standard GRD SAFE products; "
            f"COG SAFE inputs were found: {', '.join(sorted(cog))}. "
            "Pass --safe-root pointing to the restored_grd directory."
        )


def _find_gpt_for_dry_run(value: str) -> str | None:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        resolved = expanded.resolve()
        return str(resolved) if resolved.is_file() and os.access(resolved, os.X_OK) else None
    return shutil.which(value)


def _probe_snap_cache(
    args: argparse.Namespace,
    products,
    graph: Path,
    cache_root: Path | None,
    gpt: str | None,
) -> dict[str, str]:
    if cache_root is None:
        return {role: "disabled" for role in ROLE_DATES}
    if gpt is None:
        return {
            role: "unavailable (gpt not found; cache key not computable)"
            for role in ROLE_DATES
        }

    statuses = {}
    for role, product in products.items():
        try:
            cache_key, inputs = build_snap_cache_key(product, graph, gpt, args)
            entry_dir = _snap_cache_entry_dir(cache_root, product, cache_key)
            cached = load_snap_cache_entry(entry_dir, cache_key, inputs)
            statuses[role] = "hit" if cached is not None else "miss"
        except (InferenceError, OSError, ValueError) as exc:
            statuses[role] = f"unavailable ({exc})"
    return statuses


def _count_source_gt_pixels(mask_refs: dict[str, MaskRef]) -> dict[str, int]:
    peak = load_binary_water_mask(mask_refs["peak"].png_path)
    counts = {}
    for variant in PAIR_VARIANTS:
        baseline = load_binary_water_mask(
            mask_refs[variant.gt_baseline_role].png_path
        )
        counts[variant.name] = int(compose_flood_mask(peak, baseline).sum())
    return counts


def _print_static_summary(
    source: Path,
    safe_root: Path,
    output: Path,
    mask_refs,
    source_gt_counts,
    products,
    cache_status,
) -> None:
    print(f"Mask source: {source}")
    print(f"SAFE root: {safe_root}")
    print(f"Output: {output}")
    print("Source masks:")
    for role in ROLE_DATES:
        ref = mask_refs[role]
        print(
            f"  {role}: {ref.png_path} size={ref.size} "
            f"water_pixels={ref.positive_pixels}"
        )
    print("Source-grid GT positive pixels:")
    for name, count in source_gt_counts.items():
        print(f"  {name}: {count:,}")
    print("SAFE role bindings (manifest acquisition dates):")
    for role in ROLE_DATES:
        product = products[role]
        print(
            f"  {role}: {product.start_time.date().isoformat()} "
            f"{product.identifier}"
        )
    print("Pair variants:")
    for variant in PAIR_VARIANTS:
        chronology = "CHRONOLOGICAL" if variant.chronological else "REVERSED CHRONOLOGY"
        print(
            f"  {variant.name}: A={ROLE_DATES[variant.a_role]} "
            f"B={ROLE_DATES[variant.b_role]} GT={variant.gt_formula} "
            f"WATER_GT_A={variant.a_role}_water "
            f"WATER_GT_B={variant.b_role}_water [{chronology}]"
        )
    print("SNAP cache probe:")
    for role in ROLE_DATES:
        print(f"  {role}: {cache_status[role]}")


def _reproject_mask(source: np.ndarray, ref: MaskRef, grid: CommonGrid) -> np.ndarray:
    destination = np.zeros((grid.height, grid.width), dtype=np.uint8)
    reproject(
        source=np.asarray(source, dtype=np.uint8),
        destination=destination,
        src_transform=ref.transform,
        src_crs=MASK_CRS,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
    )
    return destination > 0


def _warp_masks(mask_refs: dict[str, MaskRef], grid: CommonGrid):
    peak_ref = mask_refs["peak"]
    coverage_source = np.ones(
        (peak_ref.size[1], peak_ref.size[0]),
        dtype=np.uint8,
    )
    coverage = _reproject_mask(coverage_source, peak_ref, grid)
    water_masks = {
        role: _reproject_mask(
            load_binary_water_mask(mask_refs[role].png_path),
            mask_refs[role],
            grid,
        )
        for role in ROLE_DATES
    }
    return water_masks, coverage


def _tile_slice(tile: TileKey):
    row_start = tile.row * PATCH_SIZE
    col_start = tile.col * PATCH_SIZE
    return np.s_[
        row_start : row_start + PATCH_SIZE,
        col_start : col_start + PATCH_SIZE,
    ]


def _plan_tiles(
    stack: Sigma0Stack,
    mask_coverage: np.ndarray,
    args: argparse.Namespace,
):
    kept = []
    skips = []
    width_remainder = stack.grid.width % PATCH_SIZE
    height_remainder = stack.grid.height % PATCH_SIZE
    if width_remainder:
        skips.append(
            {"reason": "incomplete_right_edge", "width_pixels": width_remainder}
        )
    if height_remainder:
        skips.append(
            {"reason": "incomplete_bottom_edge", "height_pixels": height_remainder}
        )

    for tile, window in iter_full_windows(
        stack.grid.width,
        stack.grid.height,
        PATCH_SIZE,
    ):
        coverage = mask_coverage[_tile_slice(tile)]
        if not bool(coverage.all()):
            skips.append(
                {
                    "reason": "incomplete_mask_coverage",
                    "tile_row": tile.row,
                    "tile_col": tile.col,
                    "coverage_fraction": float(coverage.mean()),
                }
            )
            continue

        _, valid = stack.read(window)
        if not bool(valid.all()):
            skips.append(
                {
                    "reason": "invalid_common_pixels",
                    "tile_row": tile.row,
                    "tile_col": tile.col,
                    "valid_fraction": float(valid.mean()),
                }
            )
            continue
        kept.append(tile)

    split_by_tile = assign_spatial_blocks(
        kept,
        block_tiles=args.block_tiles,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return kept, expand_pair_variants(kept, split_by_tile), skips


def _ensure_split_dirs(staging_root: Path) -> None:
    for split_name in OUTPUT_SPLITS:
        for subdirectory in OUTPUT_SUBDIRS:
            (staging_root / split_name / subdirectory).mkdir(
                parents=True,
                exist_ok=True,
            )


def _to_rgb(
    sigma0: np.ndarray,
    valid: np.ndarray,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    intensity, final_valid = sigma0_to_model_intensity(
        sigma0,
        valid,
        db_min,
        db_max,
    )
    if not bool(final_valid.all()):
        raise RuntimeError("a planned tile became invalid during materialization")
    gray = np.clip(np.rint(intensity), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _to_mask_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def _pair_id(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _write_tiles(
    staging_root: Path,
    stack: Sigma0Stack,
    assigned: list[AssignedPair],
    water_masks: dict[str, np.ndarray],
    products,
    args: argparse.Namespace,
):
    by_tile = defaultdict(list)
    for item in assigned:
        by_tile[item.tile].append(item)

    records = []
    with tqdm(
        total=len(assigned) * len(OUTPUT_SUBDIRS),
        unit="file",
        desc="Rendering Kulsary pairs",
    ) as progress:
        for tile in sorted(by_tile):
            window = Window(
                tile.col * PATCH_SIZE,
                tile.row * PATCH_SIZE,
                PATCH_SIZE,
                PATCH_SIZE,
            )
            arrays, valid = stack.read(window)
            if not bool(valid.all()):
                raise RuntimeError(
                    f"planned tile r{tile.row} c{tile.col} is no longer valid"
                )
            rgb = {
                role: _to_rgb(
                    arrays[role],
                    valid,
                    args.db_min,
                    args.db_max,
                )
                for role in ROLE_DATES
            }

            for item in sorted(
                by_tile[tile],
                key=lambda value: value.variant.name,
            ):
                variant = item.variant
                tile_slice = _tile_slice(tile)
                water_a = water_masks[variant.a_role][tile_slice]
                water_b = water_masks[variant.b_role][tile_slice]
                gt = compose_flood_mask(
                    water_masks["peak"][tile_slice],
                    water_masks[variant.gt_baseline_role][tile_slice],
                )
                split_root = staging_root / item.output_split

                Image.fromarray(rgb[variant.a_role]).save(
                    split_root / "A" / item.filename,
                    format="PNG",
                )
                progress.update(1)
                Image.fromarray(rgb[variant.b_role]).save(
                    split_root / "B" / item.filename,
                    format="PNG",
                )
                progress.update(1)
                Image.fromarray(_to_mask_u8(gt), mode="L").save(
                    split_root / "GT" / item.filename,
                    format="PNG",
                )
                progress.update(1)
                Image.fromarray(_to_mask_u8(water_a), mode="L").save(
                    split_root / "WATER_GT_A" / item.filename,
                    format="PNG",
                )
                progress.update(1)
                Image.fromarray(_to_mask_u8(water_b), mode="L").save(
                    split_root / "WATER_GT_B" / item.filename,
                    format="PNG",
                )
                progress.update(1)

                positive = int(gt.sum())
                block_row, block_col = spatial_block_key(
                    tile,
                    args.block_tiles,
                )
                records.append(
                    {
                        "split": item.output_split,
                        "filename": item.filename,
                        "pair_id": _pair_id(item.filename),
                        "variant": variant.name,
                        "chronological": str(variant.chronological).lower(),
                        "tile_row": tile.row,
                        "tile_col": tile.col,
                        "block_row": block_row,
                        "block_col": block_col,
                        "a_role": variant.a_role,
                        "b_role": variant.b_role,
                        "a_date": ROLE_DATES[variant.a_role].isoformat(),
                        "b_date": ROLE_DATES[variant.b_role].isoformat(),
                        "a_safe_id": products[variant.a_role].identifier,
                        "b_safe_id": products[variant.b_role].identifier,
                        "gt_formula": variant.gt_formula,
                        "water_gt_a_pixels": int(water_a.sum()),
                        "water_gt_b_pixels": int(water_b.sum()),
                        "water_gt_formula": WATER_GT_FORMULA,
                        "gt_positive_pixels": positive,
                        "gt_fraction": positive / float(PATCH_SIZE * PATCH_SIZE),
                        "valid_fraction": 1.0,
                    }
                )
    return records


def _split_counts(assigned) -> dict[str, int]:
    counts = {split_name: 0 for split_name in OUTPUT_SPLITS}
    for item in assigned:
        counts[item.output_split] += 1
    return counts


def _affine_values(transform) -> list[float]:
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def _build_qc_report(
    mask_refs,
    source_gt_counts,
    stack,
    mask_coverage,
    assigned,
    records,
    skips,
    cache_before,
    cache_after,
    args,
):
    split_variant_counts = {
        split_name: {
            variant.name: sum(
                1
                for item in assigned
                if item.output_split == split_name
                and item.variant.name == variant.name
            )
            for variant in PAIR_VARIANTS
        }
        for split_name in OUTPUT_SPLITS
    }
    gt_output = {}
    for variant in PAIR_VARIANTS:
        variant_records = [
            record for record in records if record["variant"] == variant.name
        ]
        positives = sum(
            int(record["gt_positive_pixels"]) for record in variant_records
        )
        pixels = len(variant_records) * PATCH_SIZE * PATCH_SIZE
        gt_output[variant.name] = {
            "positive_pixels": positives,
            "positive_fraction": positives / pixels if pixels else 0.0,
            "all_negative_tiles": sum(
                int(record["gt_positive_pixels"]) == 0
                for record in variant_records
            ),
        }

    water_supervised_counts = _split_counts(assigned)
    water_pixels_per_split = {
        split_name: {
            "water_gt_a": sum(
                int(record["water_gt_a_pixels"])
                for record in records
                if record["split"] == split_name
            ),
            "water_gt_b": sum(
                int(record["water_gt_b_pixels"])
                for record in records
                if record["split"] == split_name
            ),
        }
        for split_name in OUTPUT_SPLITS
    }

    block_splits = {}
    for item in assigned:
        block = spatial_block_key(item.tile, args.block_tiles)
        previous = block_splits.setdefault(block, item.output_split)
        if previous != item.output_split:
            raise RuntimeError(f"spatial block crosses output splits: {block}")
    blocks_per_split = Counter(block_splits.values())
    block_count = len(block_splits)

    return {
        "converter_version": CONVERTER_VERSION,
        "mask_crs": MASK_CRS,
        "mask_size": list(mask_refs["peak"].size),
        "mask_transform": _affine_values(mask_refs["peak"].transform),
        "mask_files": {
            role: str(ref.png_path) for role, ref in mask_refs.items()
        },
        "source_water_positive_pixels": {
            role: ref.positive_pixels for role, ref in mask_refs.items()
        },
        "source_grid_gt_positive_pixels": source_gt_counts,
        "output_subdirectories": list(OUTPUT_SUBDIRS),
        "gt_semantics": GT_SEMANTICS,
        "water_supervision": "dense",
        "water_gt_formula": WATER_GT_FORMULA,
        "water_supervised_pairs_per_output_split": water_supervised_counts,
        "water_pixels_per_output_split": water_pixels_per_split,
        "mask_encoding": MASK_ENCODING,
        "snap": {
            "requested_target_crs": args.target_crs,
            "actual_crs": stack.grid.crs.to_string(),
            "pixel_spacing": args.pixel_spacing,
            "orbit_type": args.orbit_type,
            "dem_name": args.dem_name,
            "cache_status_before": cache_before,
            "cache_status_after": cache_after,
        },
        "common_grid": {
            "width": stack.grid.width,
            "height": stack.grid.height,
            "transform": _affine_values(stack.grid.transform),
            "bounds": list(stack.grid.bounds),
            "mask_coverage_fraction": float(mask_coverage.mean()),
        },
        "tiles": {
            "candidate_full_tiles": (
                stack.grid.width // PATCH_SIZE
            )
            * (stack.grid.height // PATCH_SIZE),
            "kept_tiles": len(assigned) // len(PAIR_VARIANTS),
            "invalid_tiles": sum(
                record["reason"] == "invalid_common_pixels"
                for record in skips
            ),
            "incomplete_mask_tiles": sum(
                record["reason"] == "incomplete_mask_coverage"
                for record in skips
            ),
            "right_edge_pixels": stack.grid.width % PATCH_SIZE,
            "bottom_edge_pixels": stack.grid.height % PATCH_SIZE,
        },
        "skip_counts": dict(Counter(record["reason"] for record in skips)),
        "samples_per_split_variant": split_variant_counts,
        "output_gt": gt_output,
        "spatial_split": {
            "block_tiles": args.block_tiles,
            "block_count": block_count,
            "blocks_per_split": {
                split_name: blocks_per_split.get(split_name, 0)
                for split_name in OUTPUT_SPLITS
            },
            "actual_block_ratios": {
                split_name: (
                    blocks_per_split.get(split_name, 0) / block_count
                    if block_count
                    else 0.0
                )
                for split_name in OUTPUT_SPLITS
            },
            "requested_ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
        },
        "db_range": [args.db_min, args.db_max],
        "semantic_note": (
            "after_to_peak deliberately uses reversed chronology: "
            "A=2024-04-26 baseline, B=2024-04-14 flood peak"
        ),
    }


def _write_artifacts(
    staging_root: Path,
    source: Path,
    safe_root: Path,
    assigned,
    records,
    skips,
    qc_report,
    args,
) -> None:
    split_order = {name: index for index, name in enumerate(OUTPUT_SPLITS)}
    records = sorted(
        records,
        key=lambda record: (
            split_order[record["split"]],
            record["filename"],
        ),
    )

    with (staging_root / "split_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "filename"))
        for record in records:
            writer.writerow((record["split"], record["filename"]))

    metadata = {
        "source": str(source),
        "safe_root": str(safe_root),
        "seed": args.seed,
        "mode": "render",
        "counts": _split_counts(assigned),
        "split_strategy": SPLIT_STRATEGY.format(
            block_tiles=args.block_tiles
        ),
        "converter": "prepare_kulsary_pairs.py",
        "converter_version": CONVERTER_VERSION,
        "output_subdirectories": list(OUTPUT_SUBDIRS),
        "gt_semantics": GT_SEMANTICS,
        "water_supervision": "dense",
        "water_gt_formula": WATER_GT_FORMULA,
        "mask_encoding": MASK_ENCODING,
        "water_supervised_counts": qc_report[
            "water_supervised_pairs_per_output_split"
        ],
        "water_pixels_per_output_split": qc_report[
            "water_pixels_per_output_split"
        ],
    }
    (staging_root / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    with (staging_root / "pair_manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    with (staging_root / "skipped_records.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in skips:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    (staging_root / "qc_report.json").write_text(
        json.dumps(qc_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_output_mask(path: Path) -> None:
    with Image.open(path) as image:
        array = np.asarray(image)
        if image.mode != "L" or image.size != (PATCH_SIZE, PATCH_SIZE):
            raise RuntimeError(f"invalid output mask contract: {path}")
        if array.dtype != np.uint8:
            raise RuntimeError(f"invalid output mask dtype: {path}")
        if not set(np.unique(array).tolist()).issubset({0, 255}):
            raise RuntimeError(f"non-binary output mask values: {path}")


def _verify_tree(root: Path) -> None:
    for split_name in OUTPUT_SPLITS:
        split_root = root / split_name
        names_by_subdirectory = {}
        for subdirectory in OUTPUT_SUBDIRS:
            directory = split_root / subdirectory
            if not directory.is_dir():
                raise RuntimeError(f"missing output directory: {directory}")
            names_by_subdirectory[subdirectory] = {
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            }

        reference_names = names_by_subdirectory["A"]
        if not reference_names:
            raise RuntimeError(f"output split is empty: {split_name}")
        if any(
            names != reference_names
            for names in names_by_subdirectory.values()
        ):
            counts = {
                name: len(names)
                for name, names in names_by_subdirectory.items()
            }
            raise RuntimeError(
                f"prepared basename mismatch in split {split_name}: {counts}"
            )

        for filename in sorted(reference_names):
            for subdirectory in ("A", "B"):
                path = split_root / subdirectory / filename
                with Image.open(path) as image:
                    array = np.asarray(image)
                    if image.mode != "RGB" or image.size != (PATCH_SIZE, PATCH_SIZE):
                        raise RuntimeError(f"invalid A/B image contract: {path}")
                    if array.dtype != np.uint8:
                        raise RuntimeError(f"invalid A/B dtype: {path}")
                    if not (
                        np.array_equal(array[..., 0], array[..., 1])
                        and np.array_equal(array[..., 0], array[..., 2])
                    ):
                        raise RuntimeError(f"A/B channels differ: {path}")

            for subdirectory in MASK_SUBDIRS:
                _verify_output_mask(split_root / subdirectory / filename)


def _loader_smoke(root: Path) -> None:
    from utils.dataloaders import FloodDetection, test_path, train_path

    data_dir = str(root) + os.sep
    train_full, val_full = train_path(data_dir)
    test_full = test_path(data_dir)
    datasets = {
        "train": FloodDetection(
            train_full,
            aug=False,
            include_water=True,
        ),
        "val": FloodDetection(
            val_full,
            aug=False,
            include_water=True,
        ),
        "test": FloodDetection(
            test_full,
            aug=False,
            include_water=True,
        ),
    }
    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"loader smoke found an empty split: {split_name}")
        image_a, image_b, targets, _ = dataset[0]
        if tuple(image_a.shape) != (3, PATCH_SIZE, PATCH_SIZE):
            raise RuntimeError(f"loader returned an invalid A tensor for {split_name}")
        if tuple(image_b.shape) != (3, PATCH_SIZE, PATCH_SIZE):
            raise RuntimeError(f"loader returned an invalid B tensor for {split_name}")
        if not targets["water_valid"].item():
            raise RuntimeError(f"loader did not expose water labels for {split_name}")
        for target_name in ("change", "water_a", "water_b"):
            if tuple(targets[target_name].shape) != (PATCH_SIZE, PATCH_SIZE):
                raise RuntimeError(
                    f"loader returned an invalid {target_name} tensor for {split_name}"
                )


def _create_sigma0_rasters(
    products,
    gpt: str,
    graph: Path,
    run_dir: Path,
    cache_root: Path | None,
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = {}
    for role in ROLE_DATES:
        output = run_dir / f"{role}_sigma0_vv.tif"
        paths[role] = get_or_create_sigma0(
            gpt,
            graph,
            products[role],
            output,
            args,
            role,
            cache_root,
            args.refresh_snap_cache,
        )
    return paths


def prepare(args: argparse.Namespace) -> dict:
    _validate_args(args)
    (
        source,
        safe_root,
        output,
        work_dir,
        graph,
        cache_root,
    ) = _resolve_paths(args)

    mask_refs = discover_mask_refs(source)
    source_gt_counts = _count_source_gt_pixels(mask_refs)
    products = _discover_products(safe_root)
    _validate_product_geometry(products)

    dry_run_gpt = _find_gpt_for_dry_run(args.gpt)
    dry_run_cache_status = _probe_snap_cache(
        args,
        products,
        graph,
        cache_root,
        dry_run_gpt,
    )
    _print_static_summary(
        source,
        safe_root,
        output,
        mask_refs,
        source_gt_counts,
        products,
        dry_run_cache_status,
    )
    if args.dry_run:
        print("Dry run completed; no files were created.")
        return {
            "dry_run": True,
            "source_gt_positive_pixels": source_gt_counts,
            "cache_status": dry_run_cache_status,
        }

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging = output.with_name(f".{output.name}.partial")
    if staging.exists():
        raise FileExistsError(
            f"staging directory already exists from an earlier run: {staging}"
        )
    _reject_cog_products(products)

    try:
        gpt = resolve_gpt(args.gpt)
    except InferenceError as exc:
        raise InferenceError(
            f"{exc}\nFull Kulsary conversion requires ESA SNAP gpt. "
            "No output staging directory was created."
        ) from exc

    cache_before = _probe_snap_cache(
        args,
        products,
        graph,
        cache_root,
        gpt,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=".kulsary-orbit159-", dir=work_dir))
    try:
        sigma0_paths = _create_sigma0_rasters(
            products,
            gpt,
            graph,
            run_dir,
            cache_root,
            args,
        )
        cache_after = _probe_snap_cache(
            args,
            products,
            graph,
            cache_root,
            gpt,
        )
        with Sigma0Stack(sigma0_paths, mask_refs["peak"]) as stack:
            water_masks, mask_coverage = _warp_masks(mask_refs, stack.grid)
            kept, assigned, skips = _plan_tiles(stack, mask_coverage, args)
            if not kept:
                raise ValueError("no fully valid 256x256 Kulsary tiles were found")

            if output.exists():
                raise FileExistsError(f"output already exists: {output}")
            if staging.exists():
                raise FileExistsError(
                    f"staging directory appeared during processing: {staging}"
                )

            staging.mkdir(parents=True)
            try:
                _ensure_split_dirs(staging)
                records = _write_tiles(
                    staging,
                    stack,
                    assigned,
                    water_masks,
                    products,
                    args,
                )
                qc_report = _build_qc_report(
                    mask_refs,
                    source_gt_counts,
                    stack,
                    mask_coverage,
                    assigned,
                    records,
                    skips,
                    cache_before,
                    cache_after,
                    args,
                )
                _write_artifacts(
                    staging,
                    source,
                    safe_root,
                    assigned,
                    records,
                    skips,
                    qc_report,
                    args,
                )
                _verify_tree(staging)
                _loader_smoke(staging)
                os.replace(staging, output)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(run_dir, ignore_errors=True)

    counts = _split_counts(assigned)
    print(f"Dataset prepared successfully: {output}")
    for split_name in OUTPUT_SPLITS:
        print(f"  {split_name}: {counts[split_name]} samples")
    return {"output": str(output), "counts": counts}


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        prepare(args)
    except (
        FileNotFoundError,
        FileExistsError,
        InferenceError,
        OSError,
        ValueError,
        RuntimeError,
        rasterio.errors.RasterioError,
    ) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
