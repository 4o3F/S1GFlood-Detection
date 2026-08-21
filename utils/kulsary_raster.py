"""Shared Kulsary raster grid, validity planning, and Sigma0 access."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds
import rasterio

from utils.kulsary_temporal import (
    MASK_CRS,
    ROLE_DATES,
    MaskRef,
    TileKey,
    iter_full_windows,
    load_binary_water_mask,
)


PATCH_SIZE = 256


def sampled_file_fingerprint(path: Path, sample_size: int = 1024 * 1024):
    source_path = Path(path)
    size = source_path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    offsets = sorted({
        0,
        max(0, (size - sample_size) // 2),
        max(0, size - sample_size),
    })
    with source_path.open("rb") as source:
        for offset in offsets:
            source.seek(offset)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(source.read(sample_size))
    return {"size": int(size), "sampled_sha256": digest.hexdigest()}


@dataclass(frozen=True)
class CommonGrid:
    crs: object
    transform: object
    width: int
    height: int
    peak_window: Window
    bounds: tuple[float, float, float, float]


def validate_source(dataset, role: str) -> None:
    """Reject a Sigma0 raster that cannot be used on the common grid."""
    if dataset.count != 1:
        raise ValueError(
            f"expected one Sigma0_VV band for {role}, found {dataset.count}"
        )
    if dataset.crs is None:
        raise ValueError(f"Sigma0 raster has no CRS for {role}")
    if dataset.width <= 0 or dataset.height <= 0:
        raise ValueError(f"Sigma0 raster is empty for {role}")
    if abs(dataset.transform.a) <= 0 or abs(dataset.transform.e) <= 0:
        raise ValueError(f"Sigma0 raster has an invalid transform for {role}")


def build_common_grid(datasets: dict, mask_ref: MaskRef) -> CommonGrid:
    """Clip the peak-date grid to the three-role and water-mask overlap."""
    peak = datasets["peak"]
    left, bottom, right, top = peak.bounds

    for role in ("before", "after"):
        dataset = datasets[role]
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
            role: tuple(dataset.bounds) for role, dataset in datasets.items()
        }
        raise ValueError(
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
        raise ValueError("the common mask-clipped peak-grid window is empty")

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


def tile_window(tile: TileKey, patch_size: int = PATCH_SIZE) -> Window:
    return Window(
        tile.col * patch_size,
        tile.row * patch_size,
        patch_size,
        patch_size,
    )


def tile_slice(tile: TileKey, patch_size: int = PATCH_SIZE):
    row_start = tile.row * patch_size
    col_start = tile.col * patch_size
    return np.s_[
        row_start : row_start + patch_size,
        col_start : col_start + patch_size,
    ]


def reproject_mask(source: np.ndarray, ref: MaskRef, grid: CommonGrid) -> np.ndarray:
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


def warp_masks(mask_refs: dict[str, MaskRef], grid: CommonGrid):
    """Warp role water masks onto the common grid and return coverage."""
    peak_ref = mask_refs["peak"]
    coverage_source = np.ones(
        (peak_ref.size[1], peak_ref.size[0]),
        dtype=np.uint8,
    )
    coverage = reproject_mask(coverage_source, peak_ref, grid)
    water_masks = {
        role: reproject_mask(
            load_binary_water_mask(mask_refs[role].png_path),
            mask_refs[role],
            grid,
        )
        for role in ROLE_DATES
    }
    return water_masks, coverage


def linear_sigma0_to_clipped_db(
    sigma0: np.ndarray,
    valid: np.ndarray,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    """Convert linear Sigma0 to float32 dB clipped to ``[db_min, db_max]``."""
    if not math.isfinite(db_min) or not math.isfinite(db_max):
        raise ValueError("db_min and db_max must be finite")
    if db_min >= db_max:
        raise ValueError("db_min must be smaller than db_max")

    sigma = np.asarray(sigma0, dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool)
    if sigma.shape != valid_mask.shape:
        raise ValueError(
            f"sigma0 shape {sigma.shape} does not match valid shape {valid_mask.shape}"
        )

    final_valid = valid_mask & np.isfinite(sigma) & (sigma > 0)
    if not bool(final_valid.all()):
        raise ValueError(
            "clipped-dB conversion requires a fully valid positive sigma0 tile"
        )

    db = (np.float32(10.0) * np.log10(sigma)).astype(np.float32)
    return np.clip(db, np.float32(db_min), np.float32(db_max)).astype(
        np.float32,
        copy=False,
    )


class Sigma0Stack:
    """Read three Sigma0 rasters on a mask-clipped peak-date grid."""

    def __init__(
        self,
        paths: dict[str, Path],
        mask_ref: MaskRef | None = None,
        grid: CommonGrid | None = None,
    ):
        if (mask_ref is None) == (grid is None):
            raise ValueError("exactly one of mask_ref or grid is required")
        missing = [role for role in ROLE_DATES if role not in paths]
        if missing:
            raise ValueError(
                "missing Sigma0 path for role(s): " + ", ".join(missing)
            )

        self.datasets = {}
        self.vrts = {}
        try:
            self.datasets = {
                role: rasterio.open(paths[role]) for role in ROLE_DATES
            }
            for role, dataset in self.datasets.items():
                validate_source(dataset, role)
            self.grid = (
                build_common_grid(self.datasets, mask_ref)
                if mask_ref is not None
                else grid
            )
            self._validate_grid(self.grid)
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

    def _validate_grid(self, grid: CommonGrid) -> None:
        peak = self.datasets["peak"]
        if grid.width <= 0 or grid.height <= 0:
            raise ValueError("common grid is empty")
        if int(grid.peak_window.width) != grid.width:
            raise ValueError("common grid width does not match peak_window")
        if int(grid.peak_window.height) != grid.height:
            raise ValueError("common grid height does not match peak_window")
        col_stop = grid.peak_window.col_off + grid.peak_window.width
        row_stop = grid.peak_window.row_off + grid.peak_window.height
        if grid.peak_window.col_off < 0 or grid.peak_window.row_off < 0:
            raise ValueError("provided grid does not fit the peak raster")
        if col_stop > peak.width or row_stop > peak.height:
            raise ValueError("provided grid does not fit the peak raster")
        if peak.crs != grid.crs:
            raise ValueError("provided grid CRS does not match the peak raster")
        expected_transform = peak.window_transform(grid.peak_window)
        if not grid.transform.almost_equals(expected_transform, precision=1e-9):
            raise ValueError("provided grid transform does not match the peak raster")
        expected_bounds = window_bounds(grid.peak_window, peak.transform)
        if not np.allclose(
            grid.bounds,
            expected_bounds,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("provided grid bounds do not match the peak raster")

    def read_role(
        self,
        role: str,
        window: Window,
    ) -> tuple[np.ndarray, np.ndarray]:
        if role not in ROLE_DATES:
            raise ValueError(f"unknown Sigma0 role: {role}")
        if role == "peak":
            peak_window = Window(
                self.grid.peak_window.col_off + window.col_off,
                self.grid.peak_window.row_off + window.row_off,
                window.width,
                window.height,
            )
            array = self.datasets["peak"].read(
                1,
                window=peak_window,
                out_dtype="float32",
            )
            mask = self.datasets["peak"].read_masks(1, window=peak_window) > 0
        else:
            array = self.vrts[role].read(
                1,
                window=window,
                out_dtype="float32",
            )
            mask = self.vrts[role].read_masks(1, window=window) > 0
        valid = mask & np.isfinite(array) & (array > 0)
        return array, valid

    def read(self, window: Window) -> tuple[dict[str, np.ndarray], np.ndarray]:
        arrays = {}
        valid = None
        for role in ROLE_DATES:
            array, role_valid = self.read_role(role, window)
            arrays[role] = array
            valid = role_valid if valid is None else (valid & role_valid)
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


def plan_valid_tiles(
    stack: Sigma0Stack,
    mask_coverage: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> tuple[list[TileKey], list[dict]]:
    """Return fully valid 256 tiles and converter-compatible skip records."""
    if mask_coverage.shape != (stack.grid.height, stack.grid.width):
        raise ValueError("mask coverage does not match the common grid")

    kept = []
    skips = []
    width_remainder = stack.grid.width % patch_size
    height_remainder = stack.grid.height % patch_size
    if width_remainder:
        skips.append(
            {"reason": "incomplete_right_edge", "width_pixels": width_remainder}
        )
    if height_remainder:
        skips.append(
            {
                "reason": "incomplete_bottom_edge",
                "height_pixels": height_remainder,
            }
        )

    for tile, window in iter_full_windows(
        stack.grid.width,
        stack.grid.height,
        patch_size,
    ):
        coverage = mask_coverage[tile_slice(tile, patch_size)]
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

    return kept, skips


class LazySigma0Stack:
    """Open Sigma0 rasters per process from serialized paths and grid."""

    def __init__(self, paths: dict[str, Path], grid: CommonGrid):
        missing = [role for role in ROLE_DATES if role not in paths]
        if missing:
            raise ValueError(
                "missing Sigma0 path for role(s): " + ", ".join(missing)
            )
        self.paths = {role: Path(paths[role]) for role in ROLE_DATES}
        self.grid = grid
        self._stack: Sigma0Stack | None = None
        self._pid: int | None = None

    def _ensure_open(self) -> Sigma0Stack:
        pid = os.getpid()
        if self._pid != pid:
            # Drop inherited handles after fork; do not close them here.
            self._stack = None
            self._pid = pid
        if self._stack is None:
            self._stack = Sigma0Stack(self.paths, grid=self.grid)
            self._pid = pid
        return self._stack

    def read_role(
        self,
        role: str,
        window: Window,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._ensure_open().read_role(role, window)

    def read(self, window: Window) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return self._ensure_open().read(window)

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._pid = None

    def __getstate__(self):
        self.close()
        return {"paths": dict(self.paths), "grid": self.grid}

    def __setstate__(self, state):
        self.paths = {role: Path(path) for role, path in state["paths"].items()}
        self.grid = state["grid"]
        self._stack = None
        self._pid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
