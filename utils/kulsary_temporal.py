"""Domain logic for preparing Kulsary Orbit 159 temporal pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image
from rasterio.transform import Affine
from rasterio.windows import Window


ROLE_DATES = {
    "before": date(2024, 4, 2),
    "peak": date(2024, 4, 14),
    "after": date(2024, 4, 26),
}
MASK_PATTERNS = {
    "before": "*water_before_20240402*.png",
    "peak": "*water_during_20240414*.png",
    "after": "*water_after_20240426*.png",
}
OUTPUT_SPLITS = ("train", "val", "test")
MASK_CRS = "EPSG:4326"
_ALLOWED_MASK_VALUES = {0, 1, 255}


@dataclass(frozen=True)
class PairVariant:
    name: str
    a_role: str
    b_role: str
    gt_baseline_role: str
    gt_formula: str
    chronological: bool


PAIR_VARIANTS = (
    PairVariant(
        name="before_to_peak",
        a_role="before",
        b_role="peak",
        gt_baseline_role="before",
        gt_formula="peak_water AND NOT before_water",
        chronological=True,
    ),
    PairVariant(
        name="after_to_peak",
        a_role="after",
        b_role="peak",
        gt_baseline_role="after",
        gt_formula="peak_water AND NOT after_water",
        chronological=False,
    ),
)


@dataclass(frozen=True)
class WorldFile:
    a: float
    d: float
    b: float
    e: float
    c: float
    f: float


@dataclass(frozen=True)
class MaskRef:
    role: str
    png_path: Path
    world_path: Path
    size: tuple[int, int]
    transform: Affine
    positive_pixels: int


@dataclass(frozen=True, order=True)
class TileKey:
    row: int
    col: int


@dataclass(frozen=True)
class AssignedPair:
    tile: TileKey
    output_split: str
    variant: PairVariant
    filename: str


def parse_world_file(path: Path) -> WorldFile:
    """Parse an ESRI world file whose C/F values are pixel-center coordinates."""
    try:
        values = [
            float(line.strip())
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"world file contains a non-numeric value: {path}") from exc

    if len(values) != 6:
        raise ValueError(f"world file must contain exactly six values: {path}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"world file contains a non-finite value: {path}")

    world = WorldFile(*values)
    if world.a * world.e - world.b * world.d == 0:
        raise ValueError(f"world file transform is singular: {path}")
    return world


def world_file_to_affine(world: WorldFile) -> Affine:
    """Convert the ESRI pixel-center convention to a GDAL outer-corner affine."""
    return Affine(
        world.a,
        world.b,
        world.c - world.a / 2.0 - world.b / 2.0,
        world.d,
        world.e,
        world.f - world.d / 2.0 - world.e / 2.0,
    )


def load_binary_water_mask(path: Path) -> np.ndarray:
    """Load a binary water mask as a two-dimensional boolean array."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image)
    except OSError as exc:
        raise ValueError(f"could not read water mask {path}: {exc}") from exc

    if array.ndim == 3:
        if array.shape[2] != 3:
            raise ValueError(f"unsupported water-mask shape {array.shape}: {path}")
        if not (
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])
        ):
            raise ValueError(f"water-mask RGB channels differ: {path}")
        array = array[..., 0]
    elif array.ndim != 2:
        raise ValueError(f"unsupported water-mask shape {array.shape}: {path}")

    values = set(np.unique(array).tolist())
    if not values.issubset(_ALLOWED_MASK_VALUES):
        raise ValueError(
            f"water mask must be binary {{0,1,255}}, found {sorted(values)}: {path}"
        )
    return np.asarray(array > 0, dtype=bool)


def compose_flood_mask(
    peak_water: np.ndarray,
    baseline_water: np.ndarray,
) -> np.ndarray:
    if peak_water.shape != baseline_water.shape:
        raise ValueError(
            f"water-mask shapes differ: {peak_water.shape} vs {baseline_water.shape}"
        )
    return np.asarray(peak_water & ~baseline_water, dtype=bool)


def discover_mask_refs(source: Path) -> dict[str, MaskRef]:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"mask source directory is missing: {source}")

    refs: dict[str, MaskRef] = {}
    for role, pattern in MASK_PATTERNS.items():
        matches = sorted(
            path
            for path in source.rglob(pattern)
            if path.is_file() and not path.name.startswith("_")
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {role} mask matching {pattern} under "
                f"{source}, found {len(matches)}"
            )

        png_path = matches[0]
        world_path = png_path.with_suffix(".pgw")
        if not world_path.is_file():
            raise FileNotFoundError(
                f"world file is missing for {role} mask: {world_path}"
            )

        mask = load_binary_water_mask(png_path)
        refs[role] = MaskRef(
            role=role,
            png_path=png_path,
            world_path=world_path,
            size=(int(mask.shape[1]), int(mask.shape[0])),
            transform=world_file_to_affine(parse_world_file(world_path)),
            positive_pixels=int(mask.sum()),
        )

    reference = refs["peak"]
    for role, ref in refs.items():
        if ref.size != reference.size:
            raise ValueError(
                f"mask size differs for {role}: {ref.size} vs {reference.size}"
            )
        if not ref.transform.almost_equals(
            reference.transform,
            precision=1e-12,
        ):
            raise ValueError(f"mask transform differs for {role}")
    return refs


def iter_full_windows(
    width: int,
    height: int,
    patch_size: int = 256,
) -> Iterator[tuple[TileKey, Window]]:
    if width < 0 or height < 0:
        raise ValueError("raster dimensions must be non-negative")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    for tile_row in range(height // patch_size):
        for tile_col in range(width // patch_size):
            yield (
                TileKey(tile_row, tile_col),
                Window(
                    tile_col * patch_size,
                    tile_row * patch_size,
                    patch_size,
                    patch_size,
                ),
            )


def spatial_block_key(
    tile: TileKey,
    block_tiles: int = 2,
) -> tuple[int, int]:
    if block_tiles <= 0:
        raise ValueError("block_tiles must be positive")
    return tile.row // block_tiles, tile.col // block_tiles


def _block_score(block: tuple[int, int], seed: int) -> int:
    block_row, block_col = block
    payload = f"{seed}:kulsary:{block_row}:{block_col}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def assign_spatial_blocks(
    tiles: Iterable[TileKey],
    *,
    block_tiles: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[TileKey, str]:
    unique_tiles = sorted(set(tiles))
    if not unique_tiles:
        raise ValueError("no valid tiles are available for splitting")
    if block_tiles <= 0:
        raise ValueError("block_tiles must be positive")

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("split ratios must be finite and positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/val/test ratios must sum to 1")

    blocks = {spatial_block_key(tile, block_tiles) for tile in unique_tiles}
    ordered_blocks = sorted(
        blocks,
        key=lambda block: (_block_score(block, seed), block),
    )
    block_count = len(ordered_blocks)
    if block_count < 3:
        raise ValueError(
            f"at least three spatial super-blocks are required, found {block_count}"
        )

    n_val = max(1, round(val_ratio * block_count))
    n_test = max(1, round(test_ratio * block_count))
    n_train = block_count - n_val - n_test
    if n_train < 1:
        raise ValueError(
            "split ratios leave no training super-block; increase the valid area "
            "or reduce val/test ratios"
        )

    val_blocks = set(ordered_blocks[:n_val])
    test_blocks = set(ordered_blocks[n_val : n_val + n_test])
    split_by_block = {
        block: (
            "val"
            if block in val_blocks
            else "test"
            if block in test_blocks
            else "train"
        )
        for block in ordered_blocks
    }
    return {
        tile: split_by_block[spatial_block_key(tile, block_tiles)]
        for tile in unique_tiles
    }


def build_filename(tile: TileKey, variant: PairVariant) -> str:
    return f"kulsary_r{tile.row:04d}_c{tile.col:04d}_{variant.name}.png"


def expand_pair_variants(
    tiles: Iterable[TileKey],
    split_by_tile: dict[TileKey, str],
) -> list[AssignedPair]:
    assigned = []
    for tile in sorted(set(tiles)):
        try:
            output_split = split_by_tile[tile]
        except KeyError as exc:
            raise ValueError(f"tile has no output split: {tile}") from exc
        if output_split not in OUTPUT_SPLITS:
            raise ValueError(f"unsupported output split: {output_split}")

        for variant in PAIR_VARIANTS:
            assigned.append(
                AssignedPair(
                    tile=tile,
                    output_split=output_split,
                    variant=variant,
                    filename=build_filename(tile, variant),
                )
            )
    return finalize_assignments(assigned)


def finalize_assignments(assigned: Iterable[AssignedPair]) -> list[AssignedPair]:
    split_order = {name: index for index, name in enumerate(OUTPUT_SPLITS)}
    ordered = sorted(
        assigned,
        key=lambda item: (split_order[item.output_split], item.filename),
    )
    seen = set()
    for item in ordered:
        if item.filename in seen:
            raise ValueError(f"filename collision: {item.filename}")
        seen.add(item.filename)
    return ordered
