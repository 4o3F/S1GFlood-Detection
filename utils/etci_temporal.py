"""ETCI-2021 bi-temporal pairing domain logic.

Pure functions for parsing ETCI scene/tile names, indexing labeled splits,
normalizing binary masks, running lightweight VV quality control, building
cross-date temporal pairs, and assigning output splits. No argparse, no
PyTorch, no network: this module is intentionally easy to unit-test and to
reuse from the offline converter and any future in-memory loader.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

SCENE_RE = re.compile(r"^(?P<region>.+)_(?P<timestamp>\d{8}t\d{6})$", re.IGNORECASE)
VV_RE = re.compile(
    r"^(?P<scene>.+)_x-(?P<x>-?\d+)_y-(?P<y>-?\d+)_vv\.png$", re.IGNORECASE
)
FLOOD_VH_RE = re.compile(
    r"^(?P<scene>.+)_x-(?P<x>-?\d+)_y-(?P<y>-?\d+)_vh\.png$", re.IGNORECASE
)
FLOOD_RE = re.compile(
    r"^(?P<scene>.+)_x-(?P<x>-?\d+)_y-(?P<y>-?\d+)(?:_vv)?\.png$", re.IGNORECASE
)
TIMESTAMP_FORMAT = "%Y%m%dt%H%M%S"
_ALLOWED_MASK_VALUES = {0, 1, 255}


@dataclass(frozen=True)
class TileObservation:
    source_split: str
    region: str
    scene: str
    timestamp: datetime
    x: int
    y: int
    vv_path: Path
    flood_path: Path
    water_path: Path


@dataclass(frozen=True)
class TemporalPair:
    source_split: str
    region: str
    x: int
    y: int
    pre_scene: str
    post_scene: str
    pre_timestamp: datetime
    post_timestamp: datetime
    pre_vv_path: Path
    post_vv_path: Path
    pre_flood_path: Path
    post_flood_path: Path
    pre_water_path: Path
    post_water_path: Path
    pre_flood_pixels: int
    post_flood_pixels: int
    pre_water_body_pixels: int
    post_water_body_pixels: int
    water_gt_a_pixels: int
    water_gt_b_pixels: int
    gap_days: int
    policy: str


@dataclass(frozen=True)
class AssignedPair:
    pair: TemporalPair
    output_split: str
    filename: str


@dataclass
class IndexStats:
    observations: int = 0
    scenes: int = 0
    vv_files: int = 0
    flood_files: int = 0
    water_files: int = 0


def parse_scene_name(name: str) -> Optional[Tuple[str, datetime]]:
    match = SCENE_RE.match(name)
    if not match:
        return None
    try:
        timestamp = datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)
    except ValueError:
        return None
    return match.group("region"), timestamp


def parse_vv_filename(name: str) -> Optional[Tuple[str, int, int]]:
    match = VV_RE.match(name)
    if not match:
        return None
    return match.group("scene"), int(match.group("x")), int(match.group("y"))


def _parse_label_filename(name: str, label_name: str) -> Optional[Tuple[str, int, int]]:
    if FLOOD_VH_RE.match(name):
        raise ValueError(
            f"vh-polarized {label_name} is not valid VV ground truth: {name}"
        )
    match = FLOOD_RE.match(name)
    if not match:
        return None
    return match.group("scene"), int(match.group("x")), int(match.group("y"))


def parse_flood_filename(name: str) -> Optional[Tuple[str, int, int]]:
    return _parse_label_filename(name, "flood label")


def parse_water_filename(name: str) -> Optional[Tuple[str, int, int]]:
    return _parse_label_filename(name, "water-body label")


def _scene_token(scene: str) -> str:
    return scene.rsplit("_", 1)[-1]


def build_filename(pair: TemporalPair) -> str:
    pre_token = _scene_token(pair.pre_scene)
    post_token = _scene_token(pair.post_scene)
    return f"etci_{pair.region}_x-{pair.x}_y-{pair.y}_{pre_token}_{post_token}_vv.png"


def load_binary_mask(path: Path, label_name: str = "mask") -> Image.Image:
    """Return a canonical mode-L binary mask with values in {0, 255}."""
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] != 3:
            raise ValueError(
                f"unsupported {label_name} channel count: {array.shape}"
            )
        if not (
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])
        ):
            raise ValueError(f"{label_name} channels differ")
        gray = array[..., 0]
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError(f"unsupported {label_name} shape: {array.shape}")

    if not set(np.unique(gray).tolist()).issubset(_ALLOWED_MASK_VALUES):
        raise ValueError(f"non-binary {label_name} values")

    binary = np.where(gray > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def load_binary_flood_mask(path: Path) -> Image.Image:
    return load_binary_mask(path, "flood mask")


def load_binary_water_mask(path: Path) -> Image.Image:
    return load_binary_mask(path, "water-body mask")


def compose_full_water_mask(water_path: Path, flood_path: Path) -> Image.Image:
    water = load_binary_water_mask(water_path)
    flood = load_binary_flood_mask(flood_path)
    if water.size != flood.size:
        raise ValueError(
            f"water/flood mask size mismatch: water={water.size}, flood={flood.size}"
        )
    combined = np.asarray(water) > 0
    combined |= np.asarray(flood) > 0
    return Image.fromarray((combined.astype(np.uint8) * 255), mode="L")


@lru_cache(maxsize=None)
def count_flood_pixels(path: Path) -> int:
    mask = load_binary_flood_mask(path)
    return int((np.asarray(mask) > 0).sum())


@lru_cache(maxsize=None)
def count_water_body_pixels(path: Path) -> int:
    mask = load_binary_water_mask(path)
    return int((np.asarray(mask) > 0).sum())


@lru_cache(maxsize=None)
def count_full_water_pixels(water_path: Path, flood_path: Path) -> int:
    mask = compose_full_water_mask(water_path, flood_path)
    return int((np.asarray(mask) > 0).sum())


def clear_flood_pixel_cache() -> None:
    count_flood_pixels.cache_clear()
    count_water_body_pixels.cache_clear()
    count_full_water_pixels.cache_clear()


def vv_passes_qc(
    path: Path, min_vv_bytes: int, max_saturated_fraction: float = 0.999
) -> Tuple[bool, str]:
    try:
        size = path.stat().st_size
    except OSError:
        return False, "vv_stat_failed"
    if size < min_vv_bytes:
        return False, "vv_too_small"
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        return False, "vv_unreadable"

    gray = array[..., 0]
    if float(gray.std()) == 0.0:
        return False, "vv_uniform"

    total = gray.size
    if total:
        saturated = int(((gray == 0) | (gray == 255)).sum())
        if saturated / total > max_saturated_fraction:
            return False, "vv_saturated"
    return True, "ok"


def _shapes_match(*paths: Path) -> bool:
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            sizes.append(image.size)
    return len(set(sizes)) == 1


def _skip(reason: str, observation: TileObservation, detail: str = "") -> dict:
    return {
        "reason": reason,
        "source_split": observation.source_split,
        "region": observation.region,
        "scene": observation.scene,
        "x": observation.x,
        "y": observation.y,
        "detail": detail,
    }


def build_temporal_pairs(
    observations,
    policy: str = "nearest-flood-free",
    min_vv_bytes: int = 2048,
    max_gap_days: Optional[int] = None,
    keep_negative_post: bool = True,
    max_saturated_fraction: float = 0.999,
):
    if policy not in ("nearest-flood-free", "adjacent-any"):
        raise ValueError(f"unsupported pair policy: {policy}")

    groups: dict = defaultdict(list)
    for observation in observations:
        groups[
            (observation.source_split, observation.region, observation.x, observation.y)
        ].append(observation)

    pairs = []
    skips = []
    for key in sorted(groups.keys()):
        timeline = sorted(groups[key], key=lambda item: item.timestamp)
        for post_index in range(1, len(timeline)):
            post = timeline[post_index]
            post_ok, post_reason = vv_passes_qc(
                post.vv_path, min_vv_bytes, max_saturated_fraction
            )
            if not post_ok:
                skips.append(_skip("vv_qc_post", post, post_reason))
                continue

            try:
                post_flood_pixels = count_flood_pixels(post.flood_path)
            except (OSError, ValueError) as exc:
                # A single corrupt post-event mask must not abort the whole job.
                skips.append(_skip("corrupt_flood_label", post, str(exc)))
                continue
            try:
                post_water_body_pixels = count_water_body_pixels(post.water_path)
            except (OSError, ValueError) as exc:
                skips.append(_skip("corrupt_water_label", post, str(exc)))
                continue
            try:
                post_shapes_match = _shapes_match(
                    post.vv_path,
                    post.flood_path,
                    post.water_path,
                )
            except (OSError, ValueError) as exc:
                skips.append(_skip("label_shape_check_failed", post, str(exc)))
                continue
            if not post_shapes_match:
                skips.append(_skip("label_shape_mismatch", post))
                continue
            try:
                post_full_water_pixels = count_full_water_pixels(
                    post.water_path,
                    post.flood_path,
                )
            except (OSError, ValueError) as exc:
                skips.append(_skip("corrupt_water_label", post, str(exc)))
                continue
            if post_flood_pixels == 0 and not keep_negative_post:
                skips.append(_skip("negative_post", post))
                continue

            chosen: Optional[TileObservation] = None
            if policy == "nearest-flood-free":
                for pre in reversed(timeline[:post_index]):
                    gap = (post.timestamp - pre.timestamp).days
                    if max_gap_days is not None and gap > max_gap_days:
                        continue
                    # Corrupt pre masks / unreadable shapes skip this candidate,
                    # not the entire conversion.
                    try:
                        pre_ok, pre_reason = vv_passes_qc(
                            pre.vv_path, min_vv_bytes, max_saturated_fraction
                        )
                        if not pre_ok:
                            continue
                        if count_flood_pixels(pre.flood_path) != 0:
                            continue
                        count_water_body_pixels(pre.water_path)
                        if not _shapes_match(
                            pre.vv_path,
                            pre.flood_path,
                            pre.water_path,
                            post.vv_path,
                            post.flood_path,
                            post.water_path,
                        ):
                            continue
                        count_full_water_pixels(pre.water_path, pre.flood_path)
                    except (OSError, ValueError):
                        continue
                    chosen = pre
                    break
                applied_policy = "nearest-flood-free"
            else:
                pre = timeline[post_index - 1]
                gap = (post.timestamp - pre.timestamp).days
                if max_gap_days is None or gap <= max_gap_days:
                    try:
                        pre_ok, _ = vv_passes_qc(
                            pre.vv_path, min_vv_bytes, max_saturated_fraction
                        )
                        if pre_ok:
                            count_flood_pixels(pre.flood_path)
                            count_water_body_pixels(pre.water_path)
                        shapes_ok = pre_ok and _shapes_match(
                            pre.vv_path,
                            pre.flood_path,
                            pre.water_path,
                            post.vv_path,
                            post.flood_path,
                            post.water_path,
                        )
                        if shapes_ok:
                            count_full_water_pixels(pre.water_path, pre.flood_path)
                    except (OSError, ValueError):
                        shapes_ok = False
                    if shapes_ok:
                        chosen = pre
                applied_policy = "adjacent-any"

            if chosen is None:
                skips.append(_skip("no_valid_pre", post))
                continue

            pairs.append(
                TemporalPair(
                    source_split=post.source_split,
                    region=post.region,
                    x=post.x,
                    y=post.y,
                    pre_scene=chosen.scene,
                    post_scene=post.scene,
                    pre_timestamp=chosen.timestamp,
                    post_timestamp=post.timestamp,
                    pre_vv_path=chosen.vv_path,
                    post_vv_path=post.vv_path,
                    pre_flood_path=chosen.flood_path,
                    post_flood_path=post.flood_path,
                    pre_water_path=chosen.water_path,
                    post_water_path=post.water_path,
                    pre_flood_pixels=count_flood_pixels(chosen.flood_path),
                    post_flood_pixels=post_flood_pixels,
                    pre_water_body_pixels=count_water_body_pixels(chosen.water_path),
                    post_water_body_pixels=post_water_body_pixels,
                    water_gt_a_pixels=count_full_water_pixels(
                        chosen.water_path,
                        chosen.flood_path,
                    ),
                    water_gt_b_pixels=post_full_water_pixels,
                    gap_days=(post.timestamp - chosen.timestamp).days,
                    policy=applied_policy,
                )
            )

    return pairs, skips


def _group_score(region: str, x: int, y: int, seed: int) -> int:
    payload = f"{seed}:{region}:{x}:{y}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def assign_train_val_groups(pairs, val_ratio: float, seed: int):
    groups: dict = defaultdict(list)
    for pair in pairs:
        groups[(pair.region, pair.x, pair.y)].append(pair)

    keys = sorted(groups.keys(), key=lambda key: (_group_score(*key, seed), key))
    group_count = len(keys)
    if group_count >= 2 and 0.0 < val_ratio < 1.0:
        n_val = round(val_ratio * group_count)
        n_val = min(max(n_val, 1), group_count - 1)
    else:
        n_val = 0
    val_keys = set(keys[:n_val])

    assigned = []
    for key in keys:
        output_split = "val" if key in val_keys else "train"
        for pair in groups[key]:
            assigned.append(
                AssignedPair(
                    pair=pair, output_split=output_split, filename=build_filename(pair)
                )
            )
    return assigned


def assign_all_to_split(pairs, output_split: str):
    return [
        AssignedPair(pair=pair, output_split=output_split, filename=build_filename(pair))
        for pair in pairs
    ]


def finalize_assignments(assigned):
    assigned = sorted(assigned, key=lambda item: (item.output_split, item.filename))
    seen = set()
    for item in assigned:
        if item.filename in seen:
            raise ValueError(f"filename collision: {item.filename}")
        seen.add(item.filename)
    return assigned


def _is_safe_tile_file(
    entry: Path,
    data_root: Path,
    trusted_symlink_root: Optional[Path] = None,
) -> bool:
    """Accept regular files and explicitly trusted Hub-cache symlinks only."""
    if not entry.is_file():
        return False
    if entry.is_symlink() and trusted_symlink_root is None:
        return False

    allowed_root = trusted_symlink_root if entry.is_symlink() else data_root
    try:
        entry.resolve(strict=True).relative_to(allowed_root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def index_labeled_split(
    data_root: Path,
    source_split: str,
    trusted_symlink_root: Optional[Path] = None,
):
    split_dir = data_root / source_split
    observations = []
    skips = []
    stats = IndexStats()

    if not split_dir.is_dir():
        return observations, skips, stats

    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        parsed = parse_scene_name(scene_dir.name)
        if parsed is None:
            skips.append({"reason": "invalid_scene_name", "scene": scene_dir.name})
            continue
        region, timestamp = parsed
        stats.scenes += 1

        vv_dir = scene_dir / "tiles" / "vv"
        flood_dir = scene_dir / "tiles" / "flood_label"
        water_dir = scene_dir / "tiles" / "water_body_label"
        missing_dirs = [
            directory.name
            for directory in (vv_dir, flood_dir, water_dir)
            if not directory.is_dir()
        ]
        if missing_dirs:
            skips.append(
                {
                    "reason": "missing_tile_dirs",
                    "scene": scene_dir.name,
                    "detail": ",".join(missing_dirs),
                }
            )
            continue

        vv_map = {}
        ambiguous_vv_keys = set()
        for entry in sorted(vv_dir.iterdir()):
            if entry.name.startswith(".") or ".ipynb_checkpoints" in entry.parts:
                continue
            if not _is_safe_tile_file(entry, data_root, trusted_symlink_root):
                skips.append(
                    {"reason": "unsafe_tile_file", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            if entry.suffix.lower() != ".png":
                continue
            parsed_vv = parse_vv_filename(entry.name)
            if parsed_vv is None:
                skips.append(
                    {"reason": "unparseable_vv_name", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            file_scene, x, y = parsed_vv
            if file_scene != scene_dir.name:
                skips.append(
                    {"reason": "scene_filename_mismatch", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            key = (x, y)
            stats.vv_files += 1
            if key in ambiguous_vv_keys or key in vv_map:
                vv_map.pop(key, None)
                ambiguous_vv_keys.add(key)
                skips.append(
                    {"reason": "duplicate_vv_key", "scene": scene_dir.name, "x": x, "y": y}
                )
                continue
            vv_map[key] = entry

        flood_map = {}
        ambiguous_flood_keys = set()
        for entry in sorted(flood_dir.iterdir()):
            if entry.name.startswith(".") or ".ipynb_checkpoints" in entry.parts:
                continue
            if not _is_safe_tile_file(entry, data_root, trusted_symlink_root):
                skips.append(
                    {"reason": "unsafe_tile_file", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            if entry.suffix.lower() != ".png":
                continue
            try:
                parsed_flood = parse_flood_filename(entry.name)
            except ValueError:
                skips.append(
                    {"reason": "vh_flood_label", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            if parsed_flood is None:
                skips.append(
                    {"reason": "unparseable_flood_name", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            file_scene, x, y = parsed_flood
            if file_scene != scene_dir.name:
                skips.append(
                    {"reason": "scene_filename_mismatch", "scene": scene_dir.name, "file": entry.name}
                )
                continue
            key = (x, y)
            stats.flood_files += 1
            if key in ambiguous_flood_keys or key in flood_map:
                flood_map.pop(key, None)
                ambiguous_flood_keys.add(key)
                skips.append(
                    {"reason": "duplicate_flood_alias", "scene": scene_dir.name, "x": x, "y": y}
                )
                continue
            flood_map[key] = entry

        water_map = {}
        ambiguous_water_keys = set()
        for entry in sorted(water_dir.iterdir()):
            if entry.name.startswith(".") or ".ipynb_checkpoints" in entry.parts:
                continue
            if not _is_safe_tile_file(entry, data_root, trusted_symlink_root):
                skips.append(
                    {
                        "reason": "unsafe_tile_file",
                        "scene": scene_dir.name,
                        "file": entry.name,
                    }
                )
                continue
            if entry.suffix.lower() != ".png":
                continue
            try:
                parsed_water = parse_water_filename(entry.name)
            except ValueError:
                skips.append(
                    {
                        "reason": "vh_water_label",
                        "scene": scene_dir.name,
                        "file": entry.name,
                    }
                )
                continue
            if parsed_water is None:
                skips.append(
                    {
                        "reason": "unparseable_water_name",
                        "scene": scene_dir.name,
                        "file": entry.name,
                    }
                )
                continue
            file_scene, x, y = parsed_water
            if file_scene != scene_dir.name:
                skips.append(
                    {
                        "reason": "scene_filename_mismatch",
                        "scene": scene_dir.name,
                        "file": entry.name,
                    }
                )
                continue
            key = (x, y)
            stats.water_files += 1
            if key in ambiguous_water_keys or key in water_map:
                water_map.pop(key, None)
                ambiguous_water_keys.add(key)
                skips.append(
                    {
                        "reason": "duplicate_water_alias",
                        "scene": scene_dir.name,
                        "x": x,
                        "y": y,
                    }
                )
                continue
            water_map[key] = entry

        vv_keys = vv_map.keys() | ambiguous_vv_keys
        flood_keys = flood_map.keys() | ambiguous_flood_keys
        water_keys = water_map.keys() | ambiguous_water_keys
        for (x, y) in sorted(vv_keys - flood_keys):
            skips.append(
                {"reason": "vv_without_flood", "scene": scene_dir.name, "x": x, "y": y}
            )
        for (x, y) in sorted(flood_keys - vv_keys):
            skips.append(
                {"reason": "flood_without_vv", "scene": scene_dir.name, "x": x, "y": y}
            )
        for (x, y) in sorted(vv_keys - water_keys):
            skips.append(
                {"reason": "vv_without_water", "scene": scene_dir.name, "x": x, "y": y}
            )
        for (x, y) in sorted(water_keys - vv_keys):
            skips.append(
                {"reason": "water_without_vv", "scene": scene_dir.name, "x": x, "y": y}
            )

        valid_keys = vv_map.keys() & flood_map.keys() & water_map.keys()
        for (x, y) in sorted(valid_keys):
            observations.append(
                TileObservation(
                    source_split=source_split,
                    region=region,
                    scene=scene_dir.name,
                    timestamp=timestamp,
                    x=x,
                    y=y,
                    vv_path=vv_map[(x, y)],
                    flood_path=flood_map[(x, y)],
                    water_path=water_map[(x, y)],
                )
            )

    stats.observations = len(observations)
    return observations, skips, stats


def inspect_test_internal(data_root: Path) -> dict:
    internal = data_root / "test_internal"
    info = {
        "present": False,
        "scenes": 0,
        "vv_files": 0,
        "flood_label_files": 0,
        "water_body_label_files": 0,
        "excluded_reason": None,
    }
    if not internal.is_dir():
        return info

    info["present"] = True
    info["scenes"] = sum(1 for entry in internal.iterdir() if entry.is_dir())
    info["vv_files"] = sum(
        1 for path in internal.rglob("tiles/vv/*.png") if path.is_file()
    )
    info["flood_label_files"] = sum(
        1 for path in internal.rglob("tiles/flood_label/*.png") if path.is_file()
    )
    info["water_body_label_files"] = sum(
        1 for path in internal.rglob("tiles/water_body_label/*.png") if path.is_file()
    )
    if info["flood_label_files"] == 0:
        info["excluded_reason"] = "no_flood_label"
    else:
        # The converter never indexes test_internal; present labels do not
        # imply inclusion, so reporting "kept" would be misleading.
        info["excluded_reason"] = "excluded_by_policy"
    return info
