"""Convert ETCI-2021 into the S1GFloods bi-temporal train/val/test layout.

Standalone offline converter. For each output sample::

    A          = earlier-date VV for the same region and tile coordinate
    B          = later-date VV for the same region and tile coordinate
    GT         = later-date flood_label, canonical mode L {0,255}
    WATER_GT_A = earlier water_body_label OR earlier flood_label
    WATER_GT_B = later water_body_label OR later flood_label

The output directory is a drop-in root for the existing ``utils.dataloaders``
loader; no training or model code is modified.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.etci_temporal import (
    AssignedPair,
    assign_all_to_split,
    assign_train_val_groups,
    build_temporal_pairs,
    compose_full_water_mask,
    finalize_assignments,
    index_labeled_split,
    inspect_test_internal,
    load_binary_flood_mask,
)

DEFAULT_REPO_ID = "blanchon/ETCI-2021-Flood-Detection"
DEFAULT_REVISION = "921e207ea6aa26e9366fd200725a98b0067f9d6b"
CONVERTER_VERSION = "1.1.0"
WATER_GT_FORMULA = "water_body_label OR flood_label, per acquisition date"
OUTPUT_SUBDIRS = ("A", "B", "GT", "WATER_GT_A", "WATER_GT_B")
SPLIT_STRATEGY = (
    "train pairs group-split into train/val by (region,x,y); "
    "source test pairs to output test; test_internal excluded "
    "(never indexed; labels if present are not used)"
)
OUTPUT_SPLITS = ("train", "val", "test")
PAIR_MANIFEST_COLUMNS = (
    "split,filename,pair_id,region,x,y,pre_scene,post_scene,"
    "pre_datetime,post_datetime,gap_days,pre_flood_pixels,post_flood_pixels,"
    "policy,pre_vv,post_vv,pre_flood,post_flood,pre_water,post_water,"
    "pre_water_body_pixels,post_water_body_pixels,water_gt_a_pixels,"
    "water_gt_b_pixels,water_gt_formula"
)

_hardlink_copy_fallback = 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ETCI-2021 multi-date VV scenes into the paired "
            "A/B/GT/WATER_GT_A/WATER_GT_B train/val/test layout."
        )
    )
    parser.add_argument("--source", type=Path, help="local ETCI repository or data root")
    parser.add_argument("--output", type=Path, required=True, help="output dataset root")
    parser.add_argument("--download", action="store_true", help="download from Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help=(
            "Hugging Face endpoint override (e.g. https://hf-mirror.com). "
            "Falls back to the HF_ENDPOINT environment variable. Passed "
            "explicitly to snapshot_download so it does not depend on the "
            "variable being exported into the process."
        ),
    )
    parser.add_argument(
        "--pair-policy",
        choices=("nearest-flood-free", "adjacent-any"),
        default="nearest-flood-free",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--min-vv-bytes", type=int, default=2048)
    parser.add_argument("--max-gap-days", type=int, default=None)
    parser.add_argument("--max-saturated-fraction", type=float, default=0.999)
    parser.add_argument(
        "--keep-negative-post",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.source is None and not args.download:
        raise ValueError("provide either --source or --download")
    if args.source is not None and args.download:
        raise ValueError("use either --source or --download, not both")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be strictly between 0 and 1")
    if args.max_saturated_fraction < 0.0 or args.max_saturated_fraction > 1.0:
        raise ValueError("--max-saturated-fraction must be between 0 and 1")


def find_data_root(source: Path) -> Path:
    source = source.expanduser().resolve()
    if (source / "data" / "train").is_dir():
        return source / "data"
    if (source / "train").is_dir():
        return source
    raise FileNotFoundError(
        f"could not locate ETCI data/train under {source}; "
        "pass the repository root, the data/ directory, or use --download"
    )


def resolve_data_root(args: argparse.Namespace) -> Path:
    if args.download:
        from huggingface_hub import snapshot_download

        # Resolve the endpoint explicitly and pass it through to snapshot_download
        # so routing does not depend on HF_ENDPOINT reaching this process.
        endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT")
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
            print(f"HF endpoint: {endpoint}")
        snapshot_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            allow_patterns=["README.md", "data/train/**", "data/test/**"],
            endpoint=endpoint,
        )
        return find_data_root(Path(snapshot_path))
    return find_data_root(args.source)


def _trusted_download_root(data_root: Path) -> Path | None:
    """Return the repository cache root that owns snapshot/blob symlinks."""
    for parent in data_root.parents:
        if parent.name == "snapshots":
            return parent.parent.resolve()
    return None


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    global _hardlink_copy_fallback
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        link_source = source.resolve(strict=True) if source.is_symlink() else source
        try:
            os.link(link_source, destination)
        except OSError as error:
            # EXDEV = cross-device; EPERM/EACCES = link-prohibited volume.
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                raise
            shutil.copy2(source, destination)
            _hardlink_copy_fallback += 1
        return
    relative = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative)


def _materialize_gt(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    load_binary_flood_mask(source).save(destination, format="PNG")


def _materialize_water_gt(
    water_source: Path,
    flood_source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    compose_full_water_mask(water_source, flood_source).save(
        destination,
        format="PNG",
    )


def _pair_id(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _iso(timestamp: datetime) -> str:
    return timestamp.isoformat()


def _rel(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def _materialize_all(
    assigned, staging_root: Path, data_root: Path, mode: str
) -> None:
    total = len(assigned) * len(OUTPUT_SUBDIRS)
    with tqdm(total=total, unit="file", desc="Materializing") as progress:
        for item in assigned:
            pair = item.pair
            split_root = staging_root / item.output_split
            _materialize_image(
                pair.pre_vv_path, split_root / "A" / item.filename, mode
            )
            progress.update(1)
            _materialize_image(
                pair.post_vv_path, split_root / "B" / item.filename, mode
            )
            progress.update(1)
            _materialize_gt(
                pair.post_flood_path, split_root / "GT" / item.filename
            )
            progress.update(1)
            _materialize_water_gt(
                pair.pre_water_path,
                pair.pre_flood_path,
                split_root / "WATER_GT_A" / item.filename,
            )
            progress.update(1)
            _materialize_water_gt(
                pair.post_water_path,
                pair.post_flood_path,
                split_root / "WATER_GT_B" / item.filename,
            )
            progress.update(1)


def _ensure_split_dirs(staging_root: Path) -> None:
    """Create the five prepared-data directories for every output split."""
    for split_name in OUTPUT_SPLITS:
        for subdirectory in OUTPUT_SUBDIRS:
            (staging_root / split_name / subdirectory).mkdir(
                parents=True,
                exist_ok=True,
            )


def _verify_output_mask(path: Path, expected_size) -> None:
    with Image.open(path) as image:
        mode = image.mode
        size = image.size
        values = set(np.unique(np.asarray(image)).tolist())
    if mode != "L":
        raise RuntimeError(f"output mask must use mode L: {path}")
    if size != expected_size:
        raise RuntimeError(
            f"output mask size mismatch: image={expected_size}, mask={size}, path={path}"
        )
    if not values.issubset({0, 255}):
        raise RuntimeError(f"output mask must contain only {{0,255}}: {path}")


def _verify_tree(staging_root: Path) -> None:
    for split_name in OUTPUT_SPLITS:
        split_root = staging_root / split_name
        name_sets = [
            {path.name for path in (split_root / subdirectory).glob("*.png")}
            for subdirectory in OUTPUT_SUBDIRS
        ]
        if any(names != name_sets[0] for names in name_sets[1:]):
            raise RuntimeError(
                f"prepared basename mismatch in split {split_name}: "
                f"{dict(zip(OUTPUT_SUBDIRS, map(len, name_sets)))}"
            )

        for name in sorted(name_sets[0]):
            with Image.open(split_root / "A" / name) as image_a:
                image_size = image_a.size
            with Image.open(split_root / "B" / name) as image_b:
                if image_b.size != image_size:
                    raise RuntimeError(
                        f"A/B image size mismatch in split {split_name}: {name}"
                    )
            for subdirectory in ("GT", "WATER_GT_A", "WATER_GT_B"):
                _verify_output_mask(
                    split_root / subdirectory / name,
                    image_size,
                )


def _write_manifests(
    staging_root: Path,
    assigned,
    skips,
    data_root: Path,
    args: argparse.Namespace,
    test_internal: dict,
    counts: dict,
    index_stats: dict,
) -> None:
    with (staging_root / "pair_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAIR_MANIFEST_COLUMNS.split(","))
        for item in assigned:
            pair = item.pair
            writer.writerow(
                [
                    item.output_split,
                    item.filename,
                    _pair_id(item.filename),
                    pair.region,
                    pair.x,
                    pair.y,
                    pair.pre_scene,
                    pair.post_scene,
                    _iso(pair.pre_timestamp),
                    _iso(pair.post_timestamp),
                    pair.gap_days,
                    pair.pre_flood_pixels,
                    pair.post_flood_pixels,
                    pair.policy,
                    _rel(pair.pre_vv_path, data_root),
                    _rel(pair.post_vv_path, data_root),
                    _rel(pair.pre_flood_path, data_root),
                    _rel(pair.post_flood_path, data_root),
                    _rel(pair.pre_water_path, data_root),
                    _rel(pair.post_water_path, data_root),
                    pair.pre_water_body_pixels,
                    pair.post_water_body_pixels,
                    pair.water_gt_a_pixels,
                    pair.water_gt_b_pixels,
                    WATER_GT_FORMULA,
                ]
            )

    with (staging_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "filename"))
        for item in assigned:
            writer.writerow((item.output_split, item.filename))

    with (staging_root / "skipped_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in skips:
            handle.write(json.dumps(record, default=str) + "\n")

    skip_counts = Counter(record["reason"] for record in skips)
    policy_counts = Counter(item.pair.policy for item in assigned)
    gaps = [item.pair.gap_days for item in assigned]
    gap_stats = {
        "min": min(gaps) if gaps else None,
        "max": max(gaps) if gaps else None,
        "mean": round(sum(gaps) / len(gaps), 2) if gaps else None,
    }
    water_pixels = {
        split_name: {
            "water_gt_a": sum(
                item.pair.water_gt_a_pixels
                for item in assigned
                if item.output_split == split_name
            ),
            "water_gt_b": sum(
                item.pair.water_gt_b_pixels
                for item in assigned
                if item.output_split == split_name
            ),
        }
        for split_name in OUTPUT_SPLITS
    }
    qc_report = {
        "skip_counts": dict(skip_counts),
        "policy_counts": dict(policy_counts),
        "gap_days": gap_stats,
        "pairs_per_output_split": counts,
        "water_supervised_pairs_per_output_split": dict(counts),
        "water_pixels_per_output_split": water_pixels,
        "water_gt_formula": WATER_GT_FORMULA,
        "index_stats": index_stats,
        "test_internal": test_internal,
        "hardlink_copy_fallback": _hardlink_copy_fallback,
        "alignment_note": (
            "Tile alignment inferred from matching (region, x, y) keys and equal "
            "image shapes; the PNG mirror has no Sentinel-1 product/orbit/geotransform."
        ),
    }
    (staging_root / "qc_report.json").write_text(
        json.dumps(qc_report, indent=2, default=str) + "\n", encoding="utf-8"
    )

    metadata = {
        "converter": "prepare_etci_pairs.py",
        "converter_version": CONVERTER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(data_root),
        "output": str(args.output),
        "repo_id": args.repo_id if args.download else None,
        "revision": args.revision if args.download else None,
        "pair_policy": args.pair_policy,
        "polarization": "vv",
        "output_subdirectories": list(OUTPUT_SUBDIRS),
        "gt_semantics": "post-date flood_label",
        "water_gt_formula": WATER_GT_FORMULA,
        "mask_encoding": "single-channel PNG mode L with values {0,255}",
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "mode": args.mode,
        "min_vv_bytes": args.min_vv_bytes,
        "max_gap_days": args.max_gap_days,
        "keep_negative_post": args.keep_negative_post,
        "max_saturated_fraction": args.max_saturated_fraction,
        "split_strategy": SPLIT_STRATEGY,
        "counts": counts,
        "water_supervised_counts": dict(counts),
        "assumptions": [
            "Earlier VV is treated as the pre-event image; later flood_label is GT.",
            "Each full-water target is derived from water_body_label OR flood_label.",
            "Coordinate equality does not prove strict geographic registration.",
        ],
    }
    (staging_root / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _print_summary(args, data_root, counts, test_internal, skips) -> None:
    print(f"Source data root: {data_root}")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}  Policy: {args.pair_policy}")
    print(f"Polarization: vv  Val ratio: {args.val_ratio}  Seed: {args.seed}")
    for split_name in OUTPUT_SPLITS:
        print(f"  {split_name}: {counts.get(split_name, 0)} pairs")
    print(f"Skipped records: {len(skips)}")
    for reason, count in sorted(Counter(record["reason"] for record in skips).items()):
        print(f"    {reason}: {count}")
    if test_internal.get("present"):
        print(
            f"  test_internal: {test_internal['vv_files']} VV, "
            f"{test_internal['flood_label_files']} flood_label, "
            f"{test_internal['water_body_label_files']} water_body_label -> "
            f"{test_internal.get('excluded_reason') or 'excluded'}"
        )
    if _hardlink_copy_fallback:
        print(f"  hardlink -> copy fallbacks: {_hardlink_copy_fallback}")


def _stats_dict(stats) -> dict:
    return {
        "observations": stats.observations,
        "scenes": stats.scenes,
        "vv_files": stats.vv_files,
        "flood_files": stats.flood_files,
        "water_files": stats.water_files,
    }


def prepare(args: argparse.Namespace) -> dict:
    global _hardlink_copy_fallback
    _hardlink_copy_fallback = 0
    _validate_args(args)

    data_root = resolve_data_root(args)
    trusted_symlink_root = (
        _trusted_download_root(data_root)
        if args.download
        else None
    )

    train_observations, train_skips, train_stats = index_labeled_split(
        data_root,
        "train",
        trusted_symlink_root,
    )
    test_observations, test_skips, test_stats = index_labeled_split(
        data_root,
        "test",
        trusted_symlink_root,
    )
    test_internal = inspect_test_internal(data_root)

    pair_kwargs = dict(
        policy=args.pair_policy,
        min_vv_bytes=args.min_vv_bytes,
        max_gap_days=args.max_gap_days,
        keep_negative_post=args.keep_negative_post,
        max_saturated_fraction=args.max_saturated_fraction,
    )
    train_pairs, train_pair_skips = build_temporal_pairs(train_observations, **pair_kwargs)
    test_pairs, test_pair_skips = build_temporal_pairs(test_observations, **pair_kwargs)
    skips = train_skips + test_skips + train_pair_skips + test_pair_skips

    assigned = list(assign_train_val_groups(train_pairs, args.val_ratio, args.seed))
    assigned.extend(assign_all_to_split(test_pairs, "test"))
    assigned = finalize_assignments(assigned)

    if not assigned:
        raise ValueError(
            "no temporal pairs produced; review skip reasons and consider "
            "--pair-policy adjacent-any"
        )

    counts = {split_name: 0 for split_name in OUTPUT_SPLITS}
    for item in assigned:
        counts[item.output_split] += 1

    index_stats = {
        "train": _stats_dict(train_stats),
        "test": _stats_dict(test_stats),
    }

    _print_summary(args, data_root, counts, test_internal, skips)

    if args.dry_run:
        print("Dry run completed; no files were created.")
        return {"dry_run": True, "counts": counts, "skips": len(skips)}

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging = output.with_name(f".{output.name}.partial")
    if staging.exists():
        raise FileExistsError(
            f"staging directory already exists from an earlier run: {staging}"
        )

    staging.mkdir(parents=True)
    try:
        _ensure_split_dirs(staging)
        _materialize_all(assigned, staging, data_root, args.mode)
        _write_manifests(
            staging, assigned, skips, data_root, args, test_internal, counts, index_stats
        )
        _verify_tree(staging)
        os.replace(staging, output)
    except Exception:
        # Never leave a half-written staging dir blocking the next run.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Dataset prepared successfully: {output}")
    return {"output": str(output), "counts": counts}


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        prepare(args)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
