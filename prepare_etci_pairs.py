"""Convert ETCI-2021 into the S1GFloods bi-temporal train/val/test layout.

Standalone offline converter. For each output sample::

    A  = earlier-date VV for the same region and tile coordinate
    B  = later-date VV for the same region and tile coordinate
    GT = later-date flood_label, rewritten to single-channel PNG mode L {0,255}

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

from tqdm import tqdm

from utils.etci_temporal import (
    AssignedPair,
    assign_all_to_split,
    assign_train_val_groups,
    build_temporal_pairs,
    finalize_assignments,
    index_labeled_split,
    inspect_test_internal,
    load_binary_flood_mask,
)

DEFAULT_REPO_ID = "blanchon/ETCI-2021-Flood-Detection"
DEFAULT_REVISION = "921e207ea6aa26e9366fd200725a98b0067f9d6b"
CONVERTER_VERSION = "1.0.0"
SPLIT_STRATEGY = (
    "train pairs group-split into train/val by (region,x,y); "
    "source test pairs to output test; test_internal excluded "
    "(never indexed; labels if present are not used)"
)
OUTPUT_SPLITS = ("train", "val", "test")
PAIR_MANIFEST_COLUMNS = (
    "split,filename,pair_id,region,x,y,pre_scene,post_scene,"
    "pre_datetime,post_datetime,gap_days,pre_flood_pixels,post_flood_pixels,"
    "policy,pre_vv,post_vv,pre_flood,post_flood"
)

_hardlink_copy_fallback = 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ETCI-2021 multi-date VV scenes into the S1GFloods "
            "bi-temporal train/val/test A/B/GT layout."
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


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    global _hardlink_copy_fallback
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
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


def _pair_id(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _iso(timestamp: datetime) -> str:
    return timestamp.isoformat()


def _rel(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root))
    except ValueError:
        return str(path)


def _materialize_all(
    assigned, staging_root: Path, data_root: Path, mode: str
) -> None:
    total = len(assigned) * 3
    with tqdm(total=total, unit="file", desc="Materializing") as progress:
        for item in assigned:
            split_root = staging_root / item.output_split
            _materialize_image(
                item.pair.pre_vv_path, split_root / "A" / item.filename, mode
            )
            progress.update(1)
            _materialize_image(
                item.pair.post_vv_path, split_root / "B" / item.filename, mode
            )
            progress.update(1)
            _materialize_gt(
                item.pair.post_flood_path, split_root / "GT" / item.filename
            )
            progress.update(1)


def _ensure_split_dirs(staging_root: Path) -> None:
    """Create the A/B/GT contract directories for every split.

    An empty split (e.g. ``val`` with too few coordinate groups) still gets its
    directories so downstream loaders that ``os.listdir`` the split do not raise
    ``FileNotFoundError`` on a missing path.
    """
    for split_name in OUTPUT_SPLITS:
        for sub in ("A", "B", "GT"):
            (staging_root / split_name / sub).mkdir(parents=True, exist_ok=True)


def _verify_tree(staging_root: Path) -> None:
    for split_name in OUTPUT_SPLITS:
        split_root = staging_root / split_name
        if not split_root.is_dir():
            continue
        names = []
        for sub in ("A", "B", "GT"):
            sub_dir = split_root / sub
            names.append(
                {p.name for p in sub_dir.glob("*.png")} if sub_dir.is_dir() else set()
            )
        if names[0] != names[1] or names[0] != names[2]:
            raise RuntimeError(f"A/B/GT basename mismatch in split {split_name}")


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
    qc_report = {
        "skip_counts": dict(skip_counts),
        "policy_counts": dict(policy_counts),
        "gap_days": gap_stats,
        "pairs_per_output_split": counts,
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
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "mode": args.mode,
        "min_vv_bytes": args.min_vv_bytes,
        "max_gap_days": args.max_gap_days,
        "keep_negative_post": args.keep_negative_post,
        "max_saturated_fraction": args.max_saturated_fraction,
        "split_strategy": SPLIT_STRATEGY,
        "counts": counts,
        "assumptions": [
            "Earlier VV is treated as the pre-event image; later flood_label is GT.",
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
            f"{test_internal['flood_label_files']} flood_label -> "
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
    }


def prepare(args: argparse.Namespace) -> dict:
    global _hardlink_copy_fallback
    _hardlink_copy_fallback = 0
    _validate_args(args)

    data_root = resolve_data_root(args)

    train_observations, train_skips, train_stats = index_labeled_split(data_root, "train")
    test_observations, test_skips, test_stats = index_labeled_split(data_root, "test")
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
