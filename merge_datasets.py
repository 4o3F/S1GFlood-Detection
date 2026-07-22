"""Merge multiple prepared dataset roots into one for joint training.

Each input must be a prepared root with ``{train,val,test}/{A,B,GT}/<name>.png``
(the layout produced by ``prepare_dataset.py`` and ``prepare_etci_pairs.py``).
Files are combined via hardlink/copy/symlink into a single output root that is
a drop-in ``--dataset-dir`` for ``train.py``/``eval.py``. The training loader
shuffles, so samples from every source are interleaved each epoch.

Filename collisions across sources are rejected: each basename must be owned by
exactly one source. ETCI pairs are prefixed ``etci_``, so S1GFloods and ETCI
never collide; merging two ETCI roots (which share names) is caught and fails
loudly unless ``--on-collision rename`` re-namespaces by a source tag.

This tool writes no training or model code; it only reorganizes files.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

SPLITS = ("train", "val", "test")
SUBDIRS = ("A", "B", "GT")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MERGE_VERSION = "1.0.0"

_hardlink_copy_fallback = 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple prepared dataset roots ({train,val,test}/{A,B,GT}) "
            "into one drop-in root for joint training."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="ROOT",
        help="prepared dataset root; repeat for each source (e.g. S1GFloods, ETCI)",
    )
    parser.add_argument("--output", type=Path, required=True, help="merged output root")
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="hardlink",
        help="file materialization mode (default hardlink, no disk duplication)",
    )
    parser.add_argument(
        "--on-collision",
        choices=("fail", "rename"),
        default="fail",
        help="action when two sources share a basename (default fail)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        metavar="NAME",
        help="short tag per --input (same order) used as a rename prefix; "
        "defaults to each input's directory name",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _list_images(directory: Path):
    return sorted(
        name
        for name in os.listdir(directory)
        if not name.startswith(".") and name.lower().endswith(IMAGE_EXTENSIONS)
    )


def resolve_root(path: Path) -> Path:
    """Return the directory holding train/A (prepared root or its data/ subtree)."""
    path = path.expanduser().resolve()
    if (path / "train" / "A").is_dir():
        return path
    if (path / "data" / "train" / "A").is_dir():
        return path / "data"
    raise FileNotFoundError(
        f"could not locate train/A under {path}; pass a prepared root produced "
        "by prepare_dataset.py or prepare_etci_pairs.py"
    )


def _materialize(source: Path, destination: Path, mode: str) -> None:
    global _hardlink_copy_fallback
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                raise
            shutil.copy2(source, destination)
            _hardlink_copy_fallback += 1
        return
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _scan_source(root: Path, tag: str):
    """Return {split: {basename: {A,B,GT paths}}} for the splits present."""
    out = {}
    for split in SPLITS:
        a_dir = root / split / "A"
        if not a_dir.is_dir():
            continue
        names = _list_images(a_dir)
        entries = {}
        for name in names:
            a_path = root / split / "A" / name
            b_path = root / split / "B" / name
            gt_path = root / split / "GT" / name
            if not (b_path.is_file() and gt_path.is_file()):
                raise FileNotFoundError(
                    f"source {tag}: {split}/{name} is missing its B or GT pair"
                )
            entries[name] = {"A": a_path, "B": b_path, "GT": gt_path}
        out[split] = entries
    if not out:
        raise FileNotFoundError(f"source {tag}: no train/val/test/A directories under {root}")
    return out


def build_plan(roots, tags, on_collision: str):
    """Combine per-source scans into a flat plan + collision report.

    plan: {split: [ {tag, basename, paths} ] }
    collisions: {split: [(basename, [tags])] }

    In rename mode every file is re-namespaced to ``{tag}__{name}`` so sources
    never collide; in fail mode basenames are kept as-is and any cross-source
    clash is reported (merge() then aborts).
    """
    plan = {split: [] for split in SPLITS}
    collisions = {split: [] for split in SPLITS}
    seen = {split: {} for split in SPLITS}  # final basename -> owning tag

    for root, tag in zip(roots, tags):
        scan = _scan_source(root, tag)
        for split, entries in scan.items():
            for name, paths in entries.items():
                final_name = f"{tag}__{name}" if on_collision == "rename" else name
                if final_name in seen[split]:
                    collisions[split].append(
                        (final_name, [seen[split][final_name], tag])
                    )
                    continue
                seen[split][final_name] = tag
                plan[split].append({"tag": tag, "basename": final_name, "paths": paths})

    return plan, collisions


def _materialize_plan(plan, staging_root: Path, mode: str, on_collision: str):
    total = sum(len(items) for items in plan.values()) * len(SUBDIRS)
    with tqdm(total=total, unit="file", desc="Merging") as progress:
        for split in SPLITS:
            for item in plan[split]:
                for sub in SUBDIRS:
                    _materialize(
                        item["paths"][sub],
                        staging_root / split / sub / item["basename"],
                        mode,
                    )
                    progress.update(1)


def _ensure_split_dirs(staging_root: Path) -> None:
    for split in SPLITS:
        for sub in SUBDIRS:
            (staging_root / split / sub).mkdir(parents=True, exist_ok=True)


def _write_manifest(staging_root, plan, collisions, sources, args):
    per_split = {split: len(items) for split, items in plan.items()}
    per_source = {tag: {split: 0 for split in SPLITS} for tag in [s[1] for s in sources]}
    for split, items in plan.items():
        for item in items:
            per_source[item["tag"]][split] += 1
    collision_total = sum(len(v) for v in collisions.values())
    manifest = {
        "tool": "merge_datasets.py",
        "merge_version": MERGE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"tag": tag, "root": str(root)} for root, tag in sources],
        "mode": args.mode,
        "on_collision": args.on_collision,
        "counts_per_split": per_split,
        "counts_per_source": per_source,
        "collisions_detected": collision_total,
        "hardlink_copy_fallback": _hardlink_copy_fallback,
    }
    (staging_root / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def merge(args: argparse.Namespace) -> dict:
    global _hardlink_copy_fallback
    _hardlink_copy_fallback = 0

    roots = [resolve_root(Path(p)) for p in args.input]
    if len(roots) < 2:
        print("warning: only one --input given; output will mirror that source")
    tags = args.tag
    if tags is None:
        # Default to each input's directory name, de-duplicated so manifest
        # keys and rename prefixes never clash.
        tags = []
        seen = {}
        for path in args.input:
            base = Path(path).expanduser().name or "input"
            if base in seen:
                seen[base] += 1
                base = f"{base}_{seen[base]}"
            else:
                seen[base] = 0
            tags.append(base)
    if len(tags) != len(roots):
        raise ValueError("number of --tag values must match number of --input roots")
    sources = list(zip(roots, tags))

    plan, collisions = build_plan(roots, tags, args.on_collision)
    collision_total = sum(len(v) for v in collisions.values())
    if collision_total and args.on_collision == "fail":
        sample = next((s, n, ts) for s, lst in collisions.items() for n, ts in lst[:1])
        raise ValueError(
            f"{collision_total} cross-source filename collisions; first: "
            f"{sample[0]}/{sample[1]} shared by {sample[2]}. "
            "Use --on-collision rename (with --tag) to re-namespace."
        )

    per_split = {split: len(items) for split, items in plan.items()}
    print(f"Sources: {len(roots)}")
    for root, tag in sources:
        print(f"  [{tag}] {root}")
    for split in SPLITS:
        print(f"  {split}: {per_split[split]} samples")
    if collision_total:
        print(f"  collisions renamed: {collision_total}")

    if args.dry_run:
        print("Dry run completed; no files were created.")
        return {"dry_run": True, "counts_per_split": per_split}

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
        _materialize_plan(plan, staging, args.mode, args.on_collision)
        manifest = _write_manifest(staging, plan, collisions, sources, args)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Datasets merged successfully: {output}")
    return {"output": str(output), "counts_per_split": per_split, "manifest": manifest}


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        merge(args)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
