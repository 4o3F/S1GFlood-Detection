"""Publish three stable Kulsary Sigma0 VV GeoTIFFs from restored GRD SAFE."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import rasterio

from infer_safe import (
    InferenceError,
    _snap_cache_entry_dir,
    build_snap_cache_key,
    get_or_create_sigma0,
    load_snap_cache_entry,
    resolve_gpt,
    resolve_snap_cache_root,
    validate_sigma0_raster,
)
from utils.kulsary_products import discover_kulsary_grd_products
from utils.kulsary_raster import sampled_file_fingerprint
from utils.kulsary_temporal import ROLE_DATES


PREPROCESSOR_VERSION = "1.0.0"
DEFAULT_WORK_DIR = Path.home() / "scratch" / "damnet-safe"
DEFAULT_GRAPH = Path(__file__).resolve().parent / "snap" / "s1_grd_preprocess.xml"
DEFAULT_ORBIT_TYPE = "Sentinel Precise (Auto Download)"
DEFAULT_DEM_NAME = "Copernicus 30m Global DEM"
OUTPUT_FILENAMES = {
    "before": "before_sigma0_vv.tif",
    "peak": "peak_sigma0_vv.tif",
    "after": "after_sigma0_vv.tif",
}
MANIFEST_FILENAME = "sigma0_manifest.json"
_LINK_FALLBACK_ERRNOS = {
    errno.EXDEV,
    errno.EPERM,
    errno.EACCES,
    errno.ENOTSUP,
    errno.EMLINK,
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess restored Kulsary GRD SAFE products into three stable "
            "linear Sigma0 VV GeoTIFFs."
        )
    )
    parser.add_argument("--safe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    parser.add_argument("--refresh-snap-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.pixel_spacing) or args.pixel_spacing <= 0:
        raise ValueError("--pixel-spacing must be finite and positive")


def _resolve_paths(args: argparse.Namespace):
    safe_root = args.safe_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    graph = args.graph.expanduser().resolve()
    if not safe_root.is_dir():
        raise FileNotFoundError(f"SAFE root directory is missing: {safe_root}")
    if not graph.is_file():
        raise FileNotFoundError(f"SNAP graph is missing: {graph}")

    args.work_dir = work_dir
    args.graph = graph
    args.no_snap_cache = False
    cache_root = resolve_snap_cache_root(args, work_dir)
    if cache_root is None:
        raise ValueError("a persistent SNAP cache is required")
    staging = output.with_name(f".{output.name}.partial")

    if output == safe_root or _is_within(output, safe_root):
        raise ValueError("output must not be inside --safe-root")
    for label, path in (
        ("work directory", work_dir),
        ("SNAP cache", cache_root),
    ):
        if path == safe_root or _is_within(path, safe_root):
            raise ValueError(f"{label} must not be inside --safe-root: {path}")
    if output == work_dir or _is_within(work_dir, output) or _is_within(output, work_dir):
        raise ValueError("output and work directory must not overlap")
    if output == cache_root or _is_within(cache_root, output) or _is_within(output, cache_root):
        raise ValueError("output and SNAP cache must not overlap")
    if staging == work_dir or _is_within(work_dir, staging):
        raise ValueError("staging and work directory must not overlap")
    if staging == cache_root or _is_within(cache_root, staging):
        raise ValueError("staging and SNAP cache must not overlap")
    return safe_root, output, work_dir, graph, cache_root, staging


def _find_gpt_for_dry_run(value: str) -> str | None:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        resolved = expanded.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
        return None
    return shutil.which(value)


def _probe_cache(args, products, graph, cache_root, gpt):
    if gpt is None:
        return {
            role: "unavailable (gpt not found; cache key not computable)"
            for role in ROLE_DATES
        }
    statuses = {}
    for role, product in products.items():
        try:
            cache_key, inputs = build_snap_cache_key(product, graph, gpt, args)
            entry = _snap_cache_entry_dir(cache_root, product, cache_key)
            cached = load_snap_cache_entry(entry, cache_key, inputs)
            if cached is not None and args.refresh_snap_cache:
                statuses[role] = "refresh (would rebuild)"
            else:
                statuses[role] = "hit" if cached is not None else "miss"
        except (InferenceError, OSError, ValueError) as exc:
            statuses[role] = f"unavailable ({exc})"
    return statuses


def _print_summary(safe_root, output, products, cache_status):
    print(f"SAFE root: {safe_root}")
    print(f"Output: {output}")
    print("Kulsary role bindings:")
    for role in ROLE_DATES:
        product = products[role]
        print(
            f"  {role}: {ROLE_DATES[role].isoformat()} "
            f"{product.identifier} -> {product.root}"
        )
    print("SNAP cache probe:")
    for role in ROLE_DATES:
        print(f"  {role}: {cache_status[role]}")


def _publish_raster(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in _LINK_FALLBACK_ERRNOS:
            raise
    shutil.copy2(source, destination)
    return "copy"


def _create_sigma0_rasters(products, gpt, graph, run_dir, cache_root, args):
    paths = {}
    for role in ROLE_DATES:
        paths[role] = get_or_create_sigma0(
            gpt,
            graph,
            products[role],
            run_dir / OUTPUT_FILENAMES[role],
            args,
            role,
            cache_root,
            args.refresh_snap_cache,
        )
    return paths


def _build_manifest(
    safe_root,
    graph,
    gpt,
    cache_root,
    products,
    source_paths,
    publication_methods,
    staging,
    args,
):
    roles = {}
    for role in ROLE_DATES:
        output_path = staging / OUTPUT_FILENAMES[role]
        roles[role] = {
            "date": ROLE_DATES[role].isoformat(),
            "product_identifier": products[role].identifier,
            "safe_path": str(products[role].root),
            "cache_source_path": str(source_paths[role]),
            "output_filename": OUTPUT_FILENAMES[role],
            "publication_method": publication_methods[role],
            "fingerprint": sampled_file_fingerprint(output_path),
        }
    return {
        "format": "kulsary-sigma0",
        "version": PREPROCESSOR_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "safe_root": str(safe_root),
        "roles": roles,
        "snap": {
            "gpt": str(Path(gpt).resolve()),
            "graph": str(graph),
            "orbit_type": args.orbit_type,
            "dem_name": args.dem_name,
            "target_crs": args.target_crs,
            "pixel_spacing": float(args.pixel_spacing),
            "cache_root": str(cache_root),
            "refresh": bool(args.refresh_snap_cache),
        },
    }


def _verify_staging(staging: Path) -> None:
    expected = {MANIFEST_FILENAME, *OUTPUT_FILENAMES.values()}
    actual = {path.name for path in staging.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError(
            f"Sigma0 staging tree mismatch: expected {sorted(expected)}, "
            f"found {sorted(actual)}"
        )
    for filename in OUTPUT_FILENAMES.values():
        validate_sigma0_raster(staging / filename)
    manifest = json.loads(
        (staging / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if manifest.get("format") != "kulsary-sigma0":
        raise RuntimeError("Sigma0 manifest format is invalid")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(OUTPUT_FILENAMES):
        raise RuntimeError("Sigma0 manifest roles are invalid")
    for role, filename in OUTPUT_FILENAMES.items():
        expected_fingerprint = roles[role].get("fingerprint")
        actual_fingerprint = sampled_file_fingerprint(staging / filename)
        if actual_fingerprint != expected_fingerprint:
            raise RuntimeError(f"Sigma0 manifest fingerprint mismatch for {role}")


def prepare(args: argparse.Namespace) -> dict:
    _validate_args(args)
    safe_root, output, work_dir, graph, cache_root, staging = _resolve_paths(args)
    products = discover_kulsary_grd_products(safe_root)

    dry_run_gpt = _find_gpt_for_dry_run(args.gpt)
    cache_status = _probe_cache(
        args,
        products,
        graph,
        cache_root,
        dry_run_gpt,
    )
    _print_summary(safe_root, output, products, cache_status)
    if args.dry_run:
        print("Dry run completed; no files were created.")
        return {"dry_run": True, "cache_status": cache_status}

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if staging.exists():
        raise FileExistsError(
            f"staging directory already exists from an earlier run: {staging}"
        )

    gpt = resolve_gpt(args.gpt)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=".kulsary-sigma0-", dir=work_dir))
    try:
        source_paths = _create_sigma0_rasters(
            products,
            gpt,
            graph,
            run_dir,
            cache_root,
            args,
        )
        for path in source_paths.values():
            validate_sigma0_raster(path)

        staging.mkdir(parents=True)
        try:
            methods = {}
            for role in ROLE_DATES:
                destination = staging / OUTPUT_FILENAMES[role]
                methods[role] = _publish_raster(source_paths[role], destination)
                validate_sigma0_raster(destination)
            manifest = _build_manifest(
                safe_root,
                graph,
                gpt,
                cache_root,
                products,
                source_paths,
                methods,
                staging,
                args,
            )
            (staging / MANIFEST_FILENAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _verify_staging(staging)
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print(f"Kulsary Sigma0 dataset prepared successfully: {output}")
    return {
        "output": str(output),
        "files": dict(OUTPUT_FILENAMES),
    }


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
