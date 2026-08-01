from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

import prepare_kulsary_pairs as converter
from infer_safe import InferenceError, PATCH_SIZE
from utils.kulsary_temporal import (
    PAIR_VARIANTS,
    TileKey,
    assign_spatial_blocks,
    build_filename,
    compose_flood_mask,
    discover_mask_refs,
    expand_pair_variants,
    iter_full_windows,
    load_binary_water_mask,
    parse_world_file,
    spatial_block_key,
    world_file_to_affine,
)


def write_mask(path: Path, array: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(array, 255, 0).astype(np.uint8)).save(path)
    center_c = transform.c + transform.a / 2.0 + transform.b / 2.0
    center_f = transform.f + transform.d / 2.0 + transform.e / 2.0
    path.with_suffix(".pgw").write_text(
        "\n".join(
            str(value)
            for value in (
                transform.a,
                transform.d,
                transform.b,
                transform.e,
                center_c,
                center_f,
            )
        )
        + "\n",
        encoding="utf-8",
    )


def write_safe(path: Path, acquisition_date: str) -> None:
    path.mkdir(parents=True)
    (path / "measurement").mkdir()
    (path / "measurement" / "synthetic-vv-scene.tiff").write_bytes(b"vv")
    start = f"{acquisition_date}T14:14:44.000000Z"
    stop = f"{acquisition_date}T14:15:10.000000Z"
    (path / "manifest.safe").write_text(
        (
            "<root>"
            "<productType>GRD</productType>"
            "<mode>IW</mode>"
            f"<startTime>{start}</startTime>"
            f"<stopTime>{stop}</stopTime>"
            "<pass>ASCENDING</pass>"
            '<relativeOrbitNumber type="start">159</relativeOrbitNumber>'
            "<transmitterReceiverPolarisation>VV</transmitterReceiverPolarisation>"
            "</root>"
        ),
        encoding="utf-8",
    )


def write_sigma0(path: Path, array: np.ndarray, transform: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=0.0,
    ) as dataset:
        dataset.write(array.astype(np.float32), 1)
        dataset.set_band_description(1, "Sigma0_VV")


class TemporalLogicTest(unittest.TestCase):
    def test_world_file_center_to_corner_includes_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.pgw"
            path.write_text("2\n0.5\n0.25\n-2\n11.125\n19.25\n", encoding="utf-8")
            affine = world_file_to_affine(parse_world_file(path))
        self.assertEqual(affine, Affine(2, 0.25, 10, 0.5, -2, 20))

    def test_binary_mask_and_flood_set_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            source = np.array([[0, 255], [1, 0]], dtype=np.uint8)
            Image.fromarray(source).save(path)
            loaded = load_binary_water_mask(path)
            np.testing.assert_array_equal(
                loaded,
                np.array([[False, True], [True, False]]),
            )

            peak = np.array([[True, True], [False, True]])
            baseline = np.array([[True, False], [False, False]])
            np.testing.assert_array_equal(
                compose_flood_mask(peak, baseline),
                np.array([[False, True], [False, True]]),
            )

            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):
                load_binary_water_mask(path)

    def test_window_iteration_drops_partial_edges(self):
        windows = list(iter_full_windows(500, 500, 256))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], TileKey(0, 0))
        self.assertEqual(
            (windows[0][1].width, windows[0][1].height),
            (256, 256),
        )

    def test_spatial_split_is_deterministic_and_block_grouped(self):
        tiles = [TileKey(row, col) for row in range(6) for col in range(6)]
        first = assign_spatial_blocks(tiles, block_tiles=2, seed=42)
        second = assign_spatial_blocks(reversed(tiles), block_tiles=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "val", "test"})

        split_by_block = {}
        for tile, split_name in first.items():
            block = spatial_block_key(tile, 2)
            self.assertEqual(split_by_block.setdefault(block, split_name), split_name)

        assigned = expand_pair_variants(tiles, first)
        by_tile = {}
        for item in assigned:
            by_tile.setdefault(item.tile, set()).add(item.output_split)
        self.assertTrue(all(len(splits) == 1 for splits in by_tile.values()))
        self.assertEqual(len(assigned), len(tiles) * 2)

    def test_split_requires_three_spatial_blocks(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            assign_spatial_blocks(
                [TileKey(0, 0), TileKey(0, 1)],
                block_tiles=2,
            )

    def test_mask_discovery_rejects_subpixel_transform_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            transform = Affine(0.001, 0, 53.0, 0, -0.001, 48.0)
            mask = np.zeros((2, 2), dtype=bool)
            paths = (
                source / "1_water_before_20240402.png",
                source / "2_water_during_20240414.png",
                source / "3_water_after_20240426.png",
            )
            for path in paths:
                write_mask(path, mask, transform)

            world_path = paths[2].with_suffix(".pgw")
            values = world_path.read_text(encoding="utf-8").splitlines()
            values[4] = str(float(values[4]) + 5e-6)
            world_path.write_text("\n".join(values) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mask transform differs"):
                discover_mask_refs(source)

    def test_semantic_filename(self):
        self.assertEqual(
            build_filename(TileKey(3, 12), PAIR_VARIANTS[0]),
            "kulsary_r0003_c0012_before_to_peak.png",
        )


class ConverterIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "masks"
        self.safe_root = self.root / "restored"
        self.output = self.root / "prepared"
        self.work = self.root / "work"
        self.cache = self.root / "cache"
        self.raster_root = self.root / "rasters"
        self.raster_root.mkdir()

        self.size = 768
        self.transform = Affine(0.001, 0, 53.0, 0, -0.001, 48.0)
        before_water = np.zeros((self.size, self.size), dtype=bool)
        peak_water = np.zeros_like(before_water)
        after_water = np.zeros_like(before_water)
        peak_water[256:384, 256:384] = True
        before_water[256:320, 256:320] = True
        after_water[320:384, 320:384] = True

        write_mask(
            self.source / "nested" / "1_water_before_20240402.png",
            before_water,
            self.transform,
        )
        write_mask(
            self.source / "nested" / "2_water_during_20240414.png",
            peak_water,
            self.transform,
        )
        write_mask(
            self.source / "nested" / "3_water_after_20240426.png",
            after_water,
            self.transform,
        )

        self.safe_root.mkdir()
        names = {
            "before": (
                "S1A_IW_GRDH_1SDV_20240426T141445_20240426T141510_"
                "053606_068232_UNKNOWN1.SAFE"
            ),
            "peak": (
                "S1A_IW_GRDH_1SDV_20240402T141444_20240402T141510_"
                "053256_06745E_UNKNOWN2.SAFE"
            ),
            "after": (
                "S1A_IW_GRDH_1SDV_20240414T141444_20240414T141509_"
                "053431_067B51_UNKNOWN3.SAFE"
            ),
        }
        dates = {
            "before": "2024-04-02",
            "peak": "2024-04-14",
            "after": "2024-04-26",
        }
        self.safe_paths = {}
        for role, acquisition_date in dates.items():
            path = self.safe_root / names[role]
            write_safe(path, acquisition_date)
            self.safe_paths[role] = path

        before_sigma0 = np.full(
            (self.size, self.size),
            0.01,
            dtype=np.float32,
        )
        before_sigma0[:PATCH_SIZE, :PATCH_SIZE] = 0.0
        peak_sigma0 = np.full(
            (self.size, self.size),
            0.1,
            dtype=np.float32,
        )
        after_sigma0 = np.full(
            (self.size, self.size),
            0.03162278,
            dtype=np.float32,
        )
        self.sigma0_paths = {
            "before": self.raster_root / "before.tif",
            "peak": self.raster_root / "peak.tif",
            "after": self.raster_root / "after.tif",
        }
        write_sigma0(self.sigma0_paths["before"], before_sigma0, self.transform)
        write_sigma0(self.sigma0_paths["peak"], peak_sigma0, self.transform)
        write_sigma0(self.sigma0_paths["after"], after_sigma0, self.transform)

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, *extra):
        argv = [
            "--source",
            str(self.source),
            "--safe-root",
            str(self.safe_root),
            "--output",
            str(self.output),
            "--work-dir",
            str(self.work),
            "--snap-cache-dir",
            str(self.cache),
            "--gpt",
            str(self.root / "missing-gpt"),
        ]
        argv.extend(extra)
        return converter.parse_args(argv)

    def _get_sigma0(
        self,
        gpt,
        graph,
        product,
        output,
        args,
        label,
        cache_root,
        refresh,
    ):
        return self.sigma0_paths[label]

    def _full_run(self):
        with mock.patch.object(
            converter,
            "resolve_gpt",
            return_value="/fake/gpt",
        ), mock.patch.object(
            converter,
            "get_or_create_sigma0",
            side_effect=self._get_sigma0,
        ):
            return converter.prepare(self._args())

    def test_mask_discovery_and_manifest_date_role_binding(self):
        refs = discover_mask_refs(self.source)
        self.assertEqual(set(refs), {"before", "peak", "after"})

        products = converter._discover_products(self.safe_root)
        for role, product in products.items():
            self.assertEqual(product.root, self.safe_paths[role])
        self.assertIn("20240426", products["before"].identifier)
        self.assertIn("20240414", products["after"].identifier)

    def test_discovers_restore_launcher_products_layout(self):
        products_dir = self.safe_root / "products"
        products_dir.mkdir()
        expected_paths = {}

        for role, original in self.safe_paths.items():
            cache_dir = self.safe_root / f"{role}-cache"
            cache_dir.mkdir()
            restored = cache_dir / original.name
            original.rename(restored)
            (products_dir / restored.name).symlink_to(
                Path("..") / cache_dir.name / restored.name,
                target_is_directory=True,
            )
            expected_paths[role] = restored.resolve()

        products = converter._discover_products(self.safe_root)
        self.assertEqual(
            {role: product.root for role, product in products.items()},
            expected_paths,
        )

    def test_dry_run_never_resolves_gpt_or_writes(self):
        with mock.patch.object(
            converter,
            "resolve_gpt",
            side_effect=AssertionError("dry-run resolved gpt"),
        ):
            result = converter.prepare(self._args("--dry-run"))
        self.assertTrue(result["dry_run"])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.cache.exists())
        self.assertFalse(
            self.output.with_name(f".{self.output.name}.partial").exists()
        )

    def test_full_run_without_gpt_fails_before_staging(self):
        with self.assertRaises(InferenceError):
            converter.prepare(self._args())
        self.assertFalse(self.output.exists())
        self.assertFalse(
            self.output.with_name(f".{self.output.name}.partial").exists()
        )

    def test_cog_safe_is_rejected_with_safe_root_instruction(self):
        original = self.safe_paths["before"]
        original.rename(original.with_name(original.name[:-5] + "_COG.SAFE"))
        with self.assertRaises(InferenceError) as context:
            converter.prepare(self._args())
        self.assertIn("--safe-root", str(context.exception))
        self.assertFalse(
            self.output.with_name(f".{self.output.name}.partial").exists()
        )

    def test_output_contract_manifests_grouping_and_loader(self):
        result = self._full_run()
        self.assertEqual(sum(result["counts"].values()), 16)
        self.assertTrue(all(count > 0 for count in result["counts"].values()))

        artifacts = {
            "split_manifest.csv",
            "split_metadata.json",
            "pair_manifest.csv",
            "qc_report.json",
            "skipped_records.jsonl",
        }
        self.assertTrue(
            all((self.output / filename).is_file() for filename in artifacts)
        )

        with (self.output / "pair_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            pair_rows = list(csv.DictReader(handle))
        with (self.output / "split_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            split_rows = list(csv.DictReader(handle))
        self.assertEqual(len(pair_rows), 16)
        self.assertEqual(len(split_rows), len(pair_rows))

        invalid_names = {
            "kulsary_r0000_c0000_before_to_peak.png",
            "kulsary_r0000_c0000_after_to_peak.png",
        }
        filenames = {row["filename"] for row in pair_rows}
        self.assertTrue(invalid_names.isdisjoint(filenames))
        self.assertTrue(
            any(int(row["gt_positive_pixels"]) == 0 for row in pair_rows)
        )

        after_row = next(
            row for row in pair_rows if row["variant"] == "after_to_peak"
        )
        self.assertEqual(after_row["a_date"], "2024-04-26")
        self.assertEqual(after_row["b_date"], "2024-04-14")
        self.assertEqual(after_row["chronological"], "false")

        block_splits = {}
        for row in pair_rows:
            block = (row["block_row"], row["block_col"])
            self.assertEqual(
                block_splits.setdefault(block, row["split"]),
                row["split"],
            )

        for split_name in ("train", "val", "test"):
            names = []
            for subdirectory in ("A", "B", "GT"):
                directory = self.output / split_name / subdirectory
                names.append({path.name for path in directory.glob("*.png")})
            self.assertEqual(names[0], names[1])
            self.assertEqual(names[0], names[2])
            self.assertTrue(names[0])
            for filename in names[0]:
                with Image.open(self.output / split_name / "A" / filename) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (256, 256))
                with Image.open(self.output / split_name / "GT" / filename) as image:
                    self.assertEqual(image.mode, "L")
                    self.assertTrue(set(image.getdata()).issubset({0, 255}))

        before_row = next(
            row
            for row in pair_rows
            if row["variant"] == "before_to_peak" and row["tile_row"] != "0"
        )
        matching_after = next(
            row
            for row in pair_rows
            if row["variant"] == "after_to_peak"
            and row["tile_row"] == before_row["tile_row"]
            and row["tile_col"] == before_row["tile_col"]
        )
        before_a = np.asarray(
            Image.open(
                self.output
                / before_row["split"]
                / "A"
                / before_row["filename"]
            )
        )
        after_a = np.asarray(
            Image.open(
                self.output
                / matching_after["split"]
                / "A"
                / matching_after["filename"]
            )
        )
        before_b = np.asarray(
            Image.open(
                self.output
                / before_row["split"]
                / "B"
                / before_row["filename"]
            )
        )
        after_b = np.asarray(
            Image.open(
                self.output
                / matching_after["split"]
                / "B"
                / matching_after["filename"]
            )
        )
        self.assertLess(float(before_a.mean()), float(after_a.mean()))
        np.testing.assert_array_equal(before_b, after_b)

        metadata = json.loads(
            (self.output / "split_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["mode"], "render")
        self.assertEqual(metadata["counts"], result["counts"])

        qc = json.loads(
            (self.output / "qc_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(qc["tiles"]["kept_tiles"], 8)
        self.assertEqual(qc["tiles"]["invalid_tiles"], 1)

        from utils.dataloaders import FloodDetection, train_path

        train_full, _ = train_path(str(self.output) + os.sep)
        dataset = FloodDetection(train_full, aug=False)
        image_a, image_b, gt, name = dataset[0]
        self.assertEqual(tuple(image_a.shape), (3, 256, 256))
        self.assertEqual(tuple(image_b.shape), (3, 256, 256))
        self.assertEqual(tuple(gt.shape), (256, 256))
        self.assertTrue(name.startswith("kulsary_"))

    def test_work_and_cache_cannot_overlap_reserved_output_paths(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            converter.prepare(
                self._args("--work-dir", str(self.output))
            )

        staging = self.output.with_name(f".{self.output.name}.partial")
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            converter.prepare(
                self._args("--snap-cache-dir", str(staging))
            )

        nested_output = self.cache / "entries"
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            converter.prepare(
                self._args("--output", str(nested_output))
            )

        self.assertFalse(self.output.exists())
        self.assertFalse(staging.exists())
        self.assertFalse(nested_output.exists())

    def test_existing_output_and_staging_are_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            converter.prepare(self._args())
        self.output.rmdir()

        staging = self.output.with_name(f".{self.output.name}.partial")
        staging.mkdir()
        with self.assertRaises(FileExistsError):
            converter.prepare(self._args())

    def test_materialization_failure_removes_staging(self):
        staging = self.output.with_name(f".{self.output.name}.partial")
        with mock.patch.object(
            converter,
            "resolve_gpt",
            return_value="/fake/gpt",
        ), mock.patch.object(
            converter,
            "get_or_create_sigma0",
            side_effect=self._get_sigma0,
        ), mock.patch.object(
            converter,
            "_write_tiles",
            side_effect=RuntimeError("synthetic write failure"),
        ):
            with self.assertRaises(RuntimeError):
                converter.prepare(self._args())
        self.assertFalse(self.output.exists())
        self.assertFalse(staging.exists())

    def test_main_uses_error_prefix(self):
        with self.assertRaises(SystemExit) as context:
            converter.main(
                [
                    "--source",
                    str(self.source),
                    "--safe-root",
                    str(self.safe_root),
                    "--output",
                    str(self.output),
                    "--train-ratio",
                    "2",
                ]
            )
        self.assertIn("Error:", str(context.exception))


if __name__ == "__main__":
    unittest.main()
