import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import torch
import torch.nn as nn

from infer_safe import (
    AlignedRasterPair,
    InferenceError,
    _run_model_batch,
    PATCH_SIZE,
    SafeProduct,
    axis_starts,
    build_gpt_command,
    build_snap_cache_key,
    get_or_create_sigma0,
    make_blend_kernel,
    prepare_model_tile,
    resolve_safe_product,
    resolve_snap_cache_root,
    run_sliding_inference,
    sigma0_to_model_intensity,
    validate_checkpoint_compatibility,
    validate_cli_args,
    validate_safe_pair,
    validate_sigma0_raster,
    write_outputs,
)


class CompatibleFakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.TACE_pre = self._branch()
        self.TACE_post = self._branch()

    @staticmethod
    def _branch():
        branch = nn.Module()
        branch.proj_q = nn.Linear(1, 1, bias=False)
        branch.proj_k = nn.Linear(1, 1, bias=False)
        branch.proj_v = nn.Linear(1, 1, bias=False)
        return branch

    def forward(self, pre, post):
        change = (pre[:, :1] - post[:, :1]) / 255.0
        return torch.cat((-change, change), dim=1)


class IncompatibleFakeModel(nn.Module):
    def forward(self, pre, post):
        return torch.zeros(
            pre.shape[0],
            2,
            PATCH_SIZE,
            PATCH_SIZE,
            dtype=pre.dtype,
            device=pre.device,
        )


class NonFiniteFakeModel(nn.Module):
    def forward(self, pre, post):
        return torch.full(
            (pre.shape[0], 2, PATCH_SIZE, PATCH_SIZE),
            float("nan"),
            dtype=pre.dtype,
            device=pre.device,
        )


class FakeRasterPair:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.crs = rasterio.crs.CRS.from_epsg(32639)
        self.transform = from_origin(500000, 5200000, 10, 10)

    def read(self, window):
        height = int(window.height)
        width = int(window.width)
        pre = np.ones((height, width), dtype=np.float32)
        post = np.full((height, width), 0.01, dtype=np.float32)
        valid = np.ones((height, width), dtype=bool)
        return pre, post, valid


class CliValidationTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "trust_checkpoint": True,
            "stride": 128,
            "batch_size": 4,
            "threshold": 0.5,
            "db_min": -25.0,
            "db_max": 0.0,
            "pixel_spacing": 10.0,
            "snap_cache_dir": None,
            "no_snap_cache": False,
            "refresh_snap_cache": False,
            "work_dir": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_rejects_nonfinite_float_parameters(self):
        for name, value in (
            ("threshold", float("nan")),
            ("db_min", float("nan")),
            ("db_max", float("inf")),
            ("pixel_spacing", float("inf")),
        ):
            with self.subTest(name=name), self.assertRaises(InferenceError):
                validate_cli_args(self._args(**{name: value}))

    def test_requires_checkpoint_trust_acknowledgement(self):
        with self.assertRaisesRegex(InferenceError, "--trust-checkpoint"):
            validate_cli_args(self._args(trust_checkpoint=False))

    def test_rejects_unbounded_batch_size(self):
        with self.assertRaisesRegex(InferenceError, "between 1 and 64"):
            validate_cli_args(self._args(batch_size=65))

    def test_rejects_conflicting_cache_options(self):
        with self.assertRaisesRegex(InferenceError, "cannot be combined"):
            validate_cli_args(
                self._args(no_snap_cache=True, snap_cache_dir="/tmp/cache")
            )
        with self.assertRaisesRegex(InferenceError, "cannot be combined"):
            validate_cli_args(
                self._args(no_snap_cache=True, refresh_snap_cache=True)
            )
        with mock.patch.dict(os.environ, {"SNAP_CACHE_DIR": ""}):
            with self.assertRaisesRegex(InferenceError, "requires"):
                validate_cli_args(self._args(refresh_snap_cache=True))

    def test_no_cache_overrides_environment_default(self):
        args = self._args(no_snap_cache=True)
        with mock.patch.dict(
            os.environ,
            {"SNAP_CACHE_DIR": "/tmp/environment-cache"},
        ):
            validate_cli_args(args)
            self.assertIsNone(resolve_snap_cache_root(args, Path("/tmp/work")))

    def test_refresh_accepts_environment_cache_directory(self):
        args = self._args(refresh_snap_cache=True)
        with mock.patch.dict(
            os.environ,
            {"SNAP_CACHE_DIR": "/tmp/environment-cache"},
        ):
            validate_cli_args(args)
            self.assertEqual(
                resolve_snap_cache_root(args, Path("/tmp/work")),
                Path("/tmp/environment-cache").resolve(),
            )


class IntensityConversionTest(unittest.TestCase):
    def test_maps_known_sigma0_values_to_training_scale(self):
        sigma0 = np.array([[10 ** -2.5, 10 ** -1.25, 1.0]], dtype=np.float32)
        valid = np.ones_like(sigma0, dtype=bool)
        intensity, final_valid = sigma0_to_model_intensity(
            sigma0,
            valid,
            -25.0,
            0.0,
        )
        np.testing.assert_allclose(intensity, [[0.0, 127.5, 255.0]], atol=1e-4)
        np.testing.assert_array_equal(final_valid, valid)

    def test_invalid_values_are_zero_and_masked(self):
        sigma0 = np.array([[0.0, -1.0, np.nan, np.inf, 0.1]], dtype=np.float32)
        valid = np.ones_like(sigma0, dtype=bool)
        intensity, final_valid = sigma0_to_model_intensity(
            sigma0,
            valid,
            -25.0,
            0.0,
        )
        np.testing.assert_array_equal(final_valid, [[False, False, False, False, True]])
        self.assertTrue(np.all(intensity[~final_valid] == 0))

    def test_prepares_three_identical_channels_and_padding(self):
        sigma0 = np.ones((20, 30), dtype=np.float32)
        valid = np.ones((20, 30), dtype=bool)
        tile, padded_valid = prepare_model_tile(sigma0, valid, -25.0, 0.0)
        self.assertEqual(tile.shape, (3, PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(tile.dtype, np.float32)
        np.testing.assert_array_equal(tile[0], tile[1])
        np.testing.assert_array_equal(tile[1], tile[2])
        self.assertTrue(padded_valid[:20, :30].all())
        self.assertFalse(padded_valid[20:, :].any())
        self.assertFalse(padded_valid[:, 30:].any())


class WindowGenerationTest(unittest.TestCase):
    def test_axis_starts_cover_small_exact_and_irregular_lengths(self):
        self.assertEqual(axis_starts(100, 128), [0])
        self.assertEqual(axis_starts(256, 128), [0])
        self.assertEqual(axis_starts(300, 128), [0, 44])
        self.assertEqual(axis_starts(517, 128), [0, 128, 256, 261])

    def test_rejects_stride_gaps(self):
        with self.assertRaises(ValueError):
            axis_starts(512, 257)

    def test_hann_kernel_has_positive_edges(self):
        kernel = make_blend_kernel()
        self.assertEqual(kernel.shape, (PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(kernel.dtype, np.float32)
        self.assertGreater(float(kernel.min()), 0.0)
        self.assertAlmostEqual(float(kernel.max()), 1.0)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_accepts_registered_tace_projections(self):
        model = CompatibleFakeModel()
        self.assertIs(validate_checkpoint_compatibility(model), model)

    def test_rejects_legacy_checkpoint_without_projections(self):
        with self.assertRaisesRegex(InferenceError, "predates commit 11c309a"):
            validate_checkpoint_compatibility(IncompatibleFakeModel())


class ModelBatchTest(unittest.TestCase):
    def test_rejects_nonfinite_probabilities(self):
        tile = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        with self.assertRaisesRegex(InferenceError, "non-finite"):
            _run_model_batch(
                NonFiniteFakeModel(),
                [tile],
                [tile],
                torch.device("cpu"),
            )


class SafeMetadataTest(unittest.TestCase):
    def _write_safe(self, parent, name, start, relative_orbit=159):
        root = Path(parent) / name
        measurement = root / "measurement"
        measurement.mkdir(parents=True)
        (measurement / "s1a-iw-grd-vv-test.tiff").write_bytes(b"placeholder")
        manifest = f"""<?xml version="1.0"?>
<root>
  <productType>GRD</productType>
  <mode>IW</mode>
  <startTime>{start}</startTime>
  <stopTime>{start}</stopTime>
  <pass>DESCENDING</pass>
  <relativeOrbitNumber type="start">{relative_orbit}</relativeOrbitNumber>
  <transmitterReceiverPolarisation>VV</transmitterReceiverPolarisation>
</root>
"""
        (root / "manifest.safe").write_text(manifest, encoding="utf-8")
        return resolve_safe_product(str(root))

    def test_reads_safe_manifest_and_vv_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            product = self._write_safe(
                directory,
                "S1A_IW_GRDH_1SDV_20240402T141444_TEST.SAFE",
                "2024-04-02T14:14:44Z",
            )
            self.assertEqual(product.product_type, "GRD")
            self.assertEqual(product.acquisition_mode, "IW")
            self.assertEqual(product.relative_orbit, 159)
            self.assertIn("VV", product.polarizations)

    def test_rejects_reversed_dates_and_orbit_mismatch(self):
        base = dict(
            root=Path("/tmp/example.SAFE"),
            manifest=Path("/tmp/example.SAFE/manifest.safe"),
            identifier="example.SAFE",
            platform="S1A",
            product_type="GRD",
            acquisition_mode="IW",
            stop_time=None,
            orbit_direction="DESCENDING",
            polarizations=frozenset({"VV"}),
        )
        pre = SafeProduct(
            **base,
            start_time=datetime(2024, 4, 14, tzinfo=timezone.utc),
            relative_orbit=159,
        )
        post = SafeProduct(
            **base,
            start_time=datetime(2024, 4, 2, tzinfo=timezone.utc),
            relative_orbit=160,
        )
        with self.assertRaisesRegex(InferenceError, "earlier acquisition"):
            validate_safe_pair(pre, post)

    def test_gpt_command_preserves_paths_with_spaces(self):
        product = SafeProduct(
            root=Path("/tmp/pre product.SAFE"),
            manifest=Path("/tmp/pre product.SAFE/manifest.safe"),
            identifier="pre product.SAFE",
            platform="S1A",
            product_type="GRD",
            acquisition_mode="IW",
            start_time=None,
            stop_time=None,
            orbit_direction=None,
            relative_orbit=None,
            polarizations=frozenset({"VV"}),
        )
        args = argparse.Namespace(
            orbit_type="Sentinel Precise (Auto Download)",
            dem_name="Copernicus 30m Global DEM",
            target_crs="AUTO:42001",
            pixel_spacing=10.0,
        )
        command = build_gpt_command(
            "/opt/SNAP/bin/gpt",
            Path("/tmp/graph file.xml"),
            product,
            Path("/tmp/output file.tif"),
            args,
        )
        self.assertIn("-Pinput=/tmp/pre product.SAFE/manifest.safe", command)
        self.assertIn("-Poutput=/tmp/output file.tif", command)


class SnapCacheTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "orbit_type": "Sentinel Precise (Auto Download)",
            "dem_name": "Copernicus 30m Global DEM",
            "target_crs": "EPSG:32639",
            "pixel_spacing": 10.0,
            "checkpoint": "/tmp/checkpoint-a.pth",
            "stride": 128,
            "batch_size": 4,
            "device": "cuda:0",
            "db_min": -25.0,
            "db_max": 0.0,
            "threshold": 0.5,
            "snap_cache_dir": None,
            "no_snap_cache": False,
            "refresh_snap_cache": False,
            "work_dir": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _product(directory, identifier="S1A_TEST_GRD_20240402T000000.SAFE"):
        root = Path(directory) / identifier
        root.mkdir(parents=True)
        manifest = root / "manifest.safe"
        manifest.write_text("<manifest>one</manifest>", encoding="utf-8")
        measurement = root / "measurement" / "scene-vv-test.tiff"
        measurement.parent.mkdir()
        measurement.write_bytes(b"measurement-one")
        annotation = root / "annotation" / "scene.xml"
        annotation.parent.mkdir()
        annotation.write_text("<annotation>one</annotation>", encoding="utf-8")
        return SafeProduct(
            root=root,
            manifest=manifest,
            identifier=identifier,
            platform="S1A",
            product_type="GRD",
            acquisition_mode="IW",
            start_time=datetime(2024, 4, 2, tzinfo=timezone.utc),
            stop_time=None,
            orbit_direction="DESCENDING",
            relative_orbit=159,
            polarizations=frozenset({"VV"}),
        )

    @staticmethod
    def _write_sigma0(path, value=1.0, polarizations=("VV",)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=16,
            height=16,
            count=len(polarizations),
            dtype="float32",
            crs="EPSG:32639",
            transform=from_origin(500000, 5200000, 10, 10),
            nodata=0.0,
        ) as dataset:
            for band, polarization in enumerate(polarizations, start=1):
                dataset.write(
                    np.full((16, 16), value, dtype=np.float32),
                    band,
                )
                dataset.set_band_description(
                    band,
                    f"Sigma0_{polarization}",
                )

    def _cache_fixture(self, directory):
        product = self._product(directory)
        graph = Path(directory) / "graph.xml"
        graph.write_text("<graph version='one'/>", encoding="utf-8")
        gpt = Path(directory) / "gpt"
        gpt.write_text("launcher-one", encoding="utf-8")
        return product, graph, gpt

    def test_cache_key_ignores_inference_only_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            args = self._args()
            first, _ = build_snap_cache_key(product, graph, str(gpt), args)
            args.checkpoint = "/tmp/checkpoint-b.pth"
            args.stride = 64
            args.batch_size = 1
            args.device = "cpu"
            args.db_min = -30.0
            args.db_max = 5.0
            args.threshold = 0.8
            second, _ = build_snap_cache_key(product, graph, str(gpt), args)
            self.assertEqual(first, second)

    def test_cache_key_changes_with_preprocessing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            base_args = self._args()
            base, _ = build_snap_cache_key(product, graph, str(gpt), base_args)

            for name, value in (
                ("orbit_type", "Sentinel Restituted (Auto Download)"),
                ("dem_name", "SRTM 1Sec HGT"),
                ("target_crs", "EPSG:4326"),
                ("pixel_spacing", 20.0),
            ):
                changed_args = self._args(**{name: value})
                changed, _ = build_snap_cache_key(
                    product,
                    graph,
                    str(gpt),
                    changed_args,
                )
                with self.subTest(name=name):
                    self.assertNotEqual(base, changed)

            graph.write_text("<graph version='two'/>", encoding="utf-8")
            changed, _ = build_snap_cache_key(product, graph, str(gpt), base_args)
            self.assertNotEqual(base, changed)

            graph.write_text("<graph version='one'/>", encoding="utf-8")
            product.manifest.write_text("<manifest>two</manifest>", encoding="utf-8")
            changed, _ = build_snap_cache_key(product, graph, str(gpt), base_args)
            self.assertNotEqual(base, changed)

            product.manifest.write_text("<manifest>one</manifest>", encoding="utf-8")
            gpt.write_text("launcher-two-with-new-size", encoding="utf-8")
            changed, _ = build_snap_cache_key(product, graph, str(gpt), base_args)
            self.assertNotEqual(base, changed)

    def test_cache_key_changes_when_safe_payload_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            args = self._args()
            base, _ = build_snap_cache_key(product, graph, str(gpt), args)

            preview = product.root / "preview" / "notes.txt"
            preview.parent.mkdir()
            preview.write_text("not used by SNAP", encoding="utf-8")
            unchanged, _ = build_snap_cache_key(product, graph, str(gpt), args)
            self.assertEqual(base, unchanged)

            annotation = product.root / "annotation" / "scene.xml"
            annotation.write_text("<annotation>two</annotation>", encoding="utf-8")
            changed, _ = build_snap_cache_key(product, graph, str(gpt), args)
            self.assertNotEqual(base, changed)

            annotation.write_text("<annotation>one</annotation>", encoding="utf-8")
            measurement = product.root / "measurement" / "scene-vv-test.tiff"
            stat = measurement.stat()
            measurement.write_bytes(b"measurement-two")
            os.utime(
                measurement,
                ns=(stat.st_atime_ns, stat.st_mtime_ns),
            )
            changed, _ = build_snap_cache_key(product, graph, str(gpt), args)
            self.assertNotEqual(base, changed)

    def test_dual_polarization_cache_tracks_vh_and_uses_semantic_name(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            vh_measurement = (
                product.root / "measurement" / "scene-vh-test.tiff"
            )
            vh_measurement.write_bytes(b"vh-one")
            args = self._args()
            args.polarizations = ("VV", "VH")
            first_key, inputs = build_snap_cache_key(
                product,
                graph,
                str(gpt),
                args,
            )
            self.assertEqual(inputs["polarizations"], ["VV", "VH"])
            inventory_paths = {
                record["path"] for record in inputs["safe_source_inventory"]
            }
            self.assertIn("measurement/scene-vv-test.tiff", inventory_paths)
            self.assertIn("measurement/scene-vh-test.tiff", inventory_paths)

            vh_measurement.write_bytes(b"vh-two")
            second_key, _ = build_snap_cache_key(
                product,
                graph,
                str(gpt),
                args,
            )
            self.assertNotEqual(first_key, second_key)

            cache_root = Path(directory) / "dual-cache"
            output = Path(directory) / "run" / "dual.tif"

            def build(path):
                self._write_sigma0(path, polarizations=("VV", "VH"))

            cached = get_or_create_sigma0(
                str(gpt),
                graph,
                product,
                output,
                args,
                "dual",
                cache_root,
                False,
                build=build,
            )
            self.assertEqual(cached.name, "sigma0_vv_vh.tif")
            validate_sigma0_raster(cached, ("VV", "VH"))

    def test_cache_miss_installs_and_second_call_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            cache_root = Path(directory) / "cache"
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            calls = 0

            def build(path):
                nonlocal calls
                calls += 1
                self._write_sigma0(path)

            first = get_or_create_sigma0(
                str(gpt),
                graph,
                product,
                run_dir / "pre_sigma0_vv.tif",
                self._args(),
                "pre-event",
                cache_root,
                False,
                build,
            )
            second = get_or_create_sigma0(
                str(gpt),
                graph,
                product,
                run_dir / "pre_sigma0_vv.tif",
                self._args(),
                "pre-event",
                cache_root,
                False,
                build,
            )

            self.assertEqual(calls, 1)
            self.assertEqual(first, second)
            self.assertTrue((first.parent / ".complete").is_file())
            self.assertTrue((first.parent / "meta.json").is_file())
            self.assertTrue((first.parent.parent.parent / "current").is_symlink())
            self.assertEqual(list(cache_root.rglob(".partial-*")), [])

    def test_incomplete_entry_and_refresh_force_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            cache_root = Path(directory) / "cache"
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            calls = 0

            def build(path):
                nonlocal calls
                calls += 1
                self._write_sigma0(path, value=float(calls))

            cached = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                self._args(), "pre-event", cache_root, False, build,
            )
            (cached.parent / ".complete").unlink()
            rebuilt = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                self._args(), "pre-event", cache_root, False, build,
            )
            refreshed = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                self._args(), "pre-event", cache_root, True, build,
            )

            self.assertEqual(calls, 3)
            self.assertNotEqual(rebuilt, refreshed)
            self.assertTrue(rebuilt.is_file())
            self.assertEqual(
                (refreshed.parent.parent.parent / "current").resolve(),
                refreshed.parent,
            )
            validate_sigma0_raster(refreshed)

    def test_corrupt_metadata_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            cache_root = Path(directory) / "cache"
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            calls = 0

            def build(path):
                nonlocal calls
                calls += 1
                self._write_sigma0(path, value=float(calls))

            first = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                self._args(), "pre-event", cache_root, False, build,
            )
            (first.parent / "meta.json").write_text("{", encoding="utf-8")
            rebuilt = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                self._args(), "pre-event", cache_root, False, build,
            )

            self.assertEqual(calls, 2)
            self.assertNotEqual(first, rebuilt)
            validate_sigma0_raster(rebuilt)

    def test_failed_refresh_keeps_previous_generation_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            cache_root = Path(directory) / "cache"
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            args = self._args()
            original = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                args, "pre-event", cache_root, False, self._write_sigma0,
            )
            real_replace = os.replace

            def fail_current_publication(source, destination):
                if Path(destination).name == "current":
                    raise OSError("simulated publication failure")
                return real_replace(source, destination)

            with mock.patch(
                "infer_safe.os.replace",
                side_effect=fail_current_publication,
            ):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    get_or_create_sigma0(
                        str(gpt), graph, product, run_dir / "pre.tif",
                        args, "pre-event", cache_root, True,
                        self._write_sigma0,
                    )

            reused = get_or_create_sigma0(
                str(gpt), graph, product, run_dir / "pre.tif",
                args, "pre-event", cache_root, False,
                lambda path: self.fail("cache should remain readable"),
            )
            self.assertEqual(reused, original)
            validate_sigma0_raster(reused)

    def test_failure_log_is_preserved_outside_partial_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            cache_root = Path(directory) / "cache"
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            output = run_dir / "pre_sigma0_vv.tif"
            run_log = output.with_suffix(".snap.log")

            def fail_preprocess(gpt_arg, graph_arg, product_arg, path, args, label):
                cache_log = path.with_suffix(".snap.log")
                cache_log.write_text("SNAP failed\n", encoding="utf-8")
                raise InferenceError(f"SNAP failed. Log: {cache_log}")

            with mock.patch(
                "infer_safe.preprocess_safe",
                side_effect=fail_preprocess,
            ):
                with self.assertRaisesRegex(
                    InferenceError,
                    re.escape(str(run_log)),
                ):
                    get_or_create_sigma0(
                        str(gpt), graph, product, output, self._args(),
                        "pre-event", cache_root, False,
                    )

            self.assertEqual(run_log.read_text(encoding="utf-8"), "SNAP failed\n")
            self.assertEqual(list(cache_root.rglob(".partial-*")), [])

    def test_disabled_cache_uses_run_output(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            work_parent = Path(directory) / "work"
            work_parent.mkdir()
            args = self._args(no_snap_cache=True, work_dir=str(work_parent))
            self.assertIsNone(resolve_snap_cache_root(args, work_parent))
            output = work_parent / "run" / "pre_sigma0_vv.tif"

            result = get_or_create_sigma0(
                str(gpt), graph, product, output, args, "pre-event",
                None, False, self._write_sigma0,
            )
            self.assertEqual(result, output)
            validate_sigma0_raster(result)

    def test_cache_keys_are_independent_per_product(self):
        with tempfile.TemporaryDirectory() as directory:
            product, graph, gpt = self._cache_fixture(directory)
            other = self._product(
                directory,
                identifier="S1A_TEST_GRD_20240414T000000.SAFE",
            )
            first, _ = build_snap_cache_key(product, graph, str(gpt), self._args())
            second, _ = build_snap_cache_key(other, graph, str(gpt), self._args())
            self.assertNotEqual(first, second)

    def test_sigma0_validation_rejects_nonpositive_raster(self):
        with tempfile.TemporaryDirectory() as directory:
            raster = Path(directory) / "zero.tif"
            self._write_sigma0(raster, value=0.0)
            with self.assertRaisesRegex(InferenceError, "no finite positive"):
                validate_sigma0_raster(raster)


class RasterAlignmentTest(unittest.TestCase):
    @staticmethod
    def _write_raster(path, width, height, transform, value):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs="EPSG:32639",
            transform=transform,
            nodata=0.0,
        ) as dataset:
            dataset.write(np.full((height, width), value, dtype=np.float32), 1)

    def test_aligns_post_raster_to_pre_grid_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            pre_path = Path(directory) / "pre.tif"
            post_path = Path(directory) / "post.tif"
            self._write_raster(pre_path, 300, 270, from_origin(0, 2700, 10, 10), 1.0)
            self._write_raster(post_path, 300, 270, from_origin(20, 2700, 10, 10), 0.1)

            with AlignedRasterPair(pre_path, post_path) as pair:
                self.assertEqual(pair.width, 298)
                self.assertEqual(pair.height, 270)
                pre, post, valid = pair.read(Window(0, 0, 256, 256))
                self.assertEqual(pre.shape, (256, 256))
                self.assertEqual(post.shape, (256, 256))
                self.assertTrue(valid.all())


class SlidingInferenceOutputTest(unittest.TestCase):
    def test_stitches_and_writes_georeferenced_outputs(self):
        pair = FakeRasterPair(width=300, height=270)
        model = CompatibleFakeModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "work"
            work_dir.mkdir()
            probability_sum, weight_sum = run_sliding_inference(
                pair,
                model,
                torch.device("cpu"),
                stride=128,
                batch_size=2,
                db_min=-25.0,
                db_max=0.0,
                work_dir=work_dir,
            )
            self.assertTrue(np.all(weight_sum > 0))

            mask_path = Path(directory) / "flood.tif"
            probability_path = Path(directory) / "flood_probability.tif"
            write_outputs(
                mask_path,
                probability_path,
                pair,
                probability_sum,
                weight_sum,
                threshold=0.5,
                tags={"test": "true"},
            )
            del probability_sum
            del weight_sum

            with rasterio.open(mask_path) as mask_dataset:
                mask = mask_dataset.read(1)
                self.assertEqual(mask_dataset.crs, pair.crs)
                self.assertEqual(mask_dataset.transform, pair.transform)
                self.assertEqual(mask_dataset.dtypes, ("uint8",))
                self.assertEqual(set(np.unique(mask)), {255})
                self.assertTrue((mask_dataset.dataset_mask() == 255).all())

            with rasterio.open(probability_path) as probability_dataset:
                probability = probability_dataset.read(1)
                self.assertEqual(probability_dataset.dtypes, ("float32",))
                self.assertGreaterEqual(float(probability.min()), 0.0)
                self.assertLessEqual(float(probability.max()), 1.0)
                self.assertTrue((probability_dataset.dataset_mask() == 255).all())


if __name__ == "__main__":
    unittest.main()
