import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

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
    make_blend_kernel,
    prepare_model_tile,
    resolve_safe_product,
    run_sliding_inference,
    sigma0_to_model_intensity,
    validate_checkpoint_compatibility,
    validate_cli_args,
    validate_safe_pair,
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
