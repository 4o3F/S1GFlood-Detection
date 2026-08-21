from __future__ import annotations

import inspect
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
import rasterio

from utils.kulsary_raster import (
    PATCH_SIZE,
    LazySigma0Stack,
    Sigma0Stack,
    linear_sigma0_to_clipped_db,
    plan_valid_tiles,
    tile_window,
    warp_masks,
)
from utils.kulsary_temporal import (
    MaskRef,
    TileKey,
    discover_mask_refs,
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


def write_sigma0(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    *,
    crs: str = "EPSG:4326",
    nodata: float = 0.0,
    count: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        for band in range(1, count + 1):
            dataset.write(array.astype(np.float32), band)
            dataset.set_band_description(band, "Sigma0_VV")


def dummy_mask_ref(size: tuple[int, int], transform: Affine) -> MaskRef:
    return MaskRef(
        role="peak",
        png_path=Path("peak.png"),
        world_path=Path("peak.pgw"),
        size=size,
        transform=transform,
        positive_pixels=0,
    )


class RasterFixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.transform = Affine(0.001, 0, 53.0, 0, -0.001, 48.0)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_roles(self, arrays: dict[str, np.ndarray]) -> dict[str, Path]:
        paths = {}
        for role, array in arrays.items():
            path = self.root / f"{role}.tif"
            write_sigma0(path, array, self.transform)
            paths[role] = path
        return paths

    def _write_aligned_masks(self, arrays: dict[str, np.ndarray]) -> dict[str, MaskRef]:
        names = {
            "before": "1_water_before_20240402.png",
            "peak": "2_water_during_20240414.png",
            "after": "3_water_after_20240426.png",
        }
        for role, array in arrays.items():
            write_mask(self.root / names[role], array, self.transform)
        return discover_mask_refs(self.root)


class CommonGridAndReadTest(RasterFixtureTest):
    def test_identical_extents_keep_native_peak_grid(self):
        size = 512
        arrays = {
            role: np.full((size, size), value, dtype=np.float32)
            for role, value in (("before", 0.01), ("peak", 0.1), ("after", 0.03))
        }
        arrays["peak"][10, 20] = 0.5
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((size, size), self.transform)

        with Sigma0Stack(paths, mask_ref) as stack:
            self.assertEqual(stack.grid.width, size)
            self.assertEqual(stack.grid.height, size)
            self.assertEqual(int(stack.grid.peak_window.col_off), 0)
            self.assertEqual(int(stack.grid.peak_window.row_off), 0)
            window = Window(0, 0, size, size)
            peak, peak_valid = stack.read_role("peak", window)
            before, before_valid = stack.read_role("before", window)
            arrays_all, valid = stack.read(window)
            np.testing.assert_allclose(peak, arrays["peak"], atol=1e-6)
            np.testing.assert_allclose(before, arrays["before"], atol=1e-6)
            np.testing.assert_allclose(arrays_all["after"], arrays["after"], atol=1e-6)
            self.assertTrue(peak_valid.all())
            self.assertTrue(before_valid.all())
            self.assertTrue(valid.all())

    def test_mask_clip_offsets_native_peak_window(self):
        size = 512
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        arrays["peak"][128:384, 128:384] = np.linspace(
            0.1,
            0.9,
            256 * 256,
            dtype=np.float32,
        ).reshape(256, 256)
        paths = self._write_roles(arrays)
        mask_transform = Affine(
            0.001,
            0,
            53.0 + 128 * 0.001,
            0,
            -0.001,
            48.0 - 128 * 0.001,
        )
        mask_ref = dummy_mask_ref((256, 256), mask_transform)

        with Sigma0Stack(paths, mask_ref) as stack:
            self.assertEqual(stack.grid.width, 256)
            self.assertEqual(stack.grid.height, 256)
            self.assertEqual(int(stack.grid.peak_window.col_off), 128)
            self.assertEqual(int(stack.grid.peak_window.row_off), 128)
            peak, valid = stack.read_role("peak", Window(0, 0, 256, 256))
            self.assertTrue(valid.all())
            np.testing.assert_allclose(peak, arrays["peak"][128:384, 128:384], atol=1e-6)

            with Sigma0Stack(paths, grid=stack.grid) as reused:
                reused_peak, reused_valid = reused.read_role(
                    "peak",
                    Window(0, 0, 256, 256),
                )
                np.testing.assert_array_equal(reused_peak, peak)
                np.testing.assert_array_equal(reused_valid, valid)

    def test_reused_grid_rejects_shifted_peak_georeferencing(self):
        size = 256
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((size, size), self.transform)
        with Sigma0Stack(paths, mask_ref) as stack:
            grid = stack.grid

        shifted = Affine(
            self.transform.a,
            self.transform.b,
            self.transform.c + self.transform.a,
            self.transform.d,
            self.transform.e,
            self.transform.f,
        )
        write_sigma0(paths["peak"], arrays["peak"], shifted)
        with self.assertRaisesRegex(ValueError, "transform"):
            Sigma0Stack(paths, grid=grid)

    def test_constructor_requires_exactly_one_grid_source(self):
        size = 256
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((size, size), self.transform)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Sigma0Stack(paths)
        with Sigma0Stack(paths, mask_ref) as stack:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                Sigma0Stack(paths, mask_ref, grid=stack.grid)

    def test_invalid_source_uses_local_value_error(self):
        size = 256
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        paths = self._write_roles(arrays)
        write_sigma0(
            paths["peak"],
            arrays["peak"],
            self.transform,
            count=2,
        )
        mask_ref = dummy_mask_ref((size, size), self.transform)
        with self.assertRaises(ValueError):
            Sigma0Stack(paths, mask_ref)

    def test_module_does_not_import_infer_safe(self):
        import utils.kulsary_raster as raster

        self.assertNotIn("infer_safe", inspect.getsource(raster))


class WarpAlignmentTest(RasterFixtureTest):
    def test_nearest_warp_preserves_aligned_water_square(self):
        size = 512
        water = {
            "before": np.zeros((size, size), dtype=bool),
            "peak": np.zeros((size, size), dtype=bool),
            "after": np.zeros((size, size), dtype=bool),
        }
        water["peak"][40:80, 90:130] = True
        water["before"][40:60, 90:110] = True
        water["after"][60:80, 110:130] = True
        mask_refs = self._write_aligned_masks(water)
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        paths = self._write_roles(arrays)

        with Sigma0Stack(paths, mask_refs["peak"]) as stack:
            water_masks, coverage = warp_masks(mask_refs, stack.grid)
            self.assertTrue(coverage.all())
            self.assertEqual(water_masks["peak"].shape, (size, size))
            np.testing.assert_array_equal(water_masks["peak"], water["peak"])
            np.testing.assert_array_equal(water_masks["before"], water["before"])
            np.testing.assert_array_equal(water_masks["after"], water["after"])

    def test_wgs84_masks_align_to_utm_sigma0_grid(self):
        size = 256
        geographic_transform = Affine(
            0.0001,
            0,
            53.55,
            0,
            -0.0001,
            47.30,
        )
        water = {
            role: np.zeros((size, size), dtype=bool)
            for role in ("before", "peak", "after")
        }
        for mask in water.values():
            mask[96:160, 96:160] = True
        names = {
            "before": "1_water_before_20240402.png",
            "peak": "2_water_during_20240414.png",
            "after": "3_water_after_20240426.png",
        }
        for role, mask in water.items():
            write_mask(self.root / names[role], mask, geographic_transform)
        mask_refs = discover_mask_refs(self.root)

        geographic_bounds = array_bounds(
            size,
            size,
            geographic_transform,
        )
        utm_bounds = transform_bounds(
            "EPSG:4326",
            "EPSG:32639",
            *geographic_bounds,
            densify_pts=21,
        )
        utm_transform = from_bounds(*utm_bounds, size, size)
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        paths = {}
        for role, array in arrays.items():
            path = self.root / f"{role}-utm.tif"
            write_sigma0(
                path,
                array,
                utm_transform,
                crs="EPSG:32639",
            )
            paths[role] = path

        with Sigma0Stack(paths, mask_refs["peak"]) as stack:
            water_masks, coverage = warp_masks(mask_refs, stack.grid)
        self.assertGreater(float(coverage.mean()), 0.90)
        peak = water_masks["peak"]
        rows, cols = np.nonzero(peak)
        self.assertGreater(rows.size, 0)
        self.assertLess(abs(float(rows.mean()) - stack.grid.height / 2), 8.0)
        self.assertLess(abs(float(cols.mean()) - stack.grid.width / 2), 8.0)
        np.testing.assert_array_equal(water_masks["before"], peak)
        np.testing.assert_array_equal(water_masks["after"], peak)


class PlanValidTilesTest(RasterFixtureTest):
    def test_edge_and_invalid_skip_reasons(self):
        width, height = 300, 280
        arrays = {
            role: np.full((height, width), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        arrays["before"][:PATCH_SIZE, :PATCH_SIZE] = 0.0
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((width, height), self.transform)
        coverage = np.ones((height, width), dtype=bool)
        coverage[0:10, 0:10] = False

        with Sigma0Stack(paths, mask_ref) as stack:
            kept, skips = plan_valid_tiles(stack, coverage)
        self.assertEqual(kept, [])
        reasons = [record["reason"] for record in skips]
        self.assertEqual(
            reasons,
            [
                "incomplete_right_edge",
                "incomplete_bottom_edge",
                "incomplete_mask_coverage",
            ],
        )
        self.assertEqual(skips[0]["width_pixels"], width % PATCH_SIZE)
        self.assertEqual(skips[1]["height_pixels"], height % PATCH_SIZE)
        self.assertEqual(skips[2]["tile_row"], 0)
        self.assertEqual(skips[2]["tile_col"], 0)

    def test_invalid_common_pixels_are_skipped(self):
        size = 512
        arrays = {
            role: np.full((size, size), 0.2, dtype=np.float32)
            for role in ("before", "peak", "after")
        }
        arrays["after"][256:512, 0:256] = np.nan
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((size, size), self.transform)
        coverage = np.ones((size, size), dtype=bool)

        with Sigma0Stack(paths, mask_ref) as stack:
            kept, skips = plan_valid_tiles(stack, coverage)
        self.assertEqual(kept, [TileKey(0, 0), TileKey(0, 1), TileKey(1, 1)])
        invalid = [record for record in skips if record["reason"] == "invalid_common_pixels"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["tile_row"], 1)
        self.assertEqual(invalid[0]["tile_col"], 0)
        self.assertLess(invalid[0]["valid_fraction"], 1.0)


class ClippedDbTest(unittest.TestCase):
    def test_anchors_and_clipping(self):
        sigma0 = np.array([[10 ** -2.5, 10 ** -1.25, 1.0]], dtype=np.float32)
        valid = np.ones_like(sigma0, dtype=bool)
        db = linear_sigma0_to_clipped_db(sigma0, valid, -25.0, 0.0)
        self.assertEqual(db.dtype, np.float32)
        np.testing.assert_allclose(db, [[-25.0, -12.5, 0.0]], atol=1e-4)

        clipped = linear_sigma0_to_clipped_db(
            np.array([[1e-6, 10.0]], dtype=np.float32),
            np.ones((1, 2), dtype=bool),
            -25.0,
            0.0,
        )
        np.testing.assert_allclose(clipped, [[-25.0, 0.0]], atol=1e-4)

    def test_rejects_invalid_range_and_nonpositive_input(self):
        valid = np.array([[True]], dtype=bool)
        with self.assertRaisesRegex(ValueError, "finite"):
            linear_sigma0_to_clipped_db(np.array([[0.1]]), valid, np.nan, 0.0)
        with self.assertRaisesRegex(ValueError, "smaller"):
            linear_sigma0_to_clipped_db(np.array([[0.1]]), valid, 0.0, -25.0)
        with self.assertRaisesRegex(ValueError, "fully valid"):
            linear_sigma0_to_clipped_db(np.array([[0.0]]), valid, -25.0, 0.0)
        with self.assertRaisesRegex(ValueError, "fully valid"):
            linear_sigma0_to_clipped_db(
                np.array([[0.1]]),
                np.array([[False]]),
                -25.0,
                0.0,
            )


class LazySigma0StackTest(RasterFixtureTest):
    def test_pickle_pid_reopen_and_no_live_handles(self):
        size = 256
        arrays = {
            role: np.full((size, size), value, dtype=np.float32)
            for role, value in (("before", 0.01), ("peak", 0.1), ("after", 0.03))
        }
        paths = self._write_roles(arrays)
        mask_ref = dummy_mask_ref((size, size), self.transform)
        window = tile_window(TileKey(0, 0))

        with Sigma0Stack(paths, mask_ref) as stack:
            grid = stack.grid
            expected, expected_valid = stack.read_role("peak", window)

        lazy = LazySigma0Stack(paths, grid)
        first, first_valid = lazy.read_role("peak", window)
        np.testing.assert_allclose(first, expected, atol=1e-6)
        np.testing.assert_array_equal(first_valid, expected_valid)
        self.assertIsNotNone(lazy._stack)

        payload = pickle.dumps(lazy)
        self.assertIsNone(lazy._stack)
        restored = pickle.loads(payload)
        self.assertIsNone(restored._stack)
        self.assertIsNone(restored._pid)
        restored_peak, restored_valid = restored.read_role("peak", window)
        np.testing.assert_allclose(restored_peak, expected, atol=1e-6)
        np.testing.assert_array_equal(restored_valid, expected_valid)
        restored.close()

        lazy.read_role("peak", window)
        current_stack = lazy._stack
        current_pid = lazy._pid
        self.assertIsNotNone(current_stack)
        with mock.patch(
            "utils.kulsary_raster.os.getpid",
            return_value=current_pid + 999,
        ):
            reopened, reopened_valid = lazy.read_role("peak", window)
            self.assertIsNot(lazy._stack, current_stack)
            np.testing.assert_allclose(reopened, expected, atol=1e-6)
            np.testing.assert_array_equal(reopened_valid, expected_valid)
        lazy.close()
        self.assertIsNone(lazy._stack)


if __name__ == "__main__":
    unittest.main()
