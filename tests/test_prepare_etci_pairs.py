from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from prepare_etci_pairs import main, parse_args, prepare
from utils.etci_temporal import (
    assign_all_to_split,
    assign_train_val_groups,
    build_temporal_pairs,
    clear_flood_pixel_cache,
    compose_full_water_mask,
    finalize_assignments,
    index_labeled_split,
    inspect_test_internal,
    load_binary_flood_mask,
    load_binary_water_mask,
    parse_flood_filename,
    parse_scene_name,
    parse_vv_filename,
    parse_water_filename,
    vv_passes_qc,
)


def write_rgb(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def write_flood(
    path: Path,
    mask01: np.ndarray,
    *,
    as_rgb: bool = True,
    values=(0, 255),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask01 > 0, values[1], values[0]).astype(np.uint8)
    if as_rgb:
        rgb = np.stack([binary, binary, binary], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(path)
    else:
        Image.fromarray(binary, mode="L").save(path)


def sar_tile(seed: int, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 220, size=(size, size), dtype=np.uint8)


def add_scene(data_root: Path, split: str, region: str, ts: str, tiles: dict) -> None:
    scene = f"{region}_{ts}"
    base = data_root / split / scene / "tiles"
    for (x, y), payload in tiles.items():
        if payload.get("vv") is not None or payload.get("vv_small"):
            vv_path = base / "vv" / f"{scene}_x-{x}_y-{y}_vv.png"
            if payload.get("vv_small"):
                tiny = np.zeros((32, 32), dtype=np.uint8)
                tiny[:16] = 255
                write_rgb(vv_path, tiny)
            else:
                write_rgb(vv_path, payload["vv"])
        if payload.get("flood") is not None:
            flood_name = payload.get("flood_name", f"{scene}_x-{x}_y-{y}.png")
            write_flood(
                base / "flood_label" / flood_name,
                payload["flood"],
                as_rgb=payload.get("flood_rgb", True),
                values=payload.get("flood_values", (0, 255)),
            )
        if payload.get("vh") is not None:
            write_rgb(base / "vh" / f"{scene}_x-{x}_y-{y}_vh.png", payload["vh"])
        if payload.get("water") is not None:
            write_flood(
                base / "water_body_label" / f"{scene}_x-{x}_y-{y}.png",
                payload["water"],
            )


def build_standard_fixture(root: Path) -> Path:
    data_root = root / "data"
    zero = np.zeros((32, 32), dtype=np.uint8)
    ones = np.ones((32, 32), dtype=np.uint8)

    add_scene(
        data_root,
        "train",
        "north_alabama",
        "20170314t115609",
        {
            (1, 1): {"vv": sar_tile(1), "flood": zero, "vh": sar_tile(11), "water": ones},
            (9, 9): {"vv_small": True, "flood": zero, "water": zero},
        },
    )
    add_scene(
        data_root,
        "train",
        "north_alabama",
        "20170606t115613",
        {
            (1, 1): {
                "vv": sar_tile(2),
                "flood": ones,
                "flood_name": "north_alabama_20170606t115613_x-1_y-1_vv.png",
                "water": zero,
            },
            (2, 2): {"vv": sar_tile(22), "flood": ones, "water": zero},
        },
    )
    add_scene(
        data_root,
        "train",
        "north_alabama",
        "20170712t115615",
        {
            (1, 1): {"vv": sar_tile(3), "flood": ones, "water": zero},
            (2, 2): {"vv": sar_tile(23), "flood": ones, "water": ones},
        },
    )
    add_scene(
        data_root,
        "train",
        "nebraska",
        "20180401t000000",
        {(5, 5): {"vv": sar_tile(5), "flood": zero, "water": zero}},
    )
    add_scene(
        data_root,
        "train",
        "nebraska",
        "20180413t000000",
        {(5, 5): {"vv": sar_tile(6), "flood": ones, "water": zero}},
    )
    add_scene(
        data_root,
        "test",
        "florence",
        "20180914t000000",
        {(3, 3): {"vv": sar_tile(7), "flood": zero, "water": ones}},
    )
    add_scene(
        data_root,
        "test",
        "florence",
        "20180926t000000",
        {(3, 3): {"vv": sar_tile(8), "flood": ones, "water": zero}},
    )

    junk_dir = (
        data_root / "train" / "north_alabama_20170314t115609" / "tiles" / "vv"
        / ".ipynb_checkpoints"
    )
    junk_dir.mkdir(parents=True, exist_ok=True)
    write_rgb(junk_dir / "ignored.png", sar_tile(99))
    write_rgb(
        data_root / "train" / "north_alabama_20170314t115609" / "tiles" / "vv"
        / ".hidden_vv.png",
        sar_tile(100),
    )
    add_scene(
        data_root,
        "test_internal",
        "red_river_north",
        "20190501t000000",
        {
            (0, 0): {
                "vv": sar_tile(12),
                "water": np.zeros((32, 32), dtype=np.uint8),
            }
        },
    )
    return data_root


def replace_tiles_with_hub_cache_symlinks(
    data_root: Path,
    repository_cache_root: Path,
) -> None:
    blobs = repository_cache_root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    tile_paths = sorted(
        path
        for path in data_root.rglob("*.png")
        if path.parent.name in {"vv", "flood_label", "water_body_label"}
    )
    for index, path in enumerate(tile_paths):
        blob = blobs / f"blob-{index:04d}"
        blob.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(os.path.relpath(blob, path.parent))


class ParseHelpersTest(unittest.TestCase):
    def test_parse_scene_name_supports_underscored_region(self):
        region, ts = parse_scene_name("north_alabama_20170314t115609")
        self.assertEqual(region, "north_alabama")
        self.assertEqual(ts, datetime(2017, 3, 14, 11, 56, 9))

    def test_parse_vv_flood_and_water_filenames(self):
        name = "bangladesh_20170314t115609_x-10_y-11_vv.png"
        expected = ("bangladesh_20170314t115609", 10, 11)
        self.assertEqual(parse_vv_filename(name), expected)
        self.assertEqual(
            parse_flood_filename("bangladesh_20170314t115609_x-10_y-11.png"),
            expected,
        )
        self.assertEqual(parse_flood_filename(name), expected)
        self.assertEqual(
            parse_water_filename("bangladesh_20170314t115609_x-10_y-11.png"),
            expected,
        )
        self.assertEqual(parse_water_filename(name), expected)

    def test_reject_vh_label_names(self):
        name = "bangladesh_20170314t115609_x-10_y-11_vh.png"
        with self.assertRaises(ValueError):
            parse_flood_filename(name)
        with self.assertRaises(ValueError):
            parse_water_filename(name)


class IndexAndPairTest(unittest.TestCase):
    def setUp(self):
        clear_flood_pixel_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root = build_standard_fixture(self.root)

    def tearDown(self):
        clear_flood_pixel_cache()
        self.tmp.cleanup()

    def test_index_requires_all_modalities_and_accepts_label_aliases(self):
        observations, _, stats = index_labeled_split(self.data_root, "train")
        self.assertGreaterEqual(stats.observations, 4)
        self.assertEqual(stats.observations, stats.water_files)
        coords = {(item.region, item.x, item.y, item.scene) for item in observations}
        self.assertIn(
            ("north_alabama", 1, 1, "north_alabama_20170606t115613"), coords
        )
        self.assertTrue(
            all(not item.vv_path.name.startswith(".") for item in observations)
        )
        self.assertTrue(
            all(item.water_path.parent.name == "water_body_label" for item in observations)
        )

    def test_no_cross_split_pairing(self):
        train_obs, _, _ = index_labeled_split(self.data_root, "train")
        test_obs, _, _ = index_labeled_split(self.data_root, "test")
        train_pairs, _ = build_temporal_pairs(
            train_obs, policy="nearest-flood-free", min_vv_bytes=64
        )
        test_pairs, _ = build_temporal_pairs(
            test_obs, policy="nearest-flood-free", min_vv_bytes=64
        )
        self.assertTrue(all(pair.source_split == "train" for pair in train_pairs))
        self.assertTrue(all(pair.source_split == "test" for pair in test_pairs))
        self.assertTrue(all(pair.region != "florence" for pair in train_pairs))

    def test_nearest_flood_free_uses_clean_pre(self):
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, _ = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=64
        )
        alabama = [
            pair
            for pair in pairs
            if pair.region == "north_alabama" and pair.x == 1 and pair.y == 1
        ]
        self.assertGreaterEqual(len(alabama), 1)
        self.assertTrue(all(pair.pre_flood_pixels == 0 for pair in alabama))
        july = [pair for pair in alabama if pair.post_scene.endswith("20170712t115615")]
        self.assertEqual(len(july), 1)
        self.assertEqual(july[0].pre_scene, "north_alabama_20170314t115609")
        self.assertIn(july[0].pre_scene, str(july[0].pre_water_path))
        self.assertIn(july[0].post_scene, str(july[0].post_water_path))
        self.assertEqual(july[0].water_gt_a_pixels, 32 * 32)
        self.assertEqual(july[0].water_gt_b_pixels, 32 * 32)

    def test_adjacent_any_allows_flooded_pre(self):
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, _ = build_temporal_pairs(
            observations, policy="adjacent-any", min_vv_bytes=64
        )
        alabama = [
            pair
            for pair in pairs
            if pair.region == "north_alabama"
            and pair.x == 1
            and pair.y == 1
            and pair.post_scene.endswith("20170712t115615")
        ]
        self.assertEqual(len(alabama), 1)
        self.assertEqual(alabama[0].pre_scene, "north_alabama_20170606t115613")
        self.assertGreater(alabama[0].pre_flood_pixels, 0)
        self.assertIn(alabama[0].pre_scene, str(alabama[0].pre_water_path))
        self.assertIn(alabama[0].post_scene, str(alabama[0].post_water_path))

    def test_coordinate_intersection_and_artifact_skip(self):
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, _ = build_temporal_pairs(
            observations, policy="adjacent-any", min_vv_bytes=64
        )
        coords = {(pair.x, pair.y) for pair in pairs if pair.region == "north_alabama"}
        self.assertIn((2, 2), coords)
        self.assertNotIn((9, 9), coords)

    def test_no_cross_region_pairs(self):
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, _ = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=64
        )
        for pair in pairs:
            self.assertTrue(pair.pre_scene.startswith(pair.region + "_"))
            self.assertTrue(pair.post_scene.startswith(pair.region + "_"))

    def test_group_split_is_leakage_free_and_deterministic(self):
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, _ = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=64
        )
        assigned_a = finalize_assignments(
            assign_train_val_groups(pairs, val_ratio=0.5, seed=42)
        )
        assigned_b = finalize_assignments(
            assign_train_val_groups(pairs, val_ratio=0.5, seed=42)
        )
        self.assertEqual(
            [(item.output_split, item.filename) for item in assigned_a],
            [(item.output_split, item.filename) for item in assigned_b],
        )
        train_keys = {
            (item.pair.region, item.pair.x, item.pair.y)
            for item in assigned_a
            if item.output_split == "train"
        }
        val_keys = {
            (item.pair.region, item.pair.x, item.pair.y)
            for item in assigned_a
            if item.output_split == "val"
        }
        self.assertTrue(train_keys)
        self.assertTrue(val_keys)
        self.assertFalse(train_keys & val_keys)

        # The split must be seed-sensitive. With only two groups in this
        # fixture there are just two possible orderings, so any fixed pair of
        # seeds need not differ (roughly half of seed pairs agree). Instead
        # scan a bounded range and require that at least two distinct
        # labelings are reachable, proving the seed is a genuine input.
        def labeling(seed: int):
            return tuple(
                (item.output_split, item.filename)
                for item in finalize_assignments(
                    assign_train_val_groups(pairs, val_ratio=0.5, seed=seed)
                )
            )

        seen = {labeling(42)}
        for seed in range(1, 64):
            seen.add(labeling(seed))
            if len(seen) > 1:
                break
        self.assertGreater(len(seen), 1)

    def test_source_test_pairs_go_to_output_test(self):
        observations, _, _ = index_labeled_split(self.data_root, "test")
        pairs, _ = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=64
        )
        assigned = assign_all_to_split(pairs, "test")
        self.assertTrue(assigned)
        self.assertTrue(all(item.output_split == "test" for item in assigned))

    def test_test_internal_excluded(self):
        info = inspect_test_internal(self.data_root)
        self.assertTrue(info["present"])
        self.assertEqual(info["flood_label_files"], 0)
        self.assertGreater(info["vv_files"], 0)
        self.assertGreater(info["water_body_label_files"], 0)
        self.assertEqual(info["excluded_reason"], "no_flood_label")

    def test_missing_water_directory_skips_scene(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        add_scene(
            self.data_root,
            "train",
            "missing_water_dir",
            "20170101t000000",
            {(1, 1): {"vv": sar_tile(30), "flood": zero}},
        )

        observations, skips, _ = index_labeled_split(self.data_root, "train")
        records = [
            skip
            for skip in skips
            if skip.get("scene") == "missing_water_dir_20170101t000000"
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["reason"], "missing_tile_dirs")
        self.assertIn("water_body_label", records[0]["detail"])
        self.assertFalse(
            any(obs.region == "missing_water_dir" for obs in observations)
        )

    def test_unmatched_modalities_are_recorded_as_skips(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        add_scene(
            self.data_root,
            "train",
            "unmatched",
            "20170101t000000",
            {
                (1, 1): {"vv": sar_tile(31), "water": zero},
                (2, 2): {"flood": zero, "water": zero},
                (3, 3): {"vv": sar_tile(32), "flood": zero},
                (4, 4): {"water": zero},
            },
        )
        observations, skips, _ = index_labeled_split(self.data_root, "train")
        reasons = {(skip["reason"], skip["x"], skip["y"]) for skip in skips}
        self.assertIn(("vv_without_flood", 1, 1), reasons)
        self.assertIn(("flood_without_vv", 2, 2), reasons)
        self.assertIn(("vv_without_water", 3, 3), reasons)
        self.assertIn(("water_without_vv", 4, 4), reasons)
        self.assertFalse(any(obs.region == "unmatched" for obs in observations))

    def test_ambiguous_modality_keys_are_excluded(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        one = np.ones((32, 32), dtype=np.uint8)
        scene = "duplicate_keys_20170101t000000"
        add_scene(
            self.data_root,
            "train",
            "duplicate_keys",
            "20170101t000000",
            {
                (1, 1): {"vv": sar_tile(33), "flood": zero, "water": zero},
                (2, 2): {"vv": sar_tile(34), "flood": zero, "water": zero},
                (3, 3): {"vv": sar_tile(35), "flood": zero, "water": zero},
            },
        )
        tiles = self.data_root / "train" / scene / "tiles"
        write_rgb(
            tiles / "vv" / f"{scene}_x-1_y-1_VV.PNG",
            sar_tile(36),
        )
        write_flood(
            tiles / "flood_label" / f"{scene}_x-2_y-2_vv.png",
            one,
        )
        write_flood(
            tiles / "water_body_label" / f"{scene}_x-3_y-3_vv.png",
            one,
        )
        write_flood(
            tiles / "water_body_label" / f"{scene}_x-4_y-4_vh.png",
            zero,
        )
        write_flood(tiles / "water_body_label" / "malformed.png", zero)

        observations, skips, _ = index_labeled_split(self.data_root, "train")
        reasons = {skip["reason"] for skip in skips}
        self.assertIn("duplicate_vv_key", reasons)
        self.assertIn("duplicate_flood_alias", reasons)
        self.assertIn("duplicate_water_alias", reasons)
        self.assertIn("vh_water_label", reasons)
        self.assertIn("unparseable_water_name", reasons)
        self.assertFalse(
            any(obs.region == "duplicate_keys" for obs in observations)
        )

    def test_unsafe_water_label_is_not_indexed(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        scene = "unsafe_water_20170101t000000"
        add_scene(
            self.data_root,
            "train",
            "unsafe_water",
            "20170101t000000",
            {(1, 1): {"vv": sar_tile(34), "flood": zero}},
        )
        target = self.data_root / "water-target.png"
        write_flood(target, zero)
        water_dir = self.data_root / "train" / scene / "tiles" / "water_body_label"
        water_dir.mkdir(parents=True)
        (water_dir / f"{scene}_x-1_y-1.png").symlink_to(target)

        observations, skips, _ = index_labeled_split(self.data_root, "train")
        reasons = {skip["reason"] for skip in skips}
        self.assertIn("unsafe_tile_file", reasons)
        self.assertIn("vv_without_water", reasons)
        self.assertFalse(any(obs.region == "unsafe_water" for obs in observations))

    def test_scene_filename_mismatch_skipped(self):
        scene = "mismatch_20170101t000000"
        add_scene(
            self.data_root,
            "train",
            "mismatch",
            "20170101t000000",
            {
                (1, 1): {
                    "vv": sar_tile(41),
                    "flood": np.zeros((32, 32), dtype=np.uint8),
                    "water": np.zeros((32, 32), dtype=np.uint8),
                }
            },
        )
        # Plant a tile whose scene token disagrees with its scene directory.
        write_rgb(
            self.data_root / "train" / scene / "tiles" / "vv"
            / "wrong_20170202t000000_x-1_y-1_vv.png",
            sar_tile(42),
        )
        observations, skips, _ = index_labeled_split(self.data_root, "train")
        self.assertIn(
            "scene_filename_mismatch",
            {skip["reason"] for skip in skips},
        )
        self.assertTrue(
            any(obs.region == "mismatch" and (obs.x, obs.y) == (1, 1) for obs in observations)
        )

    def test_vv_qc_post_skip_recorded(self):
        add_scene(
            self.data_root,
            "train",
            "qc_region",
            "20170101t000000",
            {
                (1, 1): {
                    "vv": sar_tile(51),
                    "flood": np.zeros((32, 32), dtype=np.uint8),
                    "water": np.zeros((32, 32), dtype=np.uint8),
                }
            },
        )
        add_scene(
            self.data_root,
            "train",
            "qc_region",
            "20170202t000000",
            {
                (1, 1): {
                    "vv": np.full((32, 32), 60, dtype=np.uint8),
                    "flood": np.ones((32, 32), dtype=np.uint8),
                    "water": np.zeros((32, 32), dtype=np.uint8),
                }
            },
        )
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, skips = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=10
        )
        self.assertIn(
            "vv_qc_post", {skip["reason"] for skip in skips}
        )
        self.assertFalse(any(pair.region == "qc_region" for pair in pairs))

    def test_corrupt_post_flood_label_skips_without_aborting(self):
        scene_pre = "corrupt_20170101t000000"
        scene_post = "corrupt_20170202t000000"
        add_scene(
            self.data_root,
            "train",
            "corrupt",
            "20170101t000000",
            {
                (1, 1): {
                    "vv": sar_tile(61),
                    "flood": np.zeros((32, 32), dtype=np.uint8),
                    "water": np.zeros((32, 32), dtype=np.uint8),
                }
            },
        )
        add_scene(
            self.data_root,
            "train",
            "corrupt",
            "20170202t000000",
            {
                (1, 1): {
                    "vv": sar_tile(62),
                    "flood": np.ones((32, 32), dtype=np.uint8),
                    "water": np.zeros((32, 32), dtype=np.uint8),
                }
            },
        )
        # Overwrite the post flood label with an illegal value (128).
        Image.fromarray(
            np.full((32, 32), 128, dtype=np.uint8), mode="L"
        ).save(self.data_root / "train" / scene_post / "tiles" / "flood_label" / f"{scene_post}_x-1_y-1.png")
        clear_flood_pixel_cache()
        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, skips = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=10
        )
        self.assertIn(
            "corrupt_flood_label", {skip["reason"] for skip in skips}
        )
        self.assertFalse(any(pair.region == "corrupt" for pair in pairs))

    def test_corrupt_post_water_label_skips_without_aborting(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        one = np.ones((32, 32), dtype=np.uint8)
        scene_post = "corrupt_water_20170202t000000"
        add_scene(
            self.data_root,
            "train",
            "corrupt_water",
            "20170101t000000",
            {(1, 1): {"vv": sar_tile(63), "flood": zero, "water": zero}},
        )
        add_scene(
            self.data_root,
            "train",
            "corrupt_water",
            "20170202t000000",
            {(1, 1): {"vv": sar_tile(64), "flood": one, "water": zero}},
        )
        Image.fromarray(
            np.full((32, 32), 128, dtype=np.uint8), mode="L"
        ).save(
            self.data_root
            / "train"
            / scene_post
            / "tiles"
            / "water_body_label"
            / f"{scene_post}_x-1_y-1.png"
        )
        clear_flood_pixel_cache()

        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, skips = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=10
        )
        self.assertIn("corrupt_water_label", {skip["reason"] for skip in skips})
        self.assertFalse(any(pair.region == "corrupt_water" for pair in pairs))

    def test_post_water_size_mismatch_is_reported(self):
        zero = np.zeros((32, 32), dtype=np.uint8)
        one = np.ones((32, 32), dtype=np.uint8)
        add_scene(
            self.data_root,
            "train",
            "water_shape",
            "20170101t000000",
            {(1, 1): {"vv": sar_tile(65), "flood": zero, "water": zero}},
        )
        add_scene(
            self.data_root,
            "train",
            "water_shape",
            "20170202t000000",
            {
                (1, 1): {
                    "vv": sar_tile(66),
                    "flood": one,
                    "water": np.zeros((16, 16), dtype=np.uint8),
                }
            },
        )
        clear_flood_pixel_cache()

        observations, _, _ = index_labeled_split(self.data_root, "train")
        pairs, skips = build_temporal_pairs(
            observations, policy="nearest-flood-free", min_vv_bytes=10
        )
        self.assertIn("label_shape_mismatch", {skip["reason"] for skip in skips})
        self.assertFalse(any(pair.region == "water_shape" for pair in pairs))


class MaskAndQcTest(unittest.TestCase):
    def setUp(self):
        clear_flood_pixel_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        clear_flood_pixel_cache()
        self.tmp.cleanup()

    def test_mask_normalization_rgb_0_255_and_0_1(self):
        path_a = self.root / "a.png"
        path_b = self.root / "b.png"
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        write_flood(path_a, mask, as_rgb=True, values=(0, 255))
        write_flood(path_b, mask, as_rgb=True, values=(0, 1))
        out_a = np.asarray(load_binary_flood_mask(path_a))
        out_b = np.asarray(load_binary_flood_mask(path_b))
        self.assertEqual(load_binary_flood_mask(path_a).mode, "L")
        self.assertEqual(load_binary_water_mask(path_b).mode, "L")
        self.assertEqual(set(np.unique(out_a).tolist()), {0, 255})
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertTrue(
            np.array_equal(np.asarray(load_binary_water_mask(path_a)), out_a)
        )

    def test_full_water_mask_is_pixelwise_union(self):
        flood_path = self.root / "flood.png"
        water_path = self.root / "water.png"
        flood = np.zeros((4, 4), dtype=np.uint8)
        water = np.zeros((4, 4), dtype=np.uint8)
        flood[0, 0] = 1
        flood[1, 1] = 1
        water[1, 1] = 1
        water[2, 2] = 1
        write_flood(flood_path, flood)
        write_flood(water_path, water)

        combined = compose_full_water_mask(water_path, flood_path)
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[0, 0] = 255
        expected[1, 1] = 255
        expected[2, 2] = 255
        self.assertEqual(combined.mode, "L")
        self.assertTrue(np.array_equal(np.asarray(combined), expected))

    def test_reject_channel_mismatch_and_illegal_values(self):
        bad = self.root / "bad.png"
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[..., 0] = 255
        Image.fromarray(arr, mode="RGB").save(bad)
        with self.assertRaises(ValueError):
            load_binary_flood_mask(bad)
        with self.assertRaises(ValueError):
            load_binary_water_mask(bad)

        illegal = self.root / "illegal.png"
        Image.fromarray(np.full((4, 4), 128, dtype=np.uint8), mode="L").save(illegal)
        with self.assertRaises(ValueError):
            load_binary_flood_mask(illegal)
        with self.assertRaises(ValueError):
            load_binary_water_mask(illegal)

    def test_vv_qc_rejects_small_or_uniform(self):
        small = self.root / "small.png"
        write_rgb(small, np.zeros((32, 32), dtype=np.uint8))
        ok, reason = vv_passes_qc(small, min_vv_bytes=10_000)
        self.assertFalse(ok)
        self.assertEqual(reason, "vv_too_small")

        uniform = self.root / "uniform.png"
        write_rgb(uniform, np.full((32, 32), 40, dtype=np.uint8))
        ok, reason = vv_passes_qc(uniform, min_vv_bytes=10)
        self.assertFalse(ok)
        self.assertEqual(reason, "vv_uniform")


class CliIntegrationTest(unittest.TestCase):
    def setUp(self):
        clear_flood_pixel_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_root = build_standard_fixture(self.root)
        self.output = self.root / "prepared"

    def tearDown(self):
        clear_flood_pixel_cache()
        self.tmp.cleanup()

    def _args(self, **overrides):
        argv = [
            "--source",
            str(self.root),
            "--output",
            str(self.output),
            "--pair-policy",
            overrides.get("pair_policy", "nearest-flood-free"),
            "--val-ratio",
            str(overrides.get("val_ratio", 0.5)),
            "--seed",
            str(overrides.get("seed", 42)),
            "--mode",
            overrides.get("mode", "hardlink"),
            "--min-vv-bytes",
            str(overrides.get("min_vv_bytes", 64)),
            "--max-saturated-fraction",
            str(overrides.get("max_saturated_fraction", 0.999)),
        ]
        if overrides.get("dry_run"):
            argv.append("--dry-run")
        if overrides.get("keep_negative_post") is False:
            argv.append("--no-keep-negative-post")
        return parse_args(argv)

    def test_dry_run_writes_nothing(self):
        prepare(self._args(dry_run=True))
        self.assertFalse(self.output.exists())
        staging = self.output.parent / f".{self.output.name}.partial"
        self.assertFalse(staging.exists())

    def test_download_cache_symlinks_are_trusted_within_repository(self):
        repository_cache = self.root / "hub" / "datasets--owner--etci"
        snapshot = repository_cache / "snapshots" / "revision"
        downloaded_data_root = build_standard_fixture(snapshot)
        replace_tiles_with_hub_cache_symlinks(
            downloaded_data_root,
            repository_cache,
        )
        download_output = self.root / "downloaded-prepared"
        args = parse_args(
            [
                "--download",
                "--output",
                str(download_output),
                "--val-ratio",
                "0.5",
                "--min-vv-bytes",
                "64",
            ]
        )

        with patch(
            "huggingface_hub.snapshot_download",
            return_value=str(snapshot),
        ) as snapshot_download:
            prepare(args)

        snapshot_download.assert_called_once()
        self.assertTrue(download_output.is_dir())
        with (download_output / "pair_manifest.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for row in rows:
            for column in (
                "pre_vv",
                "post_vv",
                "pre_flood",
                "post_flood",
                "pre_water",
                "post_water",
            ):
                self.assertFalse(Path(row[column]).is_absolute())
                self.assertNotIn("blobs", Path(row[column]).parts)
        first = rows[0]
        materialized = download_output / first["split"] / "A" / first["filename"]
        self.assertTrue(materialized.is_file())
        self.assertFalse(materialized.is_symlink())

    def test_refuse_existing_output_and_staging(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            prepare(self._args())
        self.output.rmdir()
        staging = self.output.parent / f".{self.output.name}.partial"
        staging.mkdir()
        with self.assertRaises(FileExistsError):
            prepare(self._args())

    def test_materialization_failure_cleans_staging_directory(self):
        staging = self.output.parent / f".{self.output.name}.partial"
        with patch(
            "prepare_etci_pairs._materialize_water_gt",
            side_effect=RuntimeError("water materialization failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "water materialization failed"):
                prepare(self._args())

        self.assertFalse(self.output.exists())
        self.assertFalse(staging.exists())
        prepare(self._args())
        self.assertTrue(self.output.is_dir())

    def test_materialize_hardlink_manifests_and_loader_smoke(self):
        prepare(self._args(mode="hardlink", min_vv_bytes=64, val_ratio=0.5))
        self.assertTrue(self.output.exists())
        staging = self.output.parent / f".{self.output.name}.partial"
        self.assertFalse(staging.exists())

        for split_name in ("train", "val", "test"):
            names = {
                subdirectory: sorted(
                    path.name
                    for path in (self.output / split_name / subdirectory).glob("*.png")
                )
                for subdirectory in (
                    "A",
                    "B",
                    "GT",
                    "WATER_GT_A",
                    "WATER_GT_B",
                )
            }
            names_a = names["A"]
            self.assertTrue(names_a)
            self.assertTrue(all(value == names_a for value in names.values()))
            for name in names_a:
                for subdirectory in ("GT", "WATER_GT_A", "WATER_GT_B"):
                    with Image.open(
                        self.output / split_name / subdirectory / name
                    ) as mask:
                        self.assertEqual(mask.mode, "L")
                        self.assertTrue(set(mask.getdata()).issubset({0, 255}))

            dest = self.output / split_name / "A" / names_a[0]
            with Image.open(dest) as image:
                self.assertEqual(image.mode, "RGB")
            with (self.output / "pair_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            row = next(row for row in rows if row["filename"] == names_a[0])
            src = self.data_root / row["pre_vv"]
            if src.exists():
                try:
                    self.assertTrue(os.path.samefile(src, dest))
                except OSError:
                    pass

            expected_water_a = compose_full_water_mask(
                self.data_root / row["pre_water"],
                self.data_root / row["pre_flood"],
            )
            expected_water_b = compose_full_water_mask(
                self.data_root / row["post_water"],
                self.data_root / row["post_flood"],
            )
            with Image.open(
                self.output / split_name / "WATER_GT_A" / names_a[0]
            ) as actual_water_a:
                self.assertTrue(
                    np.array_equal(
                        np.asarray(actual_water_a),
                        np.asarray(expected_water_a),
                    )
                )
            with Image.open(
                self.output / split_name / "WATER_GT_B" / names_a[0]
            ) as actual_water_b:
                self.assertTrue(
                    np.array_equal(
                        np.asarray(actual_water_b),
                        np.asarray(expected_water_b),
                    )
                )
            self.assertEqual(
                int(row["water_gt_a_pixels"]),
                int((np.asarray(expected_water_a) > 0).sum()),
            )
            self.assertEqual(
                int(row["water_gt_b_pixels"]),
                int((np.asarray(expected_water_b) > 0).sum()),
            )

        with (self.output / "pair_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertIn("policy", rows[0])
        self.assertIn("pre_water", rows[0])
        self.assertIn("post_water", rows[0])
        self.assertIn("water_gt_a_pixels", rows[0])
        self.assertIn("water_gt_b_pixels", rows[0])
        self.assertTrue(
            all(
                row["water_gt_formula"]
                == "water_body_label OR flood_label, per acquisition date"
                for row in rows
            )
        )
        with (self.output / "split_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            split_rows = list(csv.DictReader(handle))
        self.assertEqual(len(split_rows), len(rows))
        self.assertTrue((self.output / "qc_report.json").is_file())
        self.assertTrue((self.output / "conversion_metadata.json").is_file())
        self.assertTrue((self.output / "skipped_records.jsonl").is_file())

        metadata = json.loads(
            (self.output / "conversion_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["converter_version"], "1.1.0")
        self.assertEqual(
            metadata["output_subdirectories"],
            ["A", "B", "GT", "WATER_GT_A", "WATER_GT_B"],
        )
        self.assertEqual(metadata["gt_semantics"], "post-date flood_label")
        self.assertEqual(
            metadata["water_gt_formula"],
            "water_body_label OR flood_label, per acquisition date",
        )
        self.assertEqual(metadata["water_supervised_counts"], metadata["counts"])
        self.assertIn("test_internal excluded", metadata["split_strategy"])

        qc = json.loads((self.output / "qc_report.json").read_text(encoding="utf-8"))
        self.assertEqual(
            qc["water_supervised_pairs_per_output_split"],
            qc["pairs_per_output_split"],
        )
        self.assertGreater(qc["index_stats"]["train"]["water_files"], 0)
        self.assertGreater(qc["index_stats"]["test"]["water_files"], 0)
        self.assertGreater(qc["test_internal"]["water_body_label_files"], 0)

        from utils.dataloaders import FloodDetection, test_path, train_path

        data_dir = str(self.output) + os.sep
        train_full, val_full = train_path(data_dir)
        test_full = test_path(data_dir)
        train_ds = FloodDetection(train_full, aug=False, include_water=True)
        val_ds = FloodDetection(val_full, aug=False, include_water=True)
        test_ds = FloodDetection(test_full, aug=False, include_water=True)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(test_ds), 0)

        img1, img2, targets, name = train_ds[0]
        self.assertEqual(tuple(img1.shape)[0], 3)
        self.assertEqual(tuple(img2.shape)[0], 3)
        self.assertTrue(targets["water_valid"].item())
        for target_name in ("change", "water_a", "water_b"):
            self.assertEqual(targets[target_name].ndim, 2)
            self.assertTrue(
                set(np.unique(targets[target_name].numpy())).issubset({0.0, 1.0})
            )
        self.assertTrue(name.startswith("etci_"))

    def test_main_error_prefix(self):
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "--source",
                    str(self.root),
                    "--output",
                    str(self.output),
                    "--download",
                ]
            )
        self.assertIn("Error:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
