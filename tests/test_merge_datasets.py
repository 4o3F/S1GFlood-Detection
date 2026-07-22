from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from merge_datasets import main, merge, parse_args, resolve_root


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
    gray = rng.integers(20, 220, size=(16, 16), dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_gt(path: Path, flood: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = 255 if flood else 0
    Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").save(path)


def make_root(parent: Path, tag: str, split_files: dict) -> Path:
    """Create a prepared root: {split}/{A,B,GT}/<name>.png for the given names."""
    root = parent / tag
    for split, names in split_files.items():
        for name in names:
            _write_rgb(root / split / "A" / name)
            _write_rgb(root / split / "B" / name)
            _write_gt(root / split / "GT" / name, flood=False)
    return root


class MergeDatasetsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "merged"

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, inputs, **overrides):
        argv = []
        for src in inputs:
            argv += ["--input", str(src)]
        argv += ["--output", str(self.output)]
        argv += ["--mode", overrides.get("mode", "hardlink")]
        if overrides.get("dry_run"):
            argv.append("--dry-run")
        if "on_collision" in overrides:
            argv += ["--on-collision", overrides["on_collision"]]
        for tag in overrides.get("tags", []):
            argv += ["--tag", tag]
        return parse_args(argv)

    def test_resolve_root_accepts_prepared_and_data_wrapped(self):
        prepared = make_root(self.root, "p", {"train": ["a.png"]})
        self.assertEqual(resolve_root(prepared), prepared.resolve())
        wrapped = self.root / "w"
        make_root(wrapped, "data", {"train": ["a.png"]})  # creates w/data/train/A/...
        self.assertEqual(resolve_root(wrapped).name, "data")

    def test_merge_combines_counts_and_loader_sees_both(self):
        a = make_root(self.root, "s1", {"train": ["a1.png", "a2.png"], "val": ["v1.png"]})
        b = make_root(self.root, "etci", {"train": ["e1.png"], "val": ["v2.png"], "test": ["t1.png"]})
        merge(self._args([a, b]))

        for split, expected in (("train", 3), ("val", 2), ("test", 1)):
            names = sorted(p.name for p in (self.output / split / "A").glob("*.png"))
            for sub in ("A", "B", "GT"):
                self.assertEqual(
                    sorted(p.name for p in (self.output / split / sub).glob("*.png")),
                    names,
                )
            self.assertEqual(len(names), expected)

        manifest = json.loads((self.output / "merge_manifest.json").read_text())
        self.assertEqual(manifest["counts_per_split"]["train"], 3)
        self.assertEqual(manifest["counts_per_source"]["s1"]["train"], 2)
        self.assertEqual(manifest["counts_per_source"]["etci"]["train"], 1)
        self.assertEqual(manifest["collisions_detected"], 0)

        # hardlink: same inode for source and merged
        src = a / "train" / "A" / "a1.png"
        dst = self.output / "train" / "A" / "a1.png"
        try:
            self.assertTrue(os.path.samefile(src, dst))
        except OSError:
            pass

        # loader smoke: train_path lists both sources' files
        from utils.dataloaders import FloodDetection, train_path

        data_dir = str(self.output) + os.sep
        train_full, val_full = train_path(data_dir)
        self.assertEqual(len(train_full), 3)
        self.assertEqual(len(val_full), 2)
        ds = FloodDetection(train_full, aug=False)
        img1, img2, mask, name = ds[0]
        self.assertEqual(tuple(img1.shape)[0], 3)
        self.assertEqual(mask.ndim, 2)

    def test_dry_run_writes_nothing(self):
        a = make_root(self.root, "s1", {"train": ["a1.png"]})
        b = make_root(self.root, "etci", {"train": ["e1.png"]})
        result = merge(self._args([a, b], dry_run=True))
        self.assertTrue(result["dry_run"])
        self.assertFalse(self.output.exists())
        staging = self.output.parent / f".{self.output.name}.partial"
        self.assertFalse(staging.exists())

    def test_refuse_existing_output_and_staging(self):
        a = make_root(self.root, "s1", {"train": ["a1.png"]})
        b = make_root(self.root, "etci", {"train": ["e1.png"]})
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            merge(self._args([a, b]))
        self.output.rmdir()
        staging = self.output.parent / f".{self.output.name}.partial"
        staging.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            merge(self._args([a, b]))

    def test_collision_detected_in_fail_mode(self):
        a = make_root(self.root, "s1", {"train": ["shared.png"]})
        b = make_root(self.root, "etci", {"train": ["shared.png"]})
        with self.assertRaises(ValueError):
            merge(self._args([a, b], on_collision="fail"))

    def test_collision_rename_mode_prefixes_by_tag(self):
        a = make_root(self.root, "s1", {"train": ["shared.png", "x.png"]})
        b = make_root(self.root, "etci", {"train": ["shared.png"]})
        merge(self._args([a, b], on_collision="rename", tags=["s1", "etci"]))
        names = sorted(p.name for p in (self.output / "train" / "A").glob("*.png"))
        self.assertEqual(
            names,
            ["etci__shared.png", "s1__shared.png", "s1__x.png"],
        )
        manifest = json.loads((self.output / "merge_manifest.json").read_text())
        self.assertEqual(manifest["collisions_detected"], 0)
        self.assertEqual(manifest["counts_per_split"]["train"], 3)

    def test_main_error_prefix(self):
        a = make_root(self.root, "s1", {"train": ["shared.png"]})
        b = make_root(self.root, "etci", {"train": ["shared.png"]})
        with self.assertRaises(SystemExit) as ctx:
            main(
                ["--input", str(a), "--input", str(b),
                 "--output", str(self.output)]
            )
        self.assertIn("Error:", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
