from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from utils.dataloaders import _build_split_dataset
from water_seg.dataset import (
    RandomD4,
    SingleTemporalWaterDataset,
    build_water_dataset,
    flatten_water_records,
    get_water_loaders,
    get_water_test_loader,
)


def _write_rgb(path: Path, size=(8, 8), pattern=None, channels=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern is None:
        gray = np.full((size[1], size[0]), 96, dtype=np.uint8)
    else:
        gray = np.asarray(pattern, dtype=np.uint8)
    if channels is None:
        rgb = np.stack((gray, gray, gray), axis=-1)
    else:
        rgb = np.stack(channels, axis=-1).astype(np.uint8)
    Image.fromarray(rgb, mode='RGB').save(path)


def _write_mask(path: Path, values, size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.uint8)
    if array.ndim == 0:
        array = np.full((size[1], size[0]), array, dtype=np.uint8)
    Image.fromarray(array, mode='L').save(path)


def _make_required_sample(
    root: Path,
    name: str,
    size=(8, 8),
    split='train',
    pattern=None,
) -> None:
    _write_rgb(root / split / 'A' / name, size=size, pattern=pattern)
    _write_rgb(root / split / 'B' / name, size=size, pattern=pattern)
    _write_mask(root / split / 'GT' / name, 0, size=size)


def _write_water_pair(root: Path, name: str, values_a, values_b, split='train', size=(8, 8)):
    _write_mask(root / split / 'WATER_GT_A' / name, values_a, size=size)
    _write_mask(root / split / 'WATER_GT_B' / name, values_b, size=size)


class FlattenWaterRecordsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labeled_pair_flattens_to_a_then_b_with_stable_names(self):
        _make_required_sample(self.root, 'sample.png')
        pattern_a = np.zeros((8, 8), dtype=np.uint8)
        pattern_a[1:3, 2:5] = 1
        pattern_b = np.zeros((8, 8), dtype=np.uint8)
        pattern_b[4:7, 0:3] = 255
        _write_water_pair(self.root, 'sample.png', pattern_a, pattern_b)

        records = flatten_water_records(
            _build_split_dataset(str(self.root), 'train')
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(
            [record['name'] for record in records],
            ['sample.png#A', 'sample.png#B'],
        )
        self.assertEqual(Path(records[0]['image']).parent.name, 'A')
        self.assertEqual(Path(records[1]['image']).parent.name, 'B')
        self.assertEqual(Path(records[0]['mask']).parent.name, 'WATER_GT_A')
        self.assertEqual(Path(records[1]['mask']).parent.name, 'WATER_GT_B')
        self.assertEqual(Path(records[0]['image']).name, 'sample.png')
        self.assertEqual(Path(records[1]['mask']).name, 'sample.png')

        dataset = SingleTemporalWaterDataset(records, augment=False)
        image_a, mask_a, name_a = dataset[0]
        image_b, mask_b, name_b = dataset[1]
        self.assertEqual(name_a, 'sample.png#A')
        self.assertEqual(name_b, 'sample.png#B')
        self.assertEqual(tuple(image_a.shape), (1, 8, 8))
        self.assertEqual(image_a.dtype, torch.float32)
        self.assertEqual(mask_a.dtype, torch.int64)
        self.assertTrue(torch.all(image_a == 96))
        torch.testing.assert_close(
            mask_a,
            torch.from_numpy((pattern_a > 0).astype(np.int64)),
        )
        torch.testing.assert_close(
            mask_b,
            torch.from_numpy((pattern_b > 0).astype(np.int64)),
        )

    def test_sparse_unlabeled_records_are_omitted(self):
        _make_required_sample(self.root, 'labeled.png')
        _make_required_sample(self.root, 'unlabeled.png')
        _write_water_pair(self.root, 'labeled.png', 1, 255)

        records = flatten_water_records(
            _build_split_dataset(str(self.root), 'train')
        )
        self.assertEqual(
            [record['name'] for record in records],
            ['labeled.png#A', 'labeled.png#B'],
        )

    def test_no_water_labels_raises_requirement_error(self):
        _make_required_sample(self.root, 'sample.png')
        with self.assertRaisesRegex(FileNotFoundError, 'WATER_GT_A'):
            flatten_water_records(_build_split_dataset(str(self.root), 'train'))


class SingleTemporalWaterDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mask_values_one_and_255_normalize_identically(self):
        _make_required_sample(self.root, 'sample.png')
        pattern = np.zeros((8, 8), dtype=np.uint8)
        pattern[1:4, 2:6] = 1
        _write_water_pair(self.root, 'sample.png', pattern, pattern * 255)

        dataset = build_water_dataset(str(self.root), 'train', augment=False)
        _, mask_one, _ = dataset[0]
        _, mask_255, _ = dataset[1]
        torch.testing.assert_close(mask_one, mask_255)
        self.assertEqual(set(torch.unique(mask_one).tolist()), {0, 1})
        self.assertEqual(mask_one.dtype, torch.int64)

    def test_rgb_mismatch_raises(self):
        _make_required_sample(self.root, 'sample.png')
        _write_water_pair(self.root, 'sample.png', 1, 1)
        height, width = 8, 8
        channels = (
            np.full((height, width), 10, dtype=np.uint8),
            np.full((height, width), 20, dtype=np.uint8),
            np.full((height, width), 30, dtype=np.uint8),
        )
        _write_rgb(
            self.root / 'train' / 'A' / 'sample.png',
            channels=channels,
        )
        dataset = build_water_dataset(str(self.root), 'train', augment=False)
        with self.assertRaisesRegex(ValueError, 'identical'):
            dataset[0]

    def test_size_mismatch_raises(self):
        _make_required_sample(self.root, 'sample.png')
        _write_water_pair(self.root, 'sample.png', 1, 1, size=(4, 4))
        dataset = build_water_dataset(str(self.root), 'train', augment=False)
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            dataset[0]

    def test_d4_augmentation_rejects_non_square_patches(self):
        _make_required_sample(self.root, 'sample.png', size=(8, 4))
        _write_water_pair(
            self.root,
            'sample.png',
            1,
            1,
            size=(8, 4),
        )
        dataset = build_water_dataset(str(self.root), 'train', augment=True)
        with self.assertRaisesRegex(ValueError, 'requires a square'):
            dataset[0]

    def test_loader_batch_shapes(self):
        for split in ('train', 'val', 'test'):
            _make_required_sample(self.root, f'{split}.png', split=split)
            _write_water_pair(
                self.root,
                f'{split}.png',
                1,
                255,
                split=split,
            )

        train_loader, val_loader = get_water_loaders(
            str(self.root),
            batch_size=2,
            num_workers=0,
            augmentation=False,
        )
        test_loader = get_water_test_loader(
            str(self.root),
            batch_size=2,
            num_workers=0,
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)
        self.assertFalse(train_loader.drop_last)
        self.assertFalse(val_loader.drop_last)
        self.assertFalse(test_loader.drop_last)

        for loader in (train_loader, val_loader, test_loader):
            images, masks, names = next(iter(loader))
            self.assertEqual(tuple(images.shape), (2, 1, 8, 8))
            self.assertEqual(tuple(masks.shape), (2, 8, 8))
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(masks.dtype, torch.int64)
            self.assertEqual(len(names), 2)

    def test_missing_paired_water_directory_is_rejected(self):
        _make_required_sample(self.root, 'sample.png')
        _write_mask(self.root / 'train' / 'WATER_GT_A' / 'sample.png', 1)
        with self.assertRaisesRegex(FileNotFoundError, 'exist as a pair'):
            build_water_dataset(str(self.root), 'train')


class RandomD4Test(unittest.TestCase):
    def test_alignment_for_all_eight_dihedral_choices(self):
        pattern = np.zeros((4, 4), dtype=np.uint8)
        pattern[0, 1] = 180
        image = Image.fromarray(pattern, mode='L')
        mask = Image.fromarray(pattern, mode='L')
        transform = RandomD4()
        results = []

        for choice in range(8):
            with self.subTest(choice=choice), patch(
                'water_seg.dataset.random.randrange',
                return_value=choice,
            ):
                out_image, out_mask = transform(image, mask)
            image_array = np.asarray(out_image)
            mask_array = np.asarray(out_mask)
            np.testing.assert_array_equal(image_array, mask_array)
            results.append(image_array.tobytes())

        self.assertEqual(len(set(results)), 8)


if __name__ == '__main__':
    unittest.main()
