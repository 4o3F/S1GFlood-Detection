from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from utils import transforms as temporal_transforms
from utils.dataloaders import FloodChange, FloodDetection, _build_split_dataset


def _write_rgb(path: Path, size=(8, 8), pattern=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern is None:
        gray = np.full((size[1], size[0]), 96, dtype=np.uint8)
    else:
        gray = np.asarray(pattern, dtype=np.uint8)
    rgb = np.stack((gray, gray, gray), axis=-1)
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
) -> None:
    _write_rgb(root / split / 'A' / name, size=size)
    _write_rgb(root / split / 'B' / name, size=size)
    _write_mask(root / split / 'GT' / name, 0, size=size)


class OptionalWaterLabelDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, include_water=True):
        records = _build_split_dataset(str(self.root) + os.sep, 'train')
        return FloodDetection(
            records,
            flag='train',
            aug=False,
            include_water=include_water,
        )

    def test_sparse_labels_collate_without_fabricating_supervision(self):
        _make_required_sample(self.root, 'labeled.png')
        _make_required_sample(self.root, 'unlabeled.png')
        _write_mask(
            self.root / 'train' / 'WATER_GT_A' / 'labeled.png',
            1,
        )
        _write_mask(
            self.root / 'train' / 'WATER_GT_B' / 'labeled.png',
            255,
        )

        loader = DataLoader(self._dataset(), batch_size=2, shuffle=False)
        _, _, targets, names = next(iter(loader))

        self.assertEqual(list(names), ['labeled.png', 'unlabeled.png'])
        torch.testing.assert_close(
            targets['water_valid'],
            torch.tensor([True, False]),
        )
        self.assertTrue(torch.all(targets['water_a'][0] == 1))
        self.assertTrue(torch.all(targets['water_b'][0] == 1))
        self.assertTrue(torch.all(targets['water_a'][1] == 0))
        self.assertTrue(torch.all(targets['water_b'][1] == 0))

    def test_disabled_auxiliary_loading_skips_water_mask_io(self):
        _make_required_sample(self.root, 'sample.png')
        for subdirectory in ('WATER_GT_A', 'WATER_GT_B'):
            _write_mask(
                self.root / 'train' / subdirectory / 'sample.png',
                2,
            )
        records = _build_split_dataset(str(self.root), 'train')
        dataset = FloodDetection(
            records,
            flag='train',
            aug=False,
            include_water=True,
            load_water_labels=False,
        )

        _, _, targets, _ = dataset[0]
        self.assertEqual(set(targets), {'change', 'water_valid'})
        self.assertFalse(targets['water_valid'].item())

    def test_zero_weight_loaders_do_not_materialize_water_targets(self):
        for split in ('train', 'val'):
            _make_required_sample(
                self.root,
                f'{split}.png',
                split=split,
            )
            for subdirectory in ('WATER_GT_A', 'WATER_GT_B'):
                _write_mask(
                    self.root / split / subdirectory / f'{split}.png',
                    2,
                )
        options = SimpleNamespace(
            dataset_dir=str(self.root) + os.sep,
            augmentation=False,
            batch_size=1,
            num_workers=0,
            water_loss_weight=0.0,
        )

        from utils.helpers import get_loaders

        train_loader, val_loader = get_loaders(options)
        for loader in (train_loader, val_loader):
            _, _, targets, _ = next(iter(loader))
            self.assertEqual(set(targets), {'change', 'water_valid'})
            self.assertFalse(targets['water_valid'].any().item())

    def test_legacy_flood_change_positional_call_remains_available(self):
        _make_required_sample(self.root, 'sample.png')
        split_root = self.root / 'train'
        sample = FloodChange(
            [str(split_root) + os.sep, 'sample.png'],
            str(split_root / 'GT' / 'sample.png'),
            False,
        )
        self.assertEqual(len(sample), 4)
        self.assertEqual(sample[3], 'sample.png')

    def test_legacy_four_item_contract_remains_available(self):
        _make_required_sample(self.root, 'sample.png')
        sample = self._dataset(include_water=False)[0]
        self.assertEqual(len(sample), 4)
        image_a, image_b, change_gt, name = sample
        self.assertEqual(tuple(image_a.shape), (3, 8, 8))
        self.assertEqual(tuple(image_b.shape), (3, 8, 8))
        self.assertEqual(tuple(change_gt.shape), (8, 8))
        self.assertEqual(name, 'sample.png')

    def test_water_directories_must_exist_as_a_pair(self):
        _make_required_sample(self.root, 'sample.png')
        _write_mask(
            self.root / 'train' / 'WATER_GT_A' / 'sample.png',
            0,
        )
        with self.assertRaisesRegex(FileNotFoundError, 'exist as a pair'):
            _build_split_dataset(str(self.root), 'train')

    def test_water_files_must_exist_as_pairs(self):
        _make_required_sample(self.root, 'first.png')
        _make_required_sample(self.root, 'second.png')
        _write_mask(
            self.root / 'train' / 'WATER_GT_A' / 'first.png',
            0,
        )
        _write_mask(
            self.root / 'train' / 'WATER_GT_A' / 'second.png',
            0,
        )
        _write_mask(
            self.root / 'train' / 'WATER_GT_B' / 'first.png',
            0,
        )
        with self.assertRaisesRegex(FileNotFoundError, 'exist as A/B pairs'):
            _build_split_dataset(str(self.root), 'train')

    def test_orphan_water_labels_are_rejected(self):
        _make_required_sample(self.root, 'sample.png')
        for subdirectory in ('WATER_GT_A', 'WATER_GT_B'):
            _write_mask(
                self.root / 'train' / subdirectory / 'orphan.png',
                0,
            )
        with self.assertRaisesRegex(ValueError, 'orphan water label'):
            _build_split_dataset(str(self.root), 'train')

    def test_zero_one_and_zero_255_masks_are_normalized_identically(self):
        _make_required_sample(self.root, 'sample.png')
        pattern = np.zeros((8, 8), dtype=np.uint8)
        pattern[1:4, 2:6] = 1
        _write_mask(
            self.root / 'train' / 'WATER_GT_A' / 'sample.png',
            pattern,
        )
        _write_mask(
            self.root / 'train' / 'WATER_GT_B' / 'sample.png',
            pattern * 255,
        )

        _, _, targets, _ = self._dataset()[0]
        torch.testing.assert_close(targets['water_a'], targets['water_b'])
        self.assertEqual(set(torch.unique(targets['water_a']).tolist()), {0.0, 1.0})

    def test_illegal_mask_values_are_rejected(self):
        _make_required_sample(self.root, 'sample.png')
        _write_mask(self.root / 'train' / 'GT' / 'sample.png', 2)
        with self.assertRaisesRegex(ValueError, 'only .*0,1,255'):
            self._dataset(include_water=False)[0]

    def test_image_and_label_size_mismatch_is_rejected(self):
        _make_required_sample(self.root, 'sample.png')
        _write_mask(
            self.root / 'train' / 'GT' / 'sample.png',
            0,
            size=(4, 4),
        )
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self._dataset(include_water=False)[0]


class SynchronizedTemporalTransformTest(unittest.TestCase):
    def setUp(self):
        pattern = np.zeros((4, 4), dtype=np.uint8)
        pattern[0, 1] = 255
        pattern[2, 3] = 255
        rgb = np.stack((pattern, pattern, pattern), axis=-1)
        self.sample = {
            'image': (
                Image.fromarray(rgb, mode='RGB'),
                Image.fromarray(rgb, mode='RGB'),
            ),
            'targets': {
                'change': Image.fromarray(pattern, mode='L'),
                'water_a': Image.fromarray(pattern, mode='L'),
                'water_b': Image.fromarray(pattern, mode='L'),
            },
        }

    def _assert_aligned(self, transformed):
        tensor_sample = temporal_transforms.ToTensor()(transformed)
        image_mask = tensor_sample['image'][0][0] > 0
        for target in tensor_sample['targets'].values():
            torch.testing.assert_close(image_mask, target.bool())

    def test_horizontal_flip_applies_to_every_image_and_mask(self):
        with patch('utils.transforms.random.random', return_value=0.0):
            transformed = temporal_transforms.RandomHorizontalFlip()(self.sample)
        self._assert_aligned(transformed)

    def test_vertical_flip_applies_to_every_image_and_mask(self):
        with patch('utils.transforms.random.random', return_value=0.0):
            transformed = temporal_transforms.RandomVerticalFlip()(self.sample)
        self._assert_aligned(transformed)

    def test_every_fixed_rotation_applies_to_every_image_and_mask(self):
        for rotation in (
            Image.ROTATE_90,
            Image.ROTATE_180,
            Image.ROTATE_270,
        ):
            with self.subTest(rotation=rotation), patch(
                'utils.transforms.random.random',
                return_value=0.0,
            ), patch(
                'utils.transforms.random.choice',
                return_value=rotation,
            ):
                transformed = temporal_transforms.RandomFixRotate()(self.sample)
            self._assert_aligned(transformed)


if __name__ == '__main__':
    unittest.main()
