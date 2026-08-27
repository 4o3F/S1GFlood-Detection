from __future__ import annotations

import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from rasterio.transform import Affine
from torch.utils.data import RandomSampler, SequentialSampler
import rasterio

from utils.kulsary_raster import (
    PATCH_SIZE,
    LazySigma0Stack,
    linear_sigma0_to_clipped_db,
    sampled_file_fingerprint,
    tile_slice,
)
from utils.kulsary_temporal import (
    ROLES,
    TileKey,
    assign_spatial_blocks,
    build_role_name,
)
from water_seg.dataset import (
    KulsaryRawWaterDataset,
    RandomD4,
    build_kulsary_scene_index,
    compute_train_channel_stats,
    get_water_loaders,
    get_water_test_loader,
    resolve_sigma0_paths,
    SIGMA0_MANIFEST_FILENAME,
    SIGMA0_ROOT_FILENAMES,
    water_worker_init_fn,
)


SCENE_SIZE = 768
TRANSFORM = Affine(0.001, 0, 53.0, 0, -0.001, 48.0)
MASK_NAMES = {
    'before': '1_water_before_20240402.png',
    'peak': '2_water_during_20240414.png',
    'after': '3_water_after_20240426.png',
}
ROLE_LINEAR = {
    'before': np.float32(10.0 ** -2.5),
    'peak': np.float32(10.0 ** -1.25),
    'after': np.float32(1.0),
}
ROLE_DB = {
    'before': -25.0,
    'peak': -12.5,
    'after': 0.0,
}


def write_mask(path: Path, array: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(array, 255, 0).astype(np.uint8)).save(path)
    center_c = transform.c + transform.a / 2.0 + transform.b / 2.0
    center_f = transform.f + transform.d / 2.0 + transform.e / 2.0
    path.with_suffix('.pgw').write_text(
        '\n'.join(
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
        + '\n',
        encoding='utf-8',
    )


def write_sigma0(path: Path, array: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(array, dtype=np.float32)
    if source.ndim == 2:
        source = np.stack((source, source), axis=0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=source.shape[2],
        height=source.shape[1],
        count=2,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=0.0,
    ) as dataset:
        dataset.write(source)
        dataset.set_band_description(1, 'Sigma0_VV')
        dataset.set_band_description(2, 'Sigma0_VH')


def distinctive_water_masks(size: int = SCENE_SIZE) -> dict[str, np.ndarray]:
    water = {
        'before': np.zeros((size, size), dtype=bool),
        'peak': np.zeros((size, size), dtype=bool),
        'after': np.zeros((size, size), dtype=bool),
    }
    water['before'][16:80, 24:120] = True
    water['peak'][16:80, 24:120] = True
    water['peak'][200:320, 200:360] = True
    water['after'][400:520, 80:200] = True
    water['after'][16:80, 24:120] = True
    return water


def all_tiles():
    return [
        TileKey(row, col)
        for row in range(SCENE_SIZE // PATCH_SIZE)
        for col in range(SCENE_SIZE // PATCH_SIZE)
    ]


class KulsarySceneFixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sigma0_paths = {
            role: self.root / f'{role}_sigma0_vv_vh.tif' for role in ROLES
        }
        self.mask_source = self.root / 'masks'
        self.mask_source.mkdir()
        self.water = distinctive_water_masks()
        self._write_default_scene()
        self._open_datasets = []

    def tearDown(self):
        for dataset in self._open_datasets:
            try:
                dataset.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def _track(self, dataset):
        self._open_datasets.append(dataset)
        return dataset

    def _write_default_scene(self, arrays=None, water=None):
        if arrays is None:
            arrays = {
                role: np.full((SCENE_SIZE, SCENE_SIZE), value, dtype=np.float32)
                for role, value in ROLE_LINEAR.items()
            }
        if water is None:
            water = self.water
        for role, array in arrays.items():
            write_sigma0(self.sigma0_paths[role], array, TRANSFORM)
        for role, mask in water.items():
            write_mask(self.mask_source / MASK_NAMES[role], mask, TRANSFORM)

    def _build_index(self, **kwargs):
        return build_kulsary_scene_index(
            self.sigma0_paths['before'],
            self.sigma0_paths['peak'],
            self.sigma0_paths['after'],
            self.mask_source,
            **kwargs,
        )

    def _dataset(self, index, split, augment=False):
        return self._track(
            KulsaryRawWaterDataset(index, split, augment=augment)
        )


class Sigma0RootResolutionTest(KulsarySceneFixtureTest):
    def _write_sigma0_root(self):
        sigma0_root = self.root / 'sigma0-root'
        sigma0_root.mkdir()
        roles = {}
        for role, filename in SIGMA0_ROOT_FILENAMES.items():
            destination = sigma0_root / filename
            destination.write_bytes(self.sigma0_paths[role].read_bytes())
            roles[role] = {
                'output_filename': filename,
                'fingerprint': sampled_file_fingerprint(destination),
            }
        (sigma0_root / SIGMA0_MANIFEST_FILENAME).write_text(
            json.dumps({
                'format': 'kulsary-sigma0',
                'version': '2.0.0',
                'polarizations': ['VV', 'VH'],
                'band_order': {'VV': 1, 'VH': 2},
                'roles': roles,
            }),
            encoding='utf-8',
        )
        return sigma0_root

    def test_resolves_canonical_sigma0_root_and_manifest(self):
        sigma0_root = self._write_sigma0_root()
        paths, provenance = resolve_sigma0_paths(sigma0_root=sigma0_root)
        self.assertEqual(
            paths,
            {
                role: (sigma0_root / filename).resolve()
                for role, filename in SIGMA0_ROOT_FILENAMES.items()
            },
        )
        self.assertEqual(provenance['input_mode'], 'sigma0-root')
        self.assertEqual(provenance['sigma0_manifest_version'], '2.0.0')

    def test_explicit_paths_remain_supported(self):
        paths, provenance = resolve_sigma0_paths(
            sigma0_before=self.sigma0_paths['before'],
            sigma0_peak=self.sigma0_paths['peak'],
            sigma0_after=self.sigma0_paths['after'],
        )
        self.assertEqual(set(paths), set(ROLES))
        self.assertEqual(provenance['input_mode'], 'explicit')

    def test_root_and_partial_explicit_inputs_are_rejected(self):
        sigma0_root = self._write_sigma0_root()
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            resolve_sigma0_paths(
                sigma0_root=sigma0_root,
                sigma0_before=self.sigma0_paths['before'],
            )
        with self.assertRaisesRegex(ValueError, 'provide --sigma0-root'):
            resolve_sigma0_paths(
                sigma0_before=self.sigma0_paths['before'],
            )

    def test_manifest_fingerprint_mismatch_is_rejected(self):
        sigma0_root = self._write_sigma0_root()
        (sigma0_root / SIGMA0_ROOT_FILENAMES['peak']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            resolve_sigma0_paths(sigma0_root=sigma0_root)


class SceneIndexTest(KulsarySceneFixtureTest):
    def test_index_keeps_valid_tiles_and_emits_three_unique_roles(self):
        index = self._build_index()
        expected_tiles = all_tiles()
        self.assertEqual(index.kept_tiles, expected_tiles)
        self.assertEqual(len(index.samples), 3 * len(expected_tiles))
        self.assertEqual(index.counts()['samples'], 3 * len(expected_tiles))
        self.assertEqual(index.counts()['tiles'], len(expected_tiles))

        names = [sample.name for sample in index.samples]
        self.assertEqual(len(names), len(set(names)))
        for tile in expected_tiles:
            roles = [
                sample.role for sample in index.samples if sample.tile == tile
            ]
            self.assertEqual(sorted(roles), ['after', 'before', 'peak'])
            self.assertEqual(roles.count('peak'), 1)
            for role in ROLES:
                expected_name = build_role_name(tile, role)
                match = [
                    sample
                    for sample in index.samples
                    if sample.tile == tile and sample.role == role
                ]
                self.assertEqual(len(match), 1)
                self.assertEqual(match[0].name, expected_name)

        peak_names = [
            sample.name for sample in index.samples if sample.role == 'peak'
        ]
        self.assertEqual(len(peak_names), len(expected_tiles))
        self.assertEqual(
            [sample.name for sample in index.samples_for('train')],
            [sample.name for sample in index.samples if sample.output_split == 'train'],
        )

    def test_split_assignments_match_assign_spatial_blocks(self):
        index = self._build_index(split_seed=42, block_tiles=2)
        expected = assign_spatial_blocks(
            index.kept_tiles,
            block_tiles=index.block_tiles,
            train_ratio=index.train_ratio,
            val_ratio=index.val_ratio,
            test_ratio=index.test_ratio,
            seed=index.split_seed,
        )
        self.assertEqual(index.split_by_tile, expected)
        self.assertEqual(set(expected.values()), {'train', 'val', 'test'})

        splits_by_tile = {}
        for sample in index.samples:
            splits_by_tile.setdefault(sample.tile, set()).add(sample.output_split)
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_tile.values()))
        for tile, split_name in expected.items():
            self.assertEqual(splits_by_tile[tile], {split_name})


class DatasetOutputTest(KulsarySceneFixtureTest):
    def test_shapes_dtypes_db_anchors_and_mask_roles(self):
        index = self._build_index()
        for split in ('train', 'val', 'test'):
            dataset = self._dataset(index, split, augment=False)
            self.assertGreater(len(dataset), 0)
            for sample_index, sample in enumerate(dataset.samples):
                image, mask, name = dataset[sample_index]
                self.assertEqual(name, sample.name)
                self.assertEqual(tuple(image.shape), (2, PATCH_SIZE, PATCH_SIZE))
                self.assertEqual(tuple(mask.shape), (PATCH_SIZE, PATCH_SIZE))
                self.assertEqual(image.dtype, torch.float32)
                self.assertEqual(mask.dtype, torch.int64)
                torch.testing.assert_close(
                    image,
                    torch.full(
                        (2, PATCH_SIZE, PATCH_SIZE),
                        ROLE_DB[sample.role],
                        dtype=torch.float32,
                    ),
                    atol=1e-4,
                    rtol=0,
                )
                expected_mask = index.water_masks[sample.role][
                    tile_slice(sample.tile)
                ]
                torch.testing.assert_close(
                    mask,
                    torch.from_numpy(np.asarray(expected_mask, dtype=np.int64)),
                )
                self.assertTrue(set(torch.unique(mask).tolist()).issubset({0, 1}))

        for split_name, dataset_dir in (
            ('A', self.root / 'A'),
            ('B', self.root / 'B'),
            ('GT', self.root / 'GT'),
            ('WATER_GT', self.root / 'WATER_GT'),
            ('WATER_GT_A', self.root / 'WATER_GT_A'),
            ('WATER_GT_B', self.root / 'WATER_GT_B'),
            ('train', self.root / 'train'),
            ('val', self.root / 'val'),
            ('test', self.root / 'test'),
        ):
            self.assertFalse(dataset_dir.exists(), split_name)

    def test_val_and_test_never_apply_d4(self):
        index = self._build_index()
        train_loader, val_loader = get_water_loaders(
            index,
            batch_size=2,
            num_workers=0,
            augmentation=True,
        )
        test_loader = get_water_test_loader(index, batch_size=2, num_workers=0)
        self._track(train_loader.dataset)
        self._track(val_loader.dataset)
        self._track(test_loader.dataset)
        self.assertTrue(train_loader.dataset.augment)
        self.assertFalse(val_loader.dataset.augment)
        self.assertFalse(test_loader.dataset.augment)

        val_dataset = val_loader.dataset
        with patch.object(RandomD4, '__call__', side_effect=AssertionError('D4')):
            image, mask, name = val_dataset[0]
        self.assertEqual(tuple(image.shape), (2, PATCH_SIZE, PATCH_SIZE))
        with patch.object(RandomD4, '__call__', side_effect=AssertionError('D4')):
            test_loader.dataset[0]

    def test_getitem_reads_only_requested_role(self):
        index = self._build_index()
        dataset = self._dataset(index, 'train', augment=False)
        sample = dataset.samples[0]
        with patch.object(
            LazySigma0Stack,
            'read',
            side_effect=AssertionError('must not read all roles'),
        ):
            image, mask, name = dataset[0]
        self.assertEqual(name, sample.name)
        self.assertEqual(tuple(image.shape), (2, PATCH_SIZE, PATCH_SIZE))


class RandomD4Test(unittest.TestCase):
    def test_all_eight_dihedral_choices_are_aligned_and_distinct(self):
        pattern = np.zeros((4, 4), dtype=np.float32)
        pattern[0, 1] = 180.0
        mask = (pattern > 0).astype(np.int64)
        transform = RandomD4()
        results = []

        for choice in range(8):
            with self.subTest(choice=choice), patch(
                'water_seg.dataset.random.randrange',
                return_value=choice,
            ):
                out_image, out_mask = transform(pattern, mask)
            self.assertEqual(out_image.shape, (4, 4))
            self.assertEqual(out_mask.shape, (4, 4))
            np.testing.assert_array_equal(out_image != 0, out_mask.astype(bool))
            results.append(np.ascontiguousarray(out_image).tobytes())

        self.assertEqual(len(set(results)), 8)

    def test_dual_channel_transform_uses_the_same_spatial_operation(self):
        vv = np.zeros((4, 4), dtype=np.float32)
        vv[0, 1] = 1.0
        vh = vv * 7.0
        image = np.stack((vv, vh), axis=0)
        mask = (vv > 0).astype(np.int64)

        for choice in range(8):
            with self.subTest(choice=choice), patch(
                'water_seg.dataset.random.randrange',
                return_value=choice,
            ):
                out_image, out_mask = RandomD4()(image, mask)
            self.assertEqual(out_image.shape, (2, 4, 4))
            self.assertEqual(out_mask.shape, (4, 4))
            np.testing.assert_array_equal(
                out_image[0] != 0,
                out_mask.astype(bool),
            )
            np.testing.assert_array_equal(out_image[1], out_image[0] * 7.0)

    def test_non_square_arrays_are_rejected(self):
        image = np.zeros((4, 5), dtype=np.float32)
        mask = np.zeros((4, 5), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'square'):
            RandomD4()(image, mask)


class TrainStatsTest(KulsarySceneFixtureTest):
    def test_mean_std_use_train_records_only(self):
        index = self._build_index()
        arrays = {
            role: np.full((SCENE_SIZE, SCENE_SIZE), 0.1, dtype=np.float32)
            for role in ROLES
        }
        train_linear = {
            'before': np.float32(0.01),
            'peak': np.float32(0.1),
            'after': np.float32(10.0 ** -1.25),
        }
        extreme_linear = np.float32(1.0)
        for tile, split_name in index.split_by_tile.items():
            sl = tile_slice(tile)
            for role in ROLES:
                arrays[role][sl] = (
                    train_linear[role]
                    if split_name == 'train'
                    else extreme_linear
                )
        for role, array in arrays.items():
            write_sigma0(self.sigma0_paths[role], array, TRANSFORM)

        means, stds = compute_train_channel_stats(index)
        train_db = np.array(
            [
                float(linear_sigma0_to_clipped_db(
                    np.array([[train_linear[role]]], dtype=np.float32),
                    np.array([[True]], dtype=bool),
                    index.db_min,
                    index.db_max,
                )[0, 0])
                for role in ROLES
            ],
            dtype=np.float64,
        )
        expected_mean = float(train_db.mean())
        expected_std = float(train_db.std())
        self.assertGreater(expected_std, 0.0)
        self.assertEqual(len(means), 2)
        self.assertEqual(len(stds), 2)
        for mean, std in zip(means, stds):
            self.assertAlmostEqual(mean, expected_mean, places=4)
            self.assertAlmostEqual(std, expected_std, places=4)

        all_values = []
        for split_name in index.split_by_tile.values():
            if split_name == 'train':
                all_values.extend(train_db.tolist())
            else:
                all_values.extend([0.0, 0.0, 0.0])
        mixed_mean = float(np.mean(all_values))
        self.assertNotAlmostEqual(means[0], mixed_mean, places=2)


class LoaderTest(KulsarySceneFixtureTest):
    def test_sampler_drop_last_and_batch_shapes(self):
        index = self._build_index()
        train_loader, val_loader = get_water_loaders(
            index,
            batch_size=2,
            num_workers=0,
            augmentation=False,
        )
        test_loader = get_water_test_loader(index, batch_size=2, num_workers=0)
        self._track(train_loader.dataset)
        self._track(val_loader.dataset)
        self._track(test_loader.dataset)

        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)
        self.assertFalse(train_loader.drop_last)
        self.assertFalse(val_loader.drop_last)
        self.assertFalse(test_loader.drop_last)

        for loader in (train_loader, val_loader, test_loader):
            images, masks, names = next(iter(loader))
            self.assertEqual(images.shape[0], min(2, len(loader.dataset)))
            self.assertEqual(tuple(images.shape[1:]), (2, PATCH_SIZE, PATCH_SIZE))
            self.assertEqual(tuple(masks.shape[1:]), (PATCH_SIZE, PATCH_SIZE))
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(masks.dtype, torch.int64)
            self.assertEqual(len(names), images.shape[0])

    def test_pickle_reopen_and_worker_init_fn(self):
        index = self._build_index()
        dataset = self._dataset(index, 'train', augment=False)
        image, mask, name = dataset[0]
        self.assertIsNotNone(dataset._stack._stack)

        payload = pickle.dumps(dataset)
        self.assertIsNone(dataset._stack._stack)
        restored = pickle.loads(payload)
        self._track(restored)
        self.assertIsNone(restored._stack._stack)
        image2, mask2, name2 = restored[0]
        self.assertEqual(name2, name)
        torch.testing.assert_close(image, image2)
        torch.testing.assert_close(mask, mask2)

        train_loader, val_loader = get_water_loaders(
            index,
            batch_size=2,
            num_workers=1,
            augmentation=False,
        )
        self._track(train_loader.dataset)
        self._track(val_loader.dataset)
        self.assertIs(train_loader.worker_init_fn, water_worker_init_fn)
        self.assertIs(val_loader.worker_init_fn, water_worker_init_fn)
        worker_images, worker_masks, worker_names = next(iter(train_loader))
        self.assertEqual(tuple(worker_images.shape[1:]), (2, PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(tuple(worker_masks.shape[1:]), (PATCH_SIZE, PATCH_SIZE))
        self.assertEqual(len(worker_names), worker_images.shape[0])

        opened = train_loader.dataset
        opened[0]
        self.assertIsNotNone(opened._stack._stack)

        class _Info:
            dataset = opened

        with patch(
            'torch.utils.data.get_worker_info',
            return_value=_Info,
        ), patch('torch.initial_seed', return_value=123456789):
            water_worker_init_fn(0)
        self.assertIsNone(opened._stack._stack)
        self.assertEqual(random.getrandbits(32), random.Random(123456789).getrandbits(32))


class ErrorPathTest(KulsarySceneFixtureTest):
    def test_missing_files_and_masks_raise_clearly(self):
        missing = self.root / 'missing_peak.tif'
        with self.assertRaisesRegex(FileNotFoundError, 'peak'):
            build_kulsary_scene_index(
                self.sigma0_paths['before'],
                missing,
                self.sigma0_paths['after'],
                self.mask_source,
            )

        missing_dir = self.root / 'no-masks'
        with self.assertRaisesRegex(FileNotFoundError, 'mask source'):
            build_kulsary_scene_index(
                self.sigma0_paths['before'],
                self.sigma0_paths['peak'],
                self.sigma0_paths['after'],
                missing_dir,
            )

        (self.mask_source / MASK_NAMES['after']).unlink()
        with self.assertRaisesRegex(ValueError, 'after'):
            self._build_index()

    def test_invalid_db_range_raises(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            self._build_index(db_min=float('nan'))
        with self.assertRaisesRegex(ValueError, 'smaller'):
            self._build_index(db_min=0.0, db_max=-25.0)

    def test_empty_valid_tiles_raise(self):
        arrays = {
            role: np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
            for role in ROLES
        }
        self._write_default_scene(arrays=arrays)
        with self.assertRaisesRegex(ValueError, 'no fully valid'):
            self._build_index()


class ConverterSemanticsTest(KulsarySceneFixtureTest):
    def test_unique_role_masks_and_clipped_db_match_converter_mapping(self):
        index = self._build_index()
        self.assertEqual(len(index.kept_tiles), 9)
        for tile in index.kept_tiles:
            roles = [
                sample.role for sample in index.samples if sample.tile == tile
            ]
            self.assertEqual(roles.count('before'), 1)
            self.assertEqual(roles.count('peak'), 1)
            self.assertEqual(roles.count('after'), 1)

        dataset = self._dataset(index, 'train', augment=False)
        seen = set()
        for sample_index, sample in enumerate(dataset.samples):
            if sample.role in seen:
                continue
            seen.add(sample.role)
            image, mask, name = dataset[sample_index]
            self.assertEqual(name, sample.name)
            self.assertTrue(name.endswith(f'#{sample.role}'))
            db = image.numpy()[0]
            np.testing.assert_allclose(db, ROLE_DB[sample.role], atol=1e-4)
            scaled = (db - index.db_min) / (index.db_max - index.db_min) * 255.0
            expected_intensity = {
                'before': 0.0,
                'peak': 127.5,
                'after': 255.0,
            }[sample.role]
            np.testing.assert_allclose(scaled, expected_intensity, atol=1e-2)
            np.testing.assert_array_equal(
                mask.numpy().astype(bool),
                index.water_masks[sample.role][tile_slice(sample.tile)],
            )
        self.assertEqual(seen, set(ROLES))


if __name__ == '__main__':
    unittest.main()
