import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from water_seg.geoid_dataset import (GEOIDRawWaterDataset,
                                     GEOID_IGNORE_INDEX,
                                     build_geoid_water_index,
                                     compute_geoid_train_channel_stats,
                                     get_geoid_water_loaders,
                                     validate_geoid_files)


CSV_FIELDS = [
    'tile_id',
    'event_id',
    'modality',
    'x',
    'y',
    'size',
    'event_date',
    'image_time',
    'cloud_cover',
    'positive_proportion',
    'valid_proportion',
    'acquisition_date',
    'label_id',
    'split',
]


def _write_raster(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        transform=from_origin(0, 256, 1, 1),
        crs='EPSG:4326',
    ) as target:
        target.write(data)


def _row(tile_id, image_time, split):
    return {
        'tile_id': tile_id,
        'event_id': 'EMSR100',
        'modality': 's1grd',
        'x': 0,
        'y': 0,
        'size': 256,
        'event_date': '2024-01-02',
        'image_time': image_time,
        'cloud_cover': 0,
        'positive_proportion': 0.25,
        'valid_proportion': 0.99,
        'acquisition_date': '2024-01-01T00:00:00',
        'label_id': 'EMSR100-1-2_label',
        'split': split,
    }


class GEOIDWaterDatasetTest(unittest.TestCase):
    def _build_tree(self, root):
        root = Path(root)
        event = root / 'EMSR100-1'
        label = np.zeros((256, 256), dtype=np.uint8)
        label[0, 0] = 0
        label[0, 1] = 1
        label[0, 2] = 2
        label[0, 3] = 255
        _write_raster(event / 'label' / 'EMSR100-1-2_label.tif', label)

        sigma0 = np.full((2, 256, 256), 0.01, dtype=np.float32)
        sigma0[0, 128:, :] = 0.1
        sigma0[1, :128, :] = 0.001
        sigma0[0, 0, 4] = 0.0
        rows = []
        for tile_id, image_time, split in (
            ('EMSR100-1-2_s1grd_pre_20240101T000000', 'pre', 'train'),
            ('EMSR100-1-2_s1grd_post_20240103T000000', 'post', 'train'),
            ('EMSR100-1-2_s1grd_post_20240104T000000', 'post', 'val'),
        ):
            _write_raster(event / 's1grd' / f'{tile_id}.tif', sigma0)
            rows.append(_row(tile_id, image_time, split))
        with (root / 'data_tiles_s256_st128.csv').open(
            'w',
            encoding='utf-8',
            newline='',
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return root

    def test_official_csv_windows_and_time_specific_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(directory)
            index = build_geoid_water_index(root)
            self.assertEqual(
                validate_geoid_files(index),
                {'s1grd_files': 3, 'label_files': 1},
            )
            self.assertEqual(
                index.counts(),
                {'train': 2, 'val': 1, 'samples': 3, 'pre': 1, 'post': 2},
            )
            train = GEOIDRawWaterDataset(index, 'train')
            pre_image, pre_mask, pre_name = train[0]
            post_image, post_mask, _ = train[1]

        self.assertEqual(tuple(pre_image.shape), (2, 256, 256))
        self.assertIn('__x0_y0_s256', pre_name)
        self.assertEqual(pre_mask[0, :5].tolist(), [0, 1, 0, 255, 255])
        self.assertEqual(post_mask[0, :5].tolist(), [0, 1, 1, 255, 255])
        self.assertAlmostEqual(float(pre_image[0, 1, 1]), -20.0, places=5)
        self.assertAlmostEqual(float(pre_image[1, 1, 1]), -25.0, places=5)
        self.assertAlmostEqual(float(post_image[0, 200, 1]), -10.0, places=5)

    def test_stats_and_loaders_use_only_supervised_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            index = build_geoid_water_index(self._build_tree(directory))
            means, stds = compute_geoid_train_channel_stats(
                index,
                progress=False,
            )
            train_loader, val_loader = get_geoid_water_loaders(
                index,
                batch_size=2,
                num_workers=0,
                augmentation=False,
            )
            images, masks, names = next(iter(train_loader))

        self.assertAlmostEqual(means[0], -15.0, places=3)
        self.assertAlmostEqual(stds[0], 5.0, places=3)
        self.assertAlmostEqual(means[1], -22.5, places=3)
        self.assertAlmostEqual(stds[1], 2.5, places=3)
        self.assertEqual(tuple(images.shape), (2, 2, 256, 256))
        self.assertEqual(tuple(masks.shape), (2, 256, 256))
        self.assertEqual(len(names), 2)
        self.assertTrue(bool((masks == GEOID_IGNORE_INDEX).any()))
        self.assertEqual(len(val_loader.dataset), 1)

    def test_grouped_stats_preserve_overlapping_window_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(directory)
            event = root / 'EMSR100-1'
            sigma0 = np.full((2, 256, 384), 0.01, dtype=np.float32)
            sigma0[0, :, 128:256] = 10 ** -1.5
            sigma0[0, :, 256:] = 0.1
            sigma0[1, :, :128] = 0.001
            sigma0[1, :, 128:256] = 10 ** -2.5
            label = np.zeros((256, 384), dtype=np.uint8)
            _write_raster(
                event / 'label' / 'EMSR100-1-2_label.tif',
                label,
            )
            for image_path in (event / 's1grd').glob('*.tif'):
                _write_raster(image_path, sigma0)

            metadata_path = root / 'data_tiles_s256_st128.csv'
            with metadata_path.open(
                'r',
                encoding='utf-8',
                newline='',
            ) as stream:
                rows = list(csv.DictReader(stream))
            overlap = dict(rows[0])
            overlap['x'] = 128
            rows.append(overlap)
            with metadata_path.open(
                'w',
                encoding='utf-8',
                newline='',
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            index = build_geoid_water_index(root)
            means, stds = compute_geoid_train_channel_stats(
                index,
                progress=False,
            )
            dataset = GEOIDRawWaterDataset(index, 'train')
            records = [
                dataset[sample_index]
                for sample_index in range(len(dataset))
            ]
            values = [
                torch.cat([
                    image[channel][mask != GEOID_IGNORE_INDEX]
                    for image, mask, _ in records
                ])
                for channel in range(2)
            ]

        for channel in range(2):
            self.assertAlmostEqual(
                means[channel],
                float(values[channel].mean()),
                places=5,
            )
            self.assertAlmostEqual(
                stds[channel],
                float(values[channel].std(correction=0)),
                places=5,
            )

    def test_invalid_raw_label_value_is_rejected_at_read_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(directory)
            label_path = (
                root / 'EMSR100-1' / 'label' / 'EMSR100-1-2_label.tif'
            )
            with rasterio.open(label_path, 'r+') as target:
                label = target.read(1)
                label[10, 10] = 3
                target.write(label, 1)
            index = build_geoid_water_index(root)
            with self.assertRaisesRegex(ValueError, 'unsupported values'):
                GEOIDRawWaterDataset(index, 'train')[0]

    def test_metadata_rejects_unsafe_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(directory)
            metadata_path = root / 'data_tiles_s256_st128.csv'
            with metadata_path.open(
                'r',
                encoding='utf-8',
                newline='',
            ) as stream:
                rows = list(csv.DictReader(stream))
            rows[0]['tile_id'] = '../outside'
            with metadata_path.open(
                'w',
                encoding='utf-8',
                newline='',
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'unsafe tile_id'):
                build_geoid_water_index(root)

    def test_file_validation_reports_incomplete_partial_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._build_tree(directory)
            index = build_geoid_water_index(root)
            index.samples[0].image_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, '1 images'):
                validate_geoid_files(index)


if __name__ == '__main__':
    unittest.main()
