from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from water_seg import pretrain_geoid


class TinyWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)
        self.register_buffer('vv_mean', torch.zeros(1, 1, 1, 1))
        self.register_buffer('vv_std', torch.ones(1, 1, 1, 1))

    def set_vv_normalization(self, mean, std):
        self.vv_mean.fill_(float(mean))
        self.vv_std.fill_(float(std))
        return self

    def forward(self, image):
        image = (image - self.vv_mean) / self.vv_std
        return self.head(self.encoder(image))


class FakeGEOIDIndex:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata_path = self.root / 'data_tiles_s256_st128.csv'
        self.metadata_path.write_text('synthetic metadata', encoding='utf-8')
        self.db_min = -25.0
        self.db_max = 0.0
        self.min_valid_proportion = 0.01

    def counts(self):
        return {'train': 2, 'val': 1, 'samples': 3, 'pre': 1, 'post': 2}


def _loader():
    images = torch.tensor([
        [[[-20.0, -10.0], [-10.0, -20.0]]],
        [[[-10.0, -20.0], [-20.0, -10.0]]],
    ])
    targets = torch.tensor([
        [[0, 1], [255, 0]],
        [[1, 0], [0, 1]],
    ])
    records = [
        (images[index], targets[index], f'geoid-{index}')
        for index in range(images.size(0))
    ]
    return DataLoader(records, batch_size=2, shuffle=False)


class GEOIDPretrainCliTest(unittest.TestCase):
    def test_parser_requires_geoid_root(self):
        with self.assertRaises(SystemExit):
            pretrain_geoid.build_parser().parse_args([])

    def test_main_writes_transfer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'geoid-flood'
            root.mkdir()
            save_dir = Path(directory) / 'run'
            index = FakeGEOIDIndex(root)
            loader = _loader()
            with patch(
                'water_seg.pretrain_geoid.build_geoid_water_index',
                return_value=index,
            ), patch(
                'water_seg.pretrain_geoid._load_geoid_vv_constants',
                return_value=(-15.0, 5.0),
            ), patch(
                'water_seg.pretrain_geoid.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.pretrain_geoid.get_geoid_water_loaders',
                return_value=(loader, loader),
            ), patch(
                'water_seg.pretrain_geoid.validate_geoid_files',
                return_value={'s1grd_files': 2, 'label_files': 1},
            ):
                best_path = pretrain_geoid.main([
                    '--geoid-root', str(root),
                    '--epochs', '1',
                    '--batch-size', '2',
                    '--device', 'cpu',
                    '--no-imagenet-pretrained',
                    '--no-augmentation',
                    '--no-progress',
                    '--early-stopping-patience', '0',
                    '--save-dir', str(save_dir),
                ])

            checkpoint = torch.load(
                best_path,
                map_location='cpu',
                weights_only=True,
            )

        self.assertEqual(checkpoint['kind'], 'geoid-water-pretraining')
        self.assertEqual(checkpoint['format_version'], 1)
        self.assertEqual(checkpoint['config']['band'], 'VV')
        self.assertEqual(checkpoint['config']['ignore_index'], 255)
        self.assertFalse(checkpoint['config']['distributed'])
        self.assertEqual(checkpoint['config']['world_size'], 1)
        self.assertEqual(checkpoint['config']['global_batch_size'], 2)
        self.assertEqual(
            checkpoint['config']['samples_per_split'],
            {'train': 2, 'val': 1},
        )


if __name__ == '__main__':
    unittest.main()
