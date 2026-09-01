from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from water_seg import pretrain_geoid
from water_seg.engine import (GEOID_INPUT_CONTRACT,
                              GEOID_NORMALIZATION_CONTRACT)
from water_seg.geoid_dataset import GEOID_RADIOMETRY


class TinyWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(2, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)
        self.register_buffer('channel_mean', torch.zeros(1, 2, 1, 1))
        self.register_buffer('channel_std', torch.ones(1, 2, 1, 1))

    def set_channel_normalization(self, means, stds):
        self.channel_mean.copy_(torch.tensor(means).view(1, 2, 1, 1))
        self.channel_std.copy_(torch.tensor(stds).view(1, 2, 1, 1))
        return self

    def forward(self, image):
        image = (image - self.channel_mean) / self.channel_std
        return self.head(self.encoder(image))


class FakeGEOIDIndex:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata_path = self.root / 'data_tiles_s256_st128.csv'
        self.metadata_path.write_text('synthetic metadata', encoding='utf-8')
        self.min_valid_proportion = 0.01

    def counts(self):
        return {'train': 2, 'val': 1, 'samples': 3, 'pre': 1, 'post': 2}


def _loader():
    images = torch.tensor([
        [[[-20.0, -10.0], [-10.0, -20.0]]],
        [[[-10.0, -20.0], [-20.0, -10.0]]],
    ]).repeat(1, 2, 1, 1)
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
                'water_seg.pretrain_geoid._load_geoid_channel_constants',
                return_value=([-15.0, -20.0], [5.0, 4.0]),
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
        self.assertEqual(checkpoint['format_version'], 3)
        self.assertEqual(checkpoint['config']['bands'], ['VV', 'VH'])
        self.assertEqual(checkpoint['config']['input'], GEOID_INPUT_CONTRACT)
        self.assertEqual(
            checkpoint['config']['normalization'],
            GEOID_NORMALIZATION_CONTRACT,
        )
        self.assertEqual(
            checkpoint['config']['radiometry'],
            GEOID_RADIOMETRY,
        )
        self.assertNotIn('db_min', checkpoint['config'])
        self.assertNotIn('db_max', checkpoint['config'])
        self.assertEqual(checkpoint['config']['ignore_index'], 255)
        self.assertFalse(checkpoint['config']['distributed'])
        self.assertEqual(checkpoint['config']['world_size'], 1)
        self.assertEqual(checkpoint['config']['global_batch_size'], 2)
        self.assertEqual(
            checkpoint['config']['samples_per_split'],
            {'train': 2, 'val': 1},
        )

    def test_resume_starts_at_next_epoch(self):
        metrics = {
            'loss': 0.5,
            'overall_accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'water_iou': 0.5,
            'samples': 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'geoid-flood'
            root.mkdir()
            save_dir = Path(directory) / 'run'
            index = FakeGEOIDIndex(root)
            args = [
                '--geoid-root', str(root),
                '--epochs', '3',
                '--batch-size', '2',
                '--device', 'cpu',
                '--no-imagenet-pretrained',
                '--no-augmentation',
                '--no-progress',
                '--early-stopping-patience', '0',
                '--save-dir', str(save_dir),
            ]

            def patches(run_side_effect):
                return (
                    patch(
                        'water_seg.pretrain_geoid.build_geoid_water_index',
                        return_value=index,
                    ),
                    patch(
                        'water_seg.pretrain_geoid._load_geoid_channel_constants',
                        return_value=([-15.0, -20.0], [5.0, 4.0]),
                    ),
                    patch(
                        'water_seg.pretrain_geoid.SwinTinyUNet',
                        side_effect=lambda **kwargs: TinyWaterModel(),
                    ),
                    patch(
                        'water_seg.pretrain_geoid.get_geoid_water_loaders',
                        return_value=(object(), object()),
                    ),
                    patch(
                        'water_seg.pretrain_geoid.validate_geoid_files',
                        return_value={'s1grd_files': 2, 'label_files': 1},
                    ),
                    patch(
                        'water_seg.pretrain_geoid.run_epoch',
                        side_effect=run_side_effect,
                    ),
                )

            first = patches([metrics, metrics, RuntimeError('interrupt')])
            with first[0], first[1], first[2], first[3], first[4], first[5]:
                with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                    pretrain_geoid.main(args)
            interrupted = torch.load(
                save_dir / 'last.pth',
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(interrupted['epoch'], 1)

            second = patches([metrics, metrics, metrics, metrics])
            with second[0], second[1], second[2], second[3], second[4], \
                    second[5] as resumed_epoch:
                pretrain_geoid.main(
                    args + ['--resume', str(save_dir / 'last.pth')]
                )
            resumed = torch.load(
                save_dir / 'last.pth',
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(resumed_epoch.call_count, 4)
            self.assertEqual(resumed['epoch'], 3)
            self.assertEqual(resumed['config']['resume']['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
