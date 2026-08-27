from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from rasterio.transform import Affine
from rasterio.windows import Window
from torch.utils.data import DataLoader

from utils.kulsary_raster import CommonGrid
from utils.kulsary_temporal import TileKey
from water_seg import eval as water_eval
from water_seg import train as water_train


class StaticScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


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


class FakeIndex:
    def __init__(self, root):
        root = Path(root)
        self.sigma0_paths = {
            'before': (root / 'before.tif').resolve(),
            'peak': (root / 'peak.tif').resolve(),
            'after': (root / 'after.tif').resolve(),
        }
        self.mask_source = (root / 'masks').resolve()
        self.mask_source.mkdir(parents=True, exist_ok=True)
        for role, path in self.sigma0_paths.items():
            path.write_bytes(f'sigma0-{role}'.encode('ascii'))
        self.mask_files = {
            role: self.mask_source / f'{role}.png'
            for role in ('before', 'peak', 'after')
        }
        for role, path in self.mask_files.items():
            path.write_bytes(f'mask-{role}'.encode('ascii'))
        self.db_min = -25.0
        self.db_max = 0.0
        self.split_seed = 42
        self.block_tiles = 2
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.kept_tiles = [TileKey(0, 0), TileKey(0, 1), TileKey(1, 0)]
        self.split_by_tile = {
            self.kept_tiles[0]: 'train',
            self.kept_tiles[1]: 'val',
            self.kept_tiles[2]: 'test',
        }
        self.grid = CommonGrid(
            crs='EPSG:32639',
            transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
            width=768,
            height=768,
            peak_window=Window(0, 0, 768, 768),
            bounds=(500000.0, 5192320.0, 507680.0, 5200000.0),
        )
        self._counts = {'train': 3, 'val': 3, 'test': 3}

    def samples_for(self, split):
        return [object()] * self._counts[split]


def _loader():
    images = torch.tensor([
        [[[-25.0, -10.0], [-10.0, -25.0]]],
        [[[-10.0, -25.0], [-25.0, -10.0]]],
    ]).repeat(1, 2, 1, 1)
    targets = torch.tensor([
        [[0, 1], [1, 0]],
        [[1, 0], [0, 1]],
    ])
    records = [
        (images[index], targets[index], f'sample-{index}')
        for index in range(images.size(0))
    ]
    return DataLoader(records, batch_size=2, shuffle=False)


def _data_args(directory):
    return [
        '--sigma0-before', str(Path(directory) / 'before.tif'),
        '--sigma0-peak', str(Path(directory) / 'peak.tif'),
        '--sigma0-after', str(Path(directory) / 'after.tif'),
        '--mask-source', str(Path(directory) / 'masks'),
    ]


class WaterCliTest(unittest.TestCase):
    def test_train_and_eval_main_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory) / 'run'
            loader = _loader()
            fake_index = FakeIndex(directory)
            with patch(
                'water_seg.train.build_kulsary_scene_index',
                return_value=fake_index,
            ), patch(
                'water_seg.train.compute_train_channel_stats',
                return_value=([-15.0, -20.0], [5.0, 4.0]),
            ), patch(
                'water_seg.train.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.train.get_water_loaders',
                return_value=(loader, loader),
            ):
                best_path = water_train.main(
                    _data_args(directory) + [
                        '--epochs', '1',
                        '--batch-size', '2',
                        '--num-workers', '0',
                        '--device', 'cpu',
                        '--no-imagenet-pretrained',
                        '--no-augmentation',
                        '--early-stopping-patience', '0',
                        '--save-dir', str(save_dir),
                    ]
                )

            self.assertEqual(best_path, save_dir / 'best.pth')
            self.assertTrue((save_dir / 'best.pth').is_file())
            self.assertTrue((save_dir / 'last.pth').is_file())
            checkpoint = torch.load(
                save_dir / 'best.pth',
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(checkpoint['format_version'], 3)
            self.assertEqual(
                checkpoint['config']['channel_mean'],
                [-15.0, -20.0],
            )
            self.assertFalse(checkpoint['config']['distributed'])
            self.assertEqual(checkpoint['config']['world_size'], 1)
            self.assertEqual(checkpoint['config']['global_batch_size'], 2)
            self.assertEqual(
                checkpoint['config']['samples_per_split'],
                {'train': 3, 'val': 3, 'test': 3},
            )

            with patch(
                'water_seg.eval.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.eval.build_kulsary_scene_index',
                return_value=fake_index,
            ) as build_index, patch(
                'water_seg.eval.get_water_test_loader',
                return_value=loader,
            ):
                metrics = water_eval.main(
                    _data_args(directory) + [
                        '--path', str(save_dir / 'best.pth'),
                        '--batch-size', '2',
                        '--num-workers', '0',
                        '--device', 'cpu',
                    ]
                )

            self.assertEqual(metrics['samples'], 2)
            self.assertGreaterEqual(metrics['water_iou'], 0.0)
            self.assertLessEqual(metrics['water_iou'], 1.0)
            build_kwargs = build_index.call_args.kwargs
            self.assertEqual(build_kwargs['db_min'], -25.0)
            self.assertEqual(build_kwargs['split_seed'], 42)

    def test_early_stopping_preserves_best_checkpoint(self):
        train_metrics = {
            'loss': 0.5,
            'overall_accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'water_iou': 0.5,
            'samples': 2,
        }
        validation_metrics = [
            dict(train_metrics, water_iou=0.5),
            dict(train_metrics, water_iou=0.4),
            dict(train_metrics, water_iou=0.3),
        ]
        side_effects = []
        for metrics in validation_metrics:
            side_effects.extend((train_metrics, metrics))

        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory) / 'run'
            fake_index = FakeIndex(directory)
            with patch(
                'water_seg.train.build_kulsary_scene_index',
                return_value=fake_index,
            ), patch(
                'water_seg.train.compute_train_channel_stats',
                return_value=([-15.0, -20.0], [5.0, 4.0]),
            ), patch(
                'water_seg.train.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.train.get_water_loaders',
                return_value=(object(), object()),
            ), patch(
                'water_seg.train.run_epoch',
                side_effect=side_effects,
            ), patch(
                'water_seg.train.torch.optim.lr_scheduler.CosineAnnealingLR',
                StaticScheduler,
            ):
                water_train.main(
                    _data_args(directory) + [
                        '--epochs', '10',
                        '--batch-size', '2',
                        '--num-workers', '0',
                        '--device', 'cpu',
                        '--no-imagenet-pretrained',
                        '--no-augmentation',
                        '--early-stopping-patience', '2',
                        '--save-dir', str(save_dir),
                    ]
                )

            best = torch.load(
                save_dir / 'best.pth',
                map_location='cpu',
                weights_only=True,
            )
            last = torch.load(
                save_dir / 'last.pth',
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 3)
            self.assertEqual(last['best_water_iou'], 0.5)

    def test_resume_restores_full_state_and_starts_at_next_epoch(self):
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
            save_dir = Path(directory) / 'run'
            fake_index = FakeIndex(directory)
            args = _data_args(directory) + [
                '--epochs', '3',
                '--batch-size', '2',
                '--num-workers', '0',
                '--device', 'cpu',
                '--no-imagenet-pretrained',
                '--no-augmentation',
                '--no-progress',
                '--early-stopping-patience', '0',
                '--save-dir', str(save_dir),
            ]
            common_patches = (
                patch(
                    'water_seg.train.build_kulsary_scene_index',
                    return_value=fake_index,
                ),
                patch(
                    'water_seg.train.compute_train_channel_stats',
                    return_value=([-15.0, -20.0], [5.0, 4.0]),
                ),
                patch(
                    'water_seg.train.SwinTinyUNet',
                    side_effect=lambda **kwargs: TinyWaterModel(),
                ),
                patch(
                    'water_seg.train.get_water_loaders',
                    return_value=(object(), object()),
                ),
                patch(
                    'water_seg.train.torch.optim.lr_scheduler.CosineAnnealingLR',
                    StaticScheduler,
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2], \
                    common_patches[3], common_patches[4], patch(
                        'water_seg.train.run_epoch',
                        side_effect=[metrics, metrics, RuntimeError('interrupt')],
                    ):
                with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                    water_train.main(args)

            interrupted = torch.load(
                save_dir / 'last.pth',
                map_location='cpu',
                weights_only=True,
            )
            self.assertEqual(interrupted['epoch'], 1)
            self.assertEqual(interrupted['best_epoch'], 1)
            self.assertEqual(len(interrupted['rng_states']), 1)

            with patch(
                'water_seg.train.build_kulsary_scene_index',
                return_value=fake_index,
            ), patch(
                'water_seg.train.compute_train_channel_stats',
                return_value=([-15.0, -20.0], [5.0, 4.0]),
            ), patch(
                'water_seg.train.SwinTinyUNet',
                side_effect=lambda **kwargs: TinyWaterModel(),
            ), patch(
                'water_seg.train.get_water_loaders',
                return_value=(object(), object()),
            ), patch(
                'water_seg.train.torch.optim.lr_scheduler.CosineAnnealingLR',
                StaticScheduler,
            ), patch(
                'water_seg.train.run_epoch',
                side_effect=[metrics, metrics, metrics, metrics],
            ) as resumed_epoch:
                water_train.main(
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

    def test_cli_requires_raw_paths_and_rejects_dataset_dir(self):
        with self.assertRaises(SystemExit):
            water_train.build_parser().parse_args([])
        eval_options = water_eval.build_parser().parse_args([
            '--path', 'checkpoint.pth',
        ])
        self.assertIsNone(eval_options.sigma0_before)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                water_train.build_parser().parse_args(
                    _data_args(directory) + ['--dataset-dir', directory]
                )
            with self.assertRaises(SystemExit):
                water_eval.build_parser().parse_args(
                    _data_args(directory)
                    + ['--path', 'checkpoint.pth', '--dataset-dir', directory]
                )

    def test_train_rejects_invalid_db_range_and_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'db-min'):
                water_train.main(
                    _data_args(directory)
                    + ['--db-min', '0', '--db-max', '-25']
                )
            with self.assertRaisesRegex(ValueError, 'sum to 1'):
                water_train.main(
                    _data_args(directory)
                    + ['--train-ratio', '0.6', '--val-ratio', '0.2']
                )


if __name__ == '__main__':
    unittest.main()
