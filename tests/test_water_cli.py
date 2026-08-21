from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from water_seg import eval as water_eval
from water_seg import train as water_train


class StaticScheduler:
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}


class TinyWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image):
        return self.head(self.encoder(image))


def _loader():
    images = torch.tensor([
        [[[0.0, 255.0], [255.0, 0.0]]],
        [[[255.0, 0.0], [0.0, 255.0]]],
    ])
    targets = torch.tensor([
        [[0, 1], [1, 0]],
        [[1, 0], [0, 1]],
    ])
    records = [
        (images[index], targets[index], f'sample-{index}')
        for index in range(images.size(0))
    ]
    return DataLoader(records, batch_size=2, shuffle=False)


class WaterCliTest(unittest.TestCase):
    def test_train_and_eval_main_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory) / 'run'
            loader = _loader()
            with patch(
                'water_seg.train.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.train.get_water_loaders',
                return_value=(loader, loader),
            ):
                best_path = water_train.main([
                    '--dataset-dir', directory,
                    '--epochs', '1',
                    '--batch-size', '2',
                    '--num-workers', '0',
                    '--device', 'cpu',
                    '--no-imagenet-pretrained',
                    '--no-augmentation',
                    '--early-stopping-patience', '0',
                    '--save-dir', str(save_dir),
                ])

            self.assertEqual(best_path, save_dir / 'best.pth')
            self.assertTrue((save_dir / 'best.pth').is_file())
            self.assertTrue((save_dir / 'last.pth').is_file())

            with patch(
                'water_seg.eval.SwinTinyUNet',
                return_value=TinyWaterModel(),
            ), patch(
                'water_seg.eval.get_water_test_loader',
                return_value=loader,
            ):
                metrics = water_eval.main([
                    '--dataset-dir', directory,
                    '--path', str(save_dir / 'best.pth'),
                    '--batch-size', '2',
                    '--num-workers', '0',
                    '--device', 'cpu',
                ])

            self.assertEqual(metrics['samples'], 2)
            self.assertGreaterEqual(metrics['water_iou'], 0.0)
            self.assertLessEqual(metrics['water_iou'], 1.0)

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
            with patch(
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
                water_train.main([
                    '--dataset-dir', directory,
                    '--epochs', '10',
                    '--batch-size', '2',
                    '--num-workers', '0',
                    '--device', 'cpu',
                    '--no-imagenet-pretrained',
                    '--no-augmentation',
                    '--early-stopping-patience', '2',
                    '--save-dir', str(save_dir),
                ])

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


if __name__ == '__main__':
    unittest.main()
