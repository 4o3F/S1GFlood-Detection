import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from water_seg.engine import (build_optimizer, checkpoint_payload,
                              load_model_checkpoint, metrics_from_confusion,
                              run_epoch, save_checkpoint)


class TinyWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image):
        return self.head(self.encoder(image))


class WaterEngineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyWaterModel()

    def _loader(self):
        images = torch.tensor([
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
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

    def test_metrics_include_water_iou_from_global_confusion(self):
        metrics = metrics_from_confusion(
            np.array([[8, 2], [1, 9]], dtype=np.int64)
        )
        self.assertAlmostEqual(metrics['precision'], 9 / 11)
        self.assertAlmostEqual(metrics['recall'], 9 / 10)
        self.assertAlmostEqual(metrics['f1'], 18 / 21)
        self.assertAlmostEqual(metrics['water_iou'], 9 / 12)
        self.assertAlmostEqual(metrics['overall_accuracy'], 17 / 20)

    def test_metrics_are_plain_json_serializable_scalars(self):
        metrics = metrics_from_confusion(
            np.array([[8, 2], [1, 9]], dtype=np.int64)
        )
        serialized = json.dumps(metrics, sort_keys=True)
        self.assertIn('water_iou', serialized)
        for value in metrics.values():
            self.assertIsInstance(value, float)

    def test_optimizer_uses_distinct_encoder_and_decoder_rates(self):
        optimizer = build_optimizer(
            self.model,
            encoder_lr=5e-5,
            decoder_lr=5e-4,
            weight_decay=0.01,
        )
        self.assertEqual(
            [group['lr'] for group in optimizer.param_groups],
            [5e-5, 5e-4],
        )
        encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
        first_group_ids = {
            id(parameter) for parameter in optimizer.param_groups[0]['params']
        }
        second_group_ids = {
            id(parameter) for parameter in optimizer.param_groups[1]['params']
        }
        self.assertEqual(first_group_ids, encoder_ids)
        self.assertFalse(first_group_ids & second_group_ids)

    def test_training_epoch_updates_parameters_and_reports_samples(self):
        optimizer = build_optimizer(
            self.model,
            encoder_lr=1e-2,
            decoder_lr=1e-2,
            weight_decay=0.0,
        )
        before = self.model.head.weight.detach().clone()
        metrics = run_epoch(
            self.model,
            self._loader(),
            nn.CrossEntropyLoss(),
            torch.device('cpu'),
            optimizer=optimizer,
        )
        self.assertEqual(metrics['samples'], 2)
        self.assertGreaterEqual(metrics['water_iou'], 0.0)
        self.assertLessEqual(metrics['water_iou'], 1.0)
        self.assertFalse(torch.equal(before, self.model.head.weight.detach()))

    def test_checkpoint_round_trip_restores_model_state(self):
        optimizer = build_optimizer(
            self.model,
            encoder_lr=5e-5,
            decoder_lr=5e-4,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=2,
        )
        expected = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        payload = checkpoint_payload(
            self.model,
            optimizer,
            scheduler,
            epoch=1,
            best_water_iou=0.75,
            train_metrics={'loss': 0.5},
            val_metrics={'water_iou': 0.75},
            config={'architecture': 'tiny'},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, payload)
            self.assertTrue(path.is_file())
            self.assertFalse(Path(f'{path}.tmp').exists())
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.zero_()
            checkpoint = load_model_checkpoint(path, self.model)

        self.assertEqual(checkpoint['epoch'], 1)
        self.assertEqual(checkpoint['best_water_iou'], 0.75)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, expected[name])

    def test_checkpoint_rejects_missing_required_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pth'
            torch.save(
                {
                    'format_version': 1,
                    'model_state_dict': self.model.state_dict(),
                    'config': {},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, 'required keys'):
                load_model_checkpoint(path, self.model)

    def test_checkpoint_rejects_wrong_architecture(self):
        optimizer = build_optimizer(
            self.model,
            encoder_lr=5e-5,
            decoder_lr=5e-4,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=2,
        )
        payload = checkpoint_payload(
            self.model,
            optimizer,
            scheduler,
            epoch=1,
            best_water_iou=0.75,
            train_metrics={'loss': 0.5},
            val_metrics={'water_iou': 0.75},
            config={'architecture': 'TinyWaterModel'},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                load_model_checkpoint(
                    path,
                    self.model,
                    expected_architecture='SwinTinyUNet',
                )


if __name__ == '__main__':
    unittest.main()
