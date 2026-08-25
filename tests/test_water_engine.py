import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from water_seg.engine import (CHECKPOINT_FORMAT_VERSION, INPUT_CONTRACT,
                              NORMALIZATION_CONTRACT, build_optimizer,
                              checkpoint_payload, load_initial_model_weights,
                              load_model_checkpoint, metrics_from_confusion,
                              run_epoch, save_checkpoint)


class TinyWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image):
        return self.head(self.encoder(image))


def checkpoint_config(architecture='tiny'):
    return {
        'architecture': architecture,
        'input': INPUT_CONTRACT,
        'in_chans': 1,
        'normalization': NORMALIZATION_CONTRACT,
        'vv_mean': -12.5,
        'vv_std': 4.0,
        'db_min': -25.0,
        'db_max': 0.0,
        'sigma0_before': '/data/before.tif',
        'sigma0_peak': '/data/peak.tif',
        'sigma0_after': '/data/after.tif',
        'mask_source': '/data/masks',
        'split_seed': 42,
        'block_tiles': 2,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'kept_tile_count': 3,
        'samples_per_split': {'train': 3, 'val': 3, 'test': 3},
        'grid_signature': {
            'crs': 'EPSG:32639',
            'transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0],
            'width': 768,
            'height': 768,
            'peak_window': [0.0, 0.0, 768.0, 768.0],
        },
        'tile_splits': [
            {'row': 0, 'col': 0, 'split': 'train'},
            {'row': 0, 'col': 1, 'split': 'val'},
            {'row': 1, 'col': 0, 'split': 'test'},
        ],
        'source_fingerprints': {
            'sigma0': {
                role: {'size': 100, 'sampled_sha256': role * 8}
                for role in ('before', 'peak', 'after')
            },
            'masks': {
                role: {'size': 10, 'sampled_sha256': role * 8}
                for role in ('before', 'peak', 'after')
            },
        },
    }


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

    def _payload(self, config=None):
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
        return checkpoint_payload(
            self.model,
            optimizer,
            scheduler,
            epoch=1,
            best_water_iou=0.75,
            train_metrics={'loss': 0.5},
            val_metrics={'water_iou': 0.75},
            config=checkpoint_config() if config is None else config,
        )

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

    def test_epoch_excludes_ignore_index_from_metrics(self):
        images = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        targets = torch.tensor([[[0, 1], [255, 0]]])
        loader = DataLoader(
            [(images[0], targets[0], 'geoid-sample')],
            batch_size=1,
            shuffle=False,
        )
        metrics = run_epoch(
            self.model,
            loader,
            nn.CrossEntropyLoss(ignore_index=255),
            torch.device('cpu'),
        )
        self.assertEqual(metrics['samples'], 1)
        self.assertGreaterEqual(metrics['overall_accuracy'], 0.0)
        self.assertLessEqual(metrics['overall_accuracy'], 1.0)

    def test_checkpoint_round_trip_restores_model_state(self):
        expected = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        payload = self._payload()
        self.assertEqual(payload['format_version'], CHECKPOINT_FORMAT_VERSION)

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

    def test_checkpoint_rejects_format_one(self):
        payload = self._payload()
        payload['format_version'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pth'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'format 1'):
                load_model_checkpoint(path, self.model)

    def test_checkpoint_rejects_missing_required_metadata(self):
        payload = self._payload()
        del payload['config']['vv_std']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.pth'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'config is missing'):
                load_model_checkpoint(path, self.model)

    def test_checkpoint_rejects_wrong_architecture(self):
        payload = self._payload(checkpoint_config('TinyWaterModel'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, 'architecture'):
                load_model_checkpoint(
                    path,
                    self.model,
                    expected_architecture='SwinTinyUNet',
                )

    def test_checkpoint_rejects_invalid_protocol_fields(self):
        for key, value, message in (
            ('input', 'raw 0-255', 'input contract'),
            ('normalization', 'imagenet', 'normalization contract'),
            ('in_chans', 3, 'one VV'),
            ('vv_std', 0.0, 'vv_std'),
            ('db_max', -30.0, 'dB range'),
        ):
            with self.subTest(key=key):
                config = checkpoint_config()
                config[key] = value
                payload = self._payload(config)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'invalid.pth'
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_model_checkpoint(path, self.model)

    def test_geoid_pretraining_checkpoint_loads_model_weights_only(self):
        expected = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        payload = {
            'kind': 'geoid-water-pretraining',
            'format_version': 1,
            'epoch': 2,
            'model_state_dict': self.model.state_dict(),
            'config': {
                'architecture': 'tiny',
                'input': INPUT_CONTRACT,
                'normalization': NORMALIZATION_CONTRACT,
                'in_chans': 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geoid.pth'
            torch.save(payload, path)
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.zero_()
            loaded = load_initial_model_weights(
                path,
                self.model,
                expected_architecture='tiny',
            )

        self.assertEqual(loaded['kind'], 'geoid-water-pretraining')
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, expected[name])


if __name__ == '__main__':
    unittest.main()
