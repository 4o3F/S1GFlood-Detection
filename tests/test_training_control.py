import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from networks import TemporalAwareChangeEnhancement
from train import (ADAMW_DEFAULTS, binary_metrics_from_confusion, build_optimizer,
                   compute_multitask_loss, is_significant_improvement,
                   resolve_optimizer_type, should_stop_early, should_validate,
                   validate)
from utils.helpers import (dice_loss, get_mean_metrics, initialize_metrics,
                           set_metrics)
from utils.parser import parser_with_args


class ValidationScheduleTest(unittest.TestCase):
    def test_validates_final_epoch_before_interval(self):
        self.assertEqual(
            [epoch for epoch in range(1, 10) if should_validate(epoch, 9, 10)],
            [9],
        )

    def test_validates_interval_and_final_epoch(self):
        self.assertEqual(
            [epoch for epoch in range(1, 26) if should_validate(epoch, 25, 10)],
            [10, 20, 25],
        )

    def test_validates_only_interval_when_evenly_divisible(self):
        self.assertEqual(
            [epoch for epoch in range(1, 31) if should_validate(epoch, 30, 10)],
            [10, 20, 30],
        )


class EarlyStoppingTest(unittest.TestCase):
    def test_requires_improvement_strictly_above_min_delta(self):
        self.assertFalse(is_significant_improvement(0.801, 0.8, 0.001))
        self.assertTrue(is_significant_improvement(0.8011, 0.8, 0.001))

    def test_patience_counts_consecutive_validation_checks(self):
        best_f1 = float('-inf')
        checks_without_improvement = 0
        stop_check = None

        for check_number, current_f1 in enumerate((0.7, 0.7, 0.7, 0.7), start=1):
            if is_significant_improvement(current_f1, best_f1, 0.001):
                best_f1 = current_f1
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            if should_stop_early(checks_without_improvement, 3):
                stop_check = check_number
                break

        self.assertEqual(stop_check, 4)

    def test_patience_zero_disables_early_stopping(self):
        self.assertFalse(should_stop_early(0, 0))
        self.assertFalse(should_stop_early(1, 0))
        self.assertFalse(should_stop_early(100, 0))
        self.assertFalse(should_stop_early(3, -1))

    def test_patience_positive_stops_when_exhausted(self):
        self.assertFalse(should_stop_early(2, 3))
        self.assertTrue(should_stop_early(3, 3))
        self.assertTrue(should_stop_early(4, 3))

    def test_accumulated_small_improvement_can_cross_threshold(self):
        best_f1 = 0.7
        self.assertFalse(is_significant_improvement(0.7008, best_f1, 0.001))
        self.assertTrue(is_significant_improvement(0.7012, best_f1, 0.001))


class BinaryMetricsTest(unittest.TestCase):
    def test_computes_metrics_from_global_confusion(self):
        metrics = binary_metrics_from_confusion(
            np.array([[8, 2], [1, 9]], dtype=np.int64)
        )
        self.assertAlmostEqual(metrics['precisions'], 9 / 11)
        self.assertAlmostEqual(metrics['recalls'], 9 / 10)
        self.assertAlmostEqual(metrics['f1_scores'], 18 / 21)
        self.assertAlmostEqual(metrics['overall_accuracy'], 17 / 20)

    def test_zero_denominators_return_finite_zero(self):
        metrics = binary_metrics_from_confusion(
            np.array([[10, 0], [0, 0]], dtype=np.int64)
        )
        self.assertEqual(metrics['precisions'], 0.0)
        self.assertEqual(metrics['recalls'], 0.0)
        self.assertEqual(metrics['f1_scores'], 0.0)
        self.assertEqual(metrics['overall_accuracy'], 1.0)


class DiceLossTest(unittest.TestCase):
    def test_reduces_over_both_spatial_dimensions_for_3d_targets(self):
        logits = torch.tensor([
            [
                [[2.0, -1.0, 0.5], [1.5, -0.5, 0.0]],
                [[-1.0, 2.0, -0.5], [-1.5, 0.5, 1.0]],
            ],
        ])
        targets = torch.tensor([[[0, 1, 0], [0, 1, 1]]])

        probabilities = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(
            targets,
            num_classes=2,
        ).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dimensions)
        cardinality = torch.sum(probabilities + one_hot, dimensions)
        expected = 1 - (
            2 * intersection / (cardinality + 1e-7)
        ).mean()

        torch.testing.assert_close(dice_loss(logits, targets), expected)
        torch.testing.assert_close(
            dice_loss(logits, targets.unsqueeze(1)),
            expected,
        )


class MultiTaskLossTest(unittest.TestCase):
    def setUp(self):
        self.criterion = nn.CrossEntropyLoss()
        self.change_logits = torch.randn(2, 2, 3, 3, requires_grad=True)
        self.water_a_logits = torch.randn(2, 2, 3, 3, requires_grad=True)
        self.water_b_logits = torch.randn(2, 2, 3, 3, requires_grad=True)
        self.targets = {
            'change': torch.tensor([
                [[0, 1, 0], [1, 1, 0], [0, 0, 1]],
                [[1, 0, 1], [0, 0, 1], [1, 1, 0]],
            ]),
            'water_a': torch.tensor([
                [[0, 1, 0], [1, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]),
            'water_b': torch.tensor([
                [[1, 1, 0], [1, 0, 0], [0, 1, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ]),
            'water_valid': torch.tensor([True, False]),
        }

    def _outputs(self):
        return {
            'change_logits': self.change_logits,
            'water_a_logits': self.water_a_logits,
            'water_b_logits': self.water_b_logits,
        }

    def test_auxiliary_loss_uses_only_valid_samples(self):
        losses = compute_multitask_loss(
            self._outputs(),
            self.targets,
            self.criterion,
            water_loss_weight=0.2,
        )
        expected_change = self.criterion(
            self.change_logits,
            self.targets['change'],
        )
        expected_water = self.criterion(
            torch.cat((self.water_a_logits[:1], self.water_b_logits[:1])),
            torch.cat((self.targets['water_a'][:1], self.targets['water_b'][:1])),
        )
        torch.testing.assert_close(losses['change'], expected_change)
        torch.testing.assert_close(losses['water'], expected_water)
        torch.testing.assert_close(
            losses['total'],
            expected_change + 0.2 * expected_water,
        )
        self.assertEqual(losses['water_supervised_samples'], 1)

    def test_invalid_placeholder_masks_have_zero_auxiliary_gradient(self):
        losses = compute_multitask_loss(
            self._outputs(),
            self.targets,
            self.criterion,
            water_loss_weight=0.2,
        )
        losses['total'].backward()
        self.assertGreater(self.water_a_logits.grad[0].abs().sum().item(), 0)
        self.assertGreater(self.water_b_logits.grad[0].abs().sum().item(), 0)
        self.assertEqual(self.water_a_logits.grad[1].abs().sum().item(), 0)
        self.assertEqual(self.water_b_logits.grad[1].abs().sum().item(), 0)

    def test_unlabeled_batch_accepts_default_tensor_output(self):
        targets = dict(self.targets)
        targets['water_valid'] = torch.tensor([False, False])
        losses = compute_multitask_loss(
            self.change_logits,
            targets,
            self.criterion,
            water_loss_weight=0.2,
        )
        expected = self.criterion(self.change_logits, targets['change'])
        torch.testing.assert_close(losses['total'], expected)
        self.assertEqual(losses['water'].item(), 0.0)
        self.assertEqual(losses['water_supervised_samples'], 0)

    def test_zero_weight_disables_auxiliary_task_even_when_labels_exist(self):
        losses = compute_multitask_loss(
            self.change_logits,
            self.targets,
            self.criterion,
            water_loss_weight=0.0,
        )
        expected = self.criterion(self.change_logits, self.targets['change'])
        torch.testing.assert_close(losses['total'], expected)
        self.assertEqual(losses['water'].item(), 0.0)
        self.assertEqual(losses['water_supervised_samples'], 0)

    def test_supervised_batch_requires_auxiliary_logits(self):
        with self.assertRaisesRegex(ValueError, 'auxiliary logits'):
            compute_multitask_loss(
                self.change_logits,
                self.targets,
                self.criterion,
                water_loss_weight=0.2,
            )


class ValidationStub(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.return_aux_requests = []

    def forward(self, image_a, image_b, return_aux=False):
        self.return_aux_requests.append(return_aux)
        if return_aux:
            return self.outputs
        return self.outputs['change_logits']


class AuxiliaryValidationMetricsTest(unittest.TestCase):
    @staticmethod
    def _logits(predictions, margin=4.0):
        logits = torch.full(
            (predictions.size(0), 2, *predictions.shape[1:]),
            -margin,
        )
        logits.scatter_(1, predictions.unsqueeze(1), margin)
        return logits

    def _batch(self, water_valid):
        change = torch.tensor([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ])
        water_a = torch.tensor([
            [[0, 1], [1, 0]],
            [[0, 0], [0, 0]],
        ])
        water_b = torch.tensor([
            [[0, 1], [0, 1]],
            [[0, 0], [0, 0]],
        ])
        water_a_predictions = torch.tensor([
            [[0, 1], [0, 0]],
            [[1, 1], [1, 1]],
        ])
        water_b_predictions = torch.tensor([
            [[1, 1], [0, 1]],
            [[1, 1], [1, 1]],
        ])
        outputs = {
            'change_logits': self._logits(change),
            'water_a_logits': self._logits(water_a_predictions),
            'water_b_logits': self._logits(water_b_predictions),
        }
        targets = {
            'change': change,
            'water_a': water_a,
            'water_b': water_b,
            'water_valid': water_valid,
        }
        images = torch.zeros(2, 3, 2, 2)
        batch = (images, images.clone(), targets, ('a.png', 'b.png'))
        return outputs, batch

    def test_metrics_use_only_valid_water_labels(self):
        outputs, batch = self._batch(torch.tensor([True, False]))
        model = ValidationStub(outputs)
        criterion = nn.CrossEntropyLoss()
        metrics = validate(
            model,
            [batch],
            criterion,
            torch.device('cpu'),
            learning_rate=1e-4,
            water_loss_weight=0.2,
        )

        expected_water_loss = criterion(
            torch.cat((
                outputs['water_a_logits'][:1],
                outputs['water_b_logits'][:1],
            )),
            torch.cat((
                batch[2]['water_a'][:1],
                batch[2]['water_b'][:1],
            )),
        )
        self.assertEqual(model.return_aux_requests, [True])
        self.assertEqual(metrics['f1_scores'], 1.0)
        self.assertEqual(metrics['water_supervised_samples'], 1)
        self.assertEqual(metrics['water_supervision_fraction'], 0.5)
        self.assertAlmostEqual(metrics['water_losses'], expected_water_loss.item())
        self.assertAlmostEqual(metrics['water_a_precision'], 1.0)
        self.assertAlmostEqual(metrics['water_a_recall'], 0.5)
        self.assertAlmostEqual(metrics['water_a_f1'], 2 / 3)
        self.assertAlmostEqual(metrics['water_a_iou'], 0.5)
        self.assertAlmostEqual(metrics['water_b_precision'], 2 / 3)
        self.assertAlmostEqual(metrics['water_b_recall'], 1.0)
        self.assertAlmostEqual(metrics['water_b_f1'], 0.8)
        self.assertAlmostEqual(metrics['water_b_iou'], 2 / 3)
        self.assertAlmostEqual(metrics['water_pooled_precision'], 0.75)
        self.assertAlmostEqual(metrics['water_pooled_recall'], 0.75)
        self.assertAlmostEqual(metrics['water_pooled_f1'], 0.75)
        self.assertAlmostEqual(metrics['water_pooled_iou'], 0.6)

    def test_unlabeled_validation_skips_auxiliary_forward_and_metrics(self):
        outputs, batch = self._batch(torch.tensor([False, False]))
        model = ValidationStub(outputs)
        metrics = validate(
            model,
            [batch],
            nn.CrossEntropyLoss(),
            torch.device('cpu'),
            learning_rate=1e-4,
            water_loss_weight=0.2,
        )

        self.assertEqual(model.return_aux_requests, [False])
        self.assertEqual(metrics['water_supervised_samples'], 0)
        self.assertEqual(metrics['water_supervision_fraction'], 0.0)
        self.assertEqual(metrics['water_losses'], 0.0)
        self.assertFalse(any(key.startswith('water_a_') for key in metrics))
        self.assertFalse(any(key.startswith('water_b_') for key in metrics))
        self.assertFalse(any(key.startswith('water_pooled_') for key in metrics))


class TrainingMetricAggregationTest(unittest.TestCase):
    def _record_batch(
        self,
        metrics,
        *,
        water_loss,
        supervised_samples,
        batch_size,
    ):
        return set_metrics(
            metrics,
            total_loss=torch.tensor(1.0 + 0.2 * water_loss),
            overall_accuracy=torch.tensor(1.0),
            report=(1.0, 1.0, 1.0, None),
            lr=[1e-4],
            change_loss=torch.tensor(1.0),
            water_loss=torch.tensor(water_loss),
            water_loss_weight=0.2,
            water_supervised_samples=supervised_samples,
            batch_size=batch_size,
        )

    def test_sparse_water_metrics_are_sample_weighted(self):
        metrics = initialize_metrics()
        self._record_batch(
            metrics,
            water_loss=1.0,
            supervised_samples=4,
            batch_size=4,
        )
        self._record_batch(
            metrics,
            water_loss=0.0,
            supervised_samples=0,
            batch_size=1,
        )

        mean = get_mean_metrics(metrics)
        self.assertEqual(mean['water_supervised_samples'], 4)
        self.assertAlmostEqual(mean['water_supervision_fraction'], 4 / 5)
        self.assertAlmostEqual(mean['water_losses'], 1.0)
        self.assertAlmostEqual(mean['weighted_water_losses'], 0.2)

    def test_water_loss_is_weighted_by_valid_sample_count(self):
        metrics = initialize_metrics()
        self._record_batch(
            metrics,
            water_loss=2.0,
            supervised_samples=1,
            batch_size=4,
        )
        self._record_batch(
            metrics,
            water_loss=1.0,
            supervised_samples=2,
            batch_size=2,
        )

        mean = get_mean_metrics(metrics)
        self.assertAlmostEqual(mean['water_losses'], 4 / 3)
        self.assertAlmostEqual(mean['weighted_water_losses'], 0.8 / 3)
        self.assertAlmostEqual(mean['water_supervision_fraction'], 3 / 6)

    def test_unlabeled_epoch_reports_zero_auxiliary_metrics(self):
        metrics = initialize_metrics()
        self._record_batch(
            metrics,
            water_loss=0.0,
            supervised_samples=0,
            batch_size=3,
        )
        mean = get_mean_metrics(metrics)
        self.assertEqual(mean['water_losses'], 0.0)
        self.assertEqual(mean['weighted_water_losses'], 0.0)
        self.assertEqual(mean['water_supervision_fraction'], 0.0)


class OptimizerSelectionTest(unittest.TestCase):
    def test_default_optimizer_is_adamw_for_vitae(self):
        self.assertEqual(resolve_optimizer_type('vitae'), 'adamw')
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(model, 'vitae')
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        group = optimizer.param_groups[0]
        self.assertAlmostEqual(group['lr'], ADAMW_DEFAULTS['lr'])
        self.assertEqual(tuple(group['betas']), ADAMW_DEFAULTS['betas'])
        self.assertAlmostEqual(group['weight_decay'], ADAMW_DEFAULTS['weight_decay'])

    def test_default_optimizer_is_adamw_for_other_backbones(self):
        for backbone in ('swin', 'resnet'):
            self.assertEqual(resolve_optimizer_type(backbone), 'adamw')
            model = nn.Linear(4, 2)
            optimizer = build_optimizer(model, backbone)
            self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_explicit_sgd_still_available(self):
        self.assertEqual(resolve_optimizer_type('vitae', 'sgd'), 'sgd')
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(model, 'vitae', optimizer_type='sgd')
        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_rejects_unknown_optimizer_type(self):
        with self.assertRaises(ValueError):
            resolve_optimizer_type('vitae', 'rmsprop')


class TaceProjectionRegistrationTest(unittest.TestCase):
    def test_last_stage_registers_qkv_projections(self):
        token_dim = 32
        module = TemporalAwareChangeEnhancement(
            token_dim=token_dim,
            H=8,
            W=8,
            is_last_stage=True,
        )
        named = dict(module.named_parameters())
        for name in (
            'proj_q.weight',
            'proj_k.weight',
            'proj_v.weight',
            'class_token',
        ):
            self.assertIn(name, named)

        # Optimizer must observe the registered projection weights.
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
        param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertIn(id(module.proj_q.weight), param_ids)
        self.assertIn(id(module.proj_k.weight), param_ids)
        self.assertIn(id(module.proj_v.weight), param_ids)

        # Forward shape contract remains compatible.
        batch = 2
        r_i = torch.randn(batch, 64, token_dim)
        f_c = torch.randn(batch, 4, token_dim)
        f_e_map, t_sem = module(r_i, f_c)
        self.assertEqual(tuple(f_e_map.shape), (batch, token_dim, 8, 8))
        self.assertEqual(tuple(t_sem.shape), (batch, token_dim))

        # Projections are stateful (not recreated each forward).
        before = module.proj_q.weight.detach().clone()
        loss = f_e_map.mean() + t_sem.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, module.proj_q.weight.detach()))

    def test_non_last_stage_has_no_projection_layers(self):
        module = TemporalAwareChangeEnhancement(
            token_dim=32,
            H=8,
            W=8,
            is_last_stage=False,
        )
        self.assertFalse(hasattr(module, 'proj_q'))
        self.assertFalse(hasattr(module, 'proj_k'))
        self.assertFalse(hasattr(module, 'proj_v'))
        self.assertFalse(hasattr(module, 'class_token'))


class TrainingParserTest(unittest.TestCase):
    def test_training_control_defaults(self):
        parser, _ = parser_with_args()
        args = parser.parse_args([])
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.validation_interval, 10)
        self.assertEqual(args.early_stopping_patience, 3)
        self.assertEqual(args.min_f1_improvement, 0.001)
        self.assertEqual(args.water_loss_weight, 0.2)

    def test_legacy_metadata_defaults_auxiliary_weight(self):
        metadata_path = Path(__file__).parents[1] / 'metadata_file.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.pop('water_loss_weight')
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / 'metadata.json'
            legacy_path.write_text(
                json.dumps(metadata),
                encoding='utf-8',
            )
            parser, _ = parser_with_args(str(legacy_path))
            args = parser.parse_args([])
        self.assertEqual(args.water_loss_weight, 0.2)

    def test_training_control_overrides(self):
        parser, _ = parser_with_args()
        args = parser.parse_args([
            '--epochs', '500',
            '--batch-size', '4',
            '--validation-interval', '20',
            '--early-stopping-patience', '5',
            '--min-f1-improvement', '0.005',
            '--water-loss-weight', '0.35',
        ])
        self.assertEqual(args.epochs, 500)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.validation_interval, 20)
        self.assertEqual(args.early_stopping_patience, 5)
        self.assertEqual(args.min_f1_improvement, 0.005)
        self.assertEqual(args.water_loss_weight, 0.35)

    def test_paper_repro_config_accepts_fixed_epochs_and_disabled_early_stop(self):
        parser, _ = parser_with_args()
        args = parser.parse_args([
            '--epochs', '100',
            '--early-stopping-patience', '0',
        ])
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.early_stopping_patience, 0)
        self.assertFalse(should_stop_early(99, args.early_stopping_patience))

    def test_rejects_epochs_above_safety_limit(self):
        parser, _ = parser_with_args()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--epochs', '1001'])

    def test_rejects_nonpositive_interval_and_negative_patience(self):
        parser, _ = parser_with_args()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--validation-interval', '0'])
        with self.assertRaises(SystemExit):
            parser.parse_args(['--early-stopping-patience', '-1'])

    def test_rejects_invalid_water_loss_weights(self):
        parser, _ = parser_with_args()
        for value in ('-0.1', 'nan', 'inf', '-inf'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(['--water-loss-weight', value])


if __name__ == '__main__':
    unittest.main()
