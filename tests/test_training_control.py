import unittest

import numpy as np
import torch
import torch.nn as nn

from networks import TemporalAwareChangeEnhancement
from train import (ADAMW_DEFAULTS, binary_metrics_from_confusion, build_optimizer,
                   is_significant_improvement, resolve_optimizer_type,
                   should_stop_early, should_validate)
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

    def test_training_control_overrides(self):
        parser, _ = parser_with_args()
        args = parser.parse_args([
            '--epochs', '500',
            '--batch-size', '4',
            '--validation-interval', '20',
            '--early-stopping-patience', '5',
            '--min-f1-improvement', '0.005',
        ])
        self.assertEqual(args.epochs, 500)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.validation_interval, 20)
        self.assertEqual(args.early_stopping_patience, 5)
        self.assertEqual(args.min_f1_improvement, 0.005)

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


if __name__ == '__main__':
    unittest.main()
