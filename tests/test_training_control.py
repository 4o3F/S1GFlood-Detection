import unittest

import numpy as np

from train import (binary_metrics_from_confusion,
                   is_significant_improvement, should_validate)
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
            if checks_without_improvement >= 3:
                stop_check = check_number
                break

        self.assertEqual(stop_check, 4)

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

    def test_rejects_epochs_above_safety_limit(self):
        parser, _ = parser_with_args()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--epochs', '1001'])

    def test_rejects_nonpositive_interval_and_patience(self):
        parser, _ = parser_with_args()
        with self.assertRaises(SystemExit):
            parser.parse_args(['--validation-interval', '0'])
        with self.assertRaises(SystemExit):
            parser.parse_args(['--early-stopping-patience', '0'])


if __name__ == '__main__':
    unittest.main()
