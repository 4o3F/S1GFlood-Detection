import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from water_seg import compute_geoid_stats
from water_seg import pretrain_geoid
from water_seg.geoid_dataset import GEOID_RADIOMETRY


class FakeIndex:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata_path = self.root / 'data_tiles_s256_st128.csv'
        self.metadata_path.write_text('metadata', encoding='utf-8')
        self.min_valid_proportion = 0.01

    def counts(self):
        return {
            'train': 621250,
            'val': 28810,
            'samples': 650060,
            'pre': 325022,
            'post': 325038,
        }


def _load_generated_module(path):
    spec = importlib.util.spec_from_file_location('generated_geoid_stats', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GEOIDStatsCliTest(unittest.TestCase):
    def test_scanner_writes_importable_constants_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'geoid-flood'
            root.mkdir()
            index = FakeIndex(root)
            output = Path(directory) / 'generated_stats.py'
            with patch(
                'water_seg.compute_geoid_stats.build_geoid_water_index',
                return_value=index,
            ), patch(
                'water_seg.compute_geoid_stats.validate_geoid_files',
                return_value={'s1grd_files': 19823, 'label_files': 9911},
            ), patch(
                'water_seg.compute_geoid_stats.compute_geoid_train_channel_stats',
                return_value=([-13.25, -18.5], [4.75, 3.25]),
            ):
                result = compute_geoid_stats.main([
                    '--geoid-root', str(root),
                    '--output', str(output),
                    '--no-progress',
                ])

            generated = _load_generated_module(output).GEOID_CHANNEL_STATS

        self.assertEqual(result, output.resolve())
        self.assertEqual(generated['channel_mean'], [-13.25, -18.5])
        self.assertEqual(generated['channel_std'], [4.75, 3.25])
        self.assertEqual(generated['polarizations'], ['VV', 'VH'])
        self.assertEqual(generated['radiometry'], GEOID_RADIOMETRY)
        self.assertNotIn('db_min', generated)
        self.assertNotIn('db_max', generated)
        self.assertEqual(generated['train_samples'], 621250)
        self.assertEqual(generated['s1grd_files'], 19823)

    def test_pretraining_rejects_uninitialized_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(directory)
            with patch.object(pretrain_geoid, 'GEOID_CHANNEL_STATS', None):
                with self.assertRaisesRegex(RuntimeError, 'not generated'):
                    pretrain_geoid._load_geoid_channel_constants(index)

    def test_pretraining_rejects_stale_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(directory)
            stats = {
                'polarizations': ['VV', 'VH'],
                'channel_mean': [-13.25, -18.5],
                'channel_std': [4.75, 3.25],
                'radiometry': GEOID_RADIOMETRY,
                'min_valid_proportion': 0.01,
                'train_samples': 1,
                'metadata_fingerprint': {},
            }
            with patch.object(pretrain_geoid, 'GEOID_CHANNEL_STATS', stats):
                with self.assertRaisesRegex(ValueError, 'train_samples'):
                    pretrain_geoid._load_geoid_channel_constants(index)

    def test_pretraining_rejects_legacy_clipped_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(directory)
            stats = {
                'polarizations': ['VV', 'VH'],
                'channel_mean': [-10.25, -16.86],
                'channel_std': [4.48, 4.49],
                'radiometry': 'clipped [-25,0] dB',
                'min_valid_proportion': 0.01,
                'train_samples': 621250,
                'metadata_fingerprint': {},
            }
            with patch.object(pretrain_geoid, 'GEOID_CHANNEL_STATS', stats):
                with self.assertRaisesRegex(ValueError, 'radiometry'):
                    pretrain_geoid._load_geoid_channel_constants(index)


if __name__ == '__main__':
    unittest.main()
