import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from water_seg import compute_geoid_stats
from water_seg import pretrain_geoid


class FakeIndex:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata_path = self.root / 'data_tiles_s256_st128.csv'
        self.metadata_path.write_text('metadata', encoding='utf-8')
        self.db_min = -25.0
        self.db_max = 0.0
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
                'water_seg.compute_geoid_stats.compute_geoid_train_vv_stats',
                return_value=(-13.25, 4.75),
            ):
                result = compute_geoid_stats.main([
                    '--geoid-root', str(root),
                    '--output', str(output),
                    '--no-progress',
                ])

            generated = _load_generated_module(output).GEOID_VV_STATS

        self.assertEqual(result, output.resolve())
        self.assertEqual(generated['vv_mean'], -13.25)
        self.assertEqual(generated['vv_std'], 4.75)
        self.assertEqual(generated['train_samples'], 621250)
        self.assertEqual(generated['s1grd_files'], 19823)

    def test_pretraining_rejects_uninitialized_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(directory)
            with patch.object(pretrain_geoid, 'GEOID_VV_STATS', None):
                with self.assertRaisesRegex(RuntimeError, 'not generated'):
                    pretrain_geoid._load_geoid_vv_constants(index)

    def test_pretraining_rejects_stale_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            index = FakeIndex(directory)
            stats = {
                'vv_mean': -13.25,
                'vv_std': 4.75,
                'db_min': -25.0,
                'db_max': 0.0,
                'min_valid_proportion': 0.01,
                'train_samples': 1,
                'metadata_fingerprint': {},
            }
            with patch.object(pretrain_geoid, 'GEOID_VV_STATS', stats):
                with self.assertRaisesRegex(ValueError, 'train_samples'):
                    pretrain_geoid._load_geoid_vv_constants(index)


if __name__ == '__main__':
    unittest.main()
