from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio
from rasterio.transform import Affine

import prepare_kulsary_sigma0 as preprocessor
from tests.test_kulsary_products import PRODUCTS, write_role_safe


TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0)


def write_sigma0(path: Path, value=0.1, *, count=1, crs='EPSG:32639'):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((64, 64), value, dtype=np.float32)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=64,
        height=64,
        count=count,
        dtype='float32',
        crs=crs,
        transform=TRANSFORM,
        nodata=0.0,
    ) as dataset:
        for band in range(1, count + 1):
            dataset.write(array, band)
            dataset.set_band_description(band, 'Sigma0_VV')


class KulsarySigma0PreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.safe_root = self.root / 'restored_grd'
        self.safe_root.mkdir()
        for role in PRODUCTS:
            write_role_safe(self.safe_root, role)
        self.output = self.root / 'kulsary_sigma0'
        self.work = self.root / 'work'
        self.cache = self.root / 'cache'
        self.graph = self.root / 'graph.xml'
        self.graph.write_text('<graph/>\n', encoding='utf-8')
        self.rasters = {}
        for role, value in (
            ('before', 0.01),
            ('peak', 0.1),
            ('after', 0.03162278),
        ):
            path = self.root / 'cache-source' / f'{role}.tif'
            write_sigma0(path, value)
            self.rasters[role] = path

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, *extra):
        argv = [
            '--safe-root', str(self.safe_root),
            '--output', str(self.output),
            '--work-dir', str(self.work),
            '--snap-cache-dir', str(self.cache),
            '--graph', str(self.graph),
            '--gpt', str(self.root / 'fake-gpt'),
        ]
        argv.extend(extra)
        return preprocessor.parse_args(argv)

    def _get_sigma0(self, gpt, graph, product, output, args, label, cache_root, refresh):
        return self.rasters[label]

    def _run(self, *extra, link_side_effect=None):
        contexts = [
            mock.patch.object(preprocessor, 'resolve_gpt', return_value='/fake/gpt'),
            mock.patch.object(
                preprocessor,
                'get_or_create_sigma0',
                side_effect=self._get_sigma0,
            ),
        ]
        if link_side_effect is not None:
            contexts.append(
                mock.patch.object(
                    preprocessor.os,
                    'link',
                    side_effect=link_side_effect,
                )
            )
        with contexts[0] as resolve_gpt, contexts[1] as get_sigma0:
            if len(contexts) == 3:
                with contexts[2]:
                    result = preprocessor.prepare(self._args(*extra))
            else:
                result = preprocessor.prepare(self._args(*extra))
        return result, resolve_gpt, get_sigma0

    def test_full_run_publishes_three_hardlinked_rasters_and_manifest(self):
        result, resolve_gpt, get_sigma0 = self._run()
        self.assertEqual(result['output'], str(self.output.resolve()))
        resolve_gpt.assert_called_once()
        self.assertEqual(get_sigma0.call_count, 3)

        manifest = json.loads(
            (self.output / preprocessor.MANIFEST_FILENAME).read_text(
                encoding='utf-8'
            )
        )
        self.assertEqual(manifest['format'], 'kulsary-sigma0')
        self.assertEqual(manifest['version'], preprocessor.PREPROCESSOR_VERSION)
        self.assertEqual(set(manifest['roles']), {'before', 'peak', 'after'})
        for role, filename in preprocessor.OUTPUT_FILENAMES.items():
            destination = self.output / filename
            self.assertTrue(destination.is_file())
            preprocessor.validate_sigma0_raster(destination)
            self.assertEqual(manifest['roles'][role]['output_filename'], filename)
            self.assertEqual(
                manifest['roles'][role]['product_identifier'],
                PRODUCTS[role]['name'],
            )
            self.assertEqual(
                manifest['roles'][role]['publication_method'],
                'hardlink',
            )
            self.assertEqual(
                os.stat(destination).st_ino,
                os.stat(self.rasters[role]).st_ino,
            )
        self.assertFalse(
            self.output.with_name(f'.{self.output.name}.partial').exists()
        )
        self.assertFalse(any(self.work.glob('.kulsary-sigma0-*')))

    def test_cross_filesystem_link_failure_falls_back_to_copy(self):
        error = OSError(errno.EXDEV, 'cross-device link')
        self._run(link_side_effect=error)
        manifest = json.loads(
            (self.output / preprocessor.MANIFEST_FILENAME).read_text(
                encoding='utf-8'
            )
        )
        for role, filename in preprocessor.OUTPUT_FILENAMES.items():
            destination = self.output / filename
            self.assertEqual(
                manifest['roles'][role]['publication_method'],
                'copy',
            )
            self.assertNotEqual(
                os.stat(destination).st_ino,
                os.stat(self.rasters[role]).st_ino,
            )

    def test_dry_run_never_resolves_gpt_or_writes(self):
        with mock.patch.object(
            preprocessor,
            'resolve_gpt',
            side_effect=AssertionError('resolved gpt'),
        ), mock.patch.object(
            preprocessor,
            'get_or_create_sigma0',
            side_effect=AssertionError('created Sigma0'),
        ):
            result = preprocessor.prepare(self._args('--dry-run'))
        self.assertTrue(result['dry_run'])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.cache.exists())

    def test_refresh_is_forwarded_once_per_role(self):
        _, _, get_sigma0 = self._run('--refresh-snap-cache')
        self.assertEqual(get_sigma0.call_count, 3)
        self.assertTrue(all(call.args[7] for call in get_sigma0.call_args_list))

    def test_failure_removes_staging_and_run_dir_but_keeps_cache(self):
        self.cache.mkdir()
        calls = []

        def fail_second(*args):
            calls.append(args[5])
            if len(calls) == 2:
                raise RuntimeError('synthetic SNAP failure')
            return self.rasters[args[5]]

        with mock.patch.object(
            preprocessor,
            'resolve_gpt',
            return_value='/fake/gpt',
        ), mock.patch.object(
            preprocessor,
            'get_or_create_sigma0',
            side_effect=fail_second,
        ):
            with self.assertRaisesRegex(RuntimeError, 'synthetic SNAP'):
                preprocessor.prepare(self._args())
        self.assertFalse(self.output.exists())
        self.assertFalse(
            self.output.with_name(f'.{self.output.name}.partial').exists()
        )
        self.assertTrue(self.cache.is_dir())
        self.assertFalse(any(self.work.glob('.kulsary-sigma0-*')))

    def test_post_publication_failure_removes_staging(self):
        self.cache.mkdir()
        with mock.patch.object(
            preprocessor,
            'resolve_gpt',
            return_value='/fake/gpt',
        ), mock.patch.object(
            preprocessor,
            'get_or_create_sigma0',
            side_effect=self._get_sigma0,
        ), mock.patch.object(
            preprocessor,
            '_verify_staging',
            side_effect=RuntimeError('synthetic staging verification failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'staging verification'):
                preprocessor.prepare(self._args())
        self.assertFalse(self.output.exists())
        self.assertFalse(
            self.output.with_name(f'.{self.output.name}.partial').exists()
        )
        self.assertTrue(self.cache.is_dir())
        self.assertFalse(any(self.work.glob('.kulsary-sigma0-*')))

    def test_invalid_sigma0_is_rejected(self):
        invalid = self.root / 'invalid.tif'
        write_sigma0(invalid, 0.1, count=2)
        self.rasters['after'] = invalid
        with self.assertRaises(Exception):
            self._run()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_staging_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            preprocessor.prepare(self._args())
        self.output.rmdir()
        staging = self.output.with_name(f'.{self.output.name}.partial')
        staging.mkdir()
        with self.assertRaises(FileExistsError):
            preprocessor.prepare(self._args())

    def test_path_overlap_is_rejected(self):
        args = self._args('--output', str(self.safe_root / 'sigma0'))
        with self.assertRaisesRegex(ValueError, 'safe-root'):
            preprocessor.prepare(args)
        args = self._args('--work-dir', str(self.safe_root / 'work'))
        with self.assertRaisesRegex(ValueError, 'work directory'):
            preprocessor.prepare(args)
        args = self._args('--snap-cache-dir', str(self.safe_root / 'cache'))
        with self.assertRaisesRegex(ValueError, 'SNAP cache'):
            preprocessor.prepare(args)

    def test_refresh_dry_run_reports_rebuild(self):
        args = self._args('--dry-run', '--refresh-snap-cache')
        products = preprocessor.discover_kulsary_grd_products(self.safe_root)
        with mock.patch.object(
            preprocessor,
            'build_snap_cache_key',
            return_value=('key', {}),
        ), mock.patch.object(
            preprocessor,
            '_snap_cache_entry_dir',
            return_value=self.cache / 'entry',
        ), mock.patch.object(
            preprocessor,
            'load_snap_cache_entry',
            return_value=self.root / 'cached.tif',
        ):
            statuses = preprocessor._probe_cache(
                args,
                products,
                self.graph,
                self.cache,
                '/fake/gpt',
            )
        self.assertEqual(
            statuses,
            {role: 'refresh (would rebuild)' for role in PRODUCTS},
        )

    def test_main_uses_error_prefix(self):
        with self.assertRaises(SystemExit) as context:
            preprocessor.main([
                '--safe-root', str(self.safe_root),
                '--output', str(self.output),
                '--graph', str(self.graph),
                '--pixel-spacing', '0',
            ])
        self.assertIn('Error:', str(context.exception))


if __name__ == '__main__':
    unittest.main()
