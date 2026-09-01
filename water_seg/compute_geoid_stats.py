import argparse
import json
import os
from pathlib import Path
import pprint
import tempfile

from utils.kulsary_raster import sampled_file_fingerprint
from utils.parser import finite_float
from water_seg.geoid_dataset import (GEOID_METADATA_FILENAME,
                                     GEOID_RADIOMETRY,
                                     build_geoid_water_index,
                                     compute_geoid_train_channel_stats,
                                     validate_geoid_files)
from utils.kulsary_raster import POLARIZATIONS


DEFAULT_OUTPUT = Path(__file__).with_name('geoid_stats.py')


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Scan GEOID train windows once and write VV+VH constants for '
            'pretraining'
        )
    )
    parser.add_argument('--geoid-root', type=Path, required=True)
    parser.add_argument(
        '--metadata-filename',
        default=GEOID_METADATA_FILENAME,
    )
    parser.add_argument(
        '--min-valid-proportion',
        type=finite_float,
        default=0.01,
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Python constants module to replace',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_options(options):
    if not 0.0 <= options.min_valid_proportion <= 1.0:
        raise ValueError('--min-valid-proportion must be between 0 and 1')


def _render_constants(stats):
    rendered = pprint.pformat(stats, sort_dicts=True, width=88)
    return (
        '"""Generated GEOID VV+VH normalization constants.\n\n'
        'Regenerate with ``python -m water_seg.compute_geoid_stats`` when '
        'the metadata selection or radiometry contract changes.\n'
        '"""\n\n\n'
        f'GEOID_CHANNEL_STATS = {rendered}\n'
    )


def _write_constants(path, stats):
    output = Path(path).expanduser().resolve()
    if output.suffix != '.py':
        raise ValueError('--output must be a Python file')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output.parent,
            prefix=f'.{output.name}.',
            suffix='.tmp',
            delete=False,
        ) as stream:
            stream.write(_render_constants(stats))
            temporary_path = Path(stream.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def main(argv=None):
    options = build_parser().parse_args(argv)
    _validate_options(options)
    index = build_geoid_water_index(
        options.geoid_root,
        metadata_filename=options.metadata_filename,
        min_valid_proportion=options.min_valid_proportion,
    )
    file_inventory = validate_geoid_files(index)
    counts = index.counts()
    print(json.dumps({'geoid_index': {
        'samples_per_split': {
            'train': counts['train'],
            'val': counts['val'],
        },
        **file_inventory,
    }}, sort_keys=True))
    channel_mean, channel_std = compute_geoid_train_channel_stats(
        index,
        progress=options.progress,
    )
    stats = {
        'polarizations': list(POLARIZATIONS),
        'channel_mean': [float(value) for value in channel_mean],
        'channel_std': [float(value) for value in channel_std],
        'radiometry': GEOID_RADIOMETRY,
        'min_valid_proportion': index.min_valid_proportion,
        'train_samples': counts['train'],
        'metadata_fingerprint': sampled_file_fingerprint(index.metadata_path),
        's1grd_files': file_inventory['s1grd_files'],
        'label_files': file_inventory['label_files'],
    }
    output = _write_constants(options.output, stats)
    print(json.dumps({'geoid_channel_stats': stats}, sort_keys=True))
    print(f'Wrote GEOID VV+VH constants: {output}')
    return output


if __name__ == '__main__':
    main()
