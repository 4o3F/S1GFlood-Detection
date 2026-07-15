import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


DEFAULT_SOURCE = Path('/mnt/6D437734319B5084/lhx/S1GFloods')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
SPLIT_DIRECTORIES = {'A': 'A', 'B': 'B', 'Label': 'GT'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert flat S1GFloods A/B/Label directories to train/val/test splits.'
    )
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help=f'flat dataset root (default: {DEFAULT_SOURCE})')
    parser.add_argument('--output', type=Path,
                        help='output root (default: <source>_prepared)')
    parser.add_argument('--train-count', type=int, default=4300)
    parser.add_argument('--val-count', type=int, default=530)
    parser.add_argument('--test-count', type=int, default=530)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed used for the deterministic shuffled split')
    parser.add_argument('--mode', choices=('copy', 'hardlink', 'symlink'), default='copy',
                        help='how source files are materialized in the output')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate and display the split without creating files')
    return parser.parse_args()


def collect_images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f'required input directory is missing: {directory}')

    images = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise ValueError(f'no supported images found in: {directory}')
    return images


def validate_pairs(images_by_source):
    reference_names = set(images_by_source['A'])
    errors = []

    for source_name in ('B', 'Label'):
        names = set(images_by_source[source_name])
        missing = sorted(reference_names - names)
        extra = sorted(names - reference_names)
        if missing:
            errors.append(f'{source_name} is missing {len(missing)} files, e.g. {missing[:5]}')
        if extra:
            errors.append(f'{source_name} has {len(extra)} extra files, e.g. {extra[:5]}')

    if errors:
        raise ValueError('\n'.join(errors))
    return sorted(reference_names)


def build_splits(names, train_count, val_count, test_count, seed):
    expected_count = train_count + val_count + test_count
    if min(train_count, val_count, test_count) < 0:
        raise ValueError('split counts must be non-negative')
    if len(names) != expected_count:
        raise ValueError(
            f'found {len(names)} paired samples, but split counts require {expected_count}'
        )

    shuffled_names = list(names)
    random.Random(seed).shuffle(shuffled_names)
    train_end = train_count
    val_end = train_end + val_count
    return {
        'train': shuffled_names[:train_end],
        'val': shuffled_names[train_end:val_end],
        'test': shuffled_names[val_end:],
    }


def materialize(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(source, destination)
    elif mode == 'hardlink':
        os.link(source, destination)
    else:
        relative_source = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative_source)


def write_manifest(output_root, source_root, splits, seed, mode):
    with (output_root / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(('split', 'filename'))
        for split_name, names in splits.items():
            writer.writerows((split_name, name) for name in names)

    metadata = {
        'source': str(source_root),
        'seed': seed,
        'mode': mode,
        'counts': {split_name: len(names) for split_name, names in splits.items()},
    }
    with (output_root / 'split_metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)
        file.write('\n')


def resolve_roots(args):
    source_root = args.source.expanduser().resolve()
    output_root = (
        args.output.expanduser().resolve()
        if args.output
        else source_root.with_name(f'{source_root.name}_prepared')
    )
    if source_root == output_root:
        raise ValueError('output must be different from source to preserve the original dataset')
    if output_root.exists():
        raise FileExistsError(f'output already exists: {output_root}')
    return source_root, output_root


def print_summary(source_root, output_root, splits, seed, mode):
    print(f'Source: {source_root}')
    print(f'Output: {output_root}')
    print(f'Mode: {mode}')
    print(f'Seed: {seed}')
    for split_name, split_names in splits.items():
        print(f'{split_name}: {len(split_names)}')
    print('Warning: this is a deterministic random split, not an official benchmark split manifest.')


def materialize_splits(staging_root, images_by_source, splits, mode):
    total_files = sum(len(names) for names in splits.values()) * len(SPLIT_DIRECTORIES)
    with tqdm(total=total_files, unit='file', desc='Preparing dataset') as progress:
        for split_name, split_names in splits.items():
            for filename in split_names:
                for source_name, destination_name in SPLIT_DIRECTORIES.items():
                    source = images_by_source[source_name][filename]
                    destination = staging_root / split_name / destination_name / filename
                    materialize(source, destination, mode)
                    progress.update(1)


def create_output(source_root, output_root, images_by_source, splits, seed, mode):
    staging_root = output_root.with_name(f'.{output_root.name}.partial')
    if staging_root.exists():
        raise FileExistsError(
            f'staging directory already exists from an earlier run: {staging_root}'
        )

    materialize_splits(staging_root, images_by_source, splits, mode)
    write_manifest(staging_root, source_root, splits, seed, mode)
    staging_root.rename(output_root)
    print(f'Dataset prepared successfully: {output_root}')


def prepare_dataset(args):
    source_root, output_root = resolve_roots(args)
    images_by_source = {
        source_name: collect_images(source_root / source_name)
        for source_name in SPLIT_DIRECTORIES
    }
    names = validate_pairs(images_by_source)
    splits = build_splits(
        names,
        args.train_count,
        args.val_count,
        args.test_count,
        args.seed,
    )
    print_summary(source_root, output_root, splits, args.seed, args.mode)

    if args.dry_run:
        print('Dry run completed; no files were created.')
        return
    create_output(
        source_root,
        output_root,
        images_by_source,
        splits,
        args.seed,
        args.mode,
    )


def main():
    args = parse_args()
    try:
        prepare_dataset(args)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
