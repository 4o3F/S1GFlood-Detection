import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.parser import nonnegative_int, positive_int
from water_seg.dataset import (build_kulsary_scene_index,
                               get_water_test_loader, grid_signature,
                               resolve_sigma0_paths, source_fingerprints,
                               tile_split_records)
from water_seg.engine import close_loader, load_model_checkpoint, run_epoch
from water_seg.model import SwinTinyUNet


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Kulsary raw-Sigma0 water checkpoint'
    )
    parser.add_argument('--sigma0-root', type=Path, default=None)
    parser.add_argument('--sigma0-before', type=Path, default=None)
    parser.add_argument('--sigma0-peak', type=Path, default=None)
    parser.add_argument('--sigma0-after', type=Path, default=None)
    parser.add_argument('--mask-source', type=Path, default=None)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--num-workers', type=nonnegative_int, default=0)
    parser.add_argument('--device', default=None)
    return parser


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _samples_per_split(index):
    return {
        split: len(index.samples_for(split))
        for split in ('train', 'val', 'test')
    }


def _validate_reconstructed_index(index, config):
    if len(index.kept_tiles) != config['kept_tile_count']:
        raise ValueError(
            'reconstructed Kulsary tile count does not match checkpoint: '
            f'{len(index.kept_tiles)} vs {config["kept_tile_count"]}'
        )
    actual_counts = _samples_per_split(index)
    if actual_counts != config['samples_per_split']:
        raise ValueError(
            'reconstructed Kulsary split counts do not match checkpoint: '
            f'{actual_counts} vs {config["samples_per_split"]}'
        )
    actual_tiles = tile_split_records(index)
    if actual_tiles != config['tile_splits']:
        raise ValueError(
            'reconstructed Kulsary tile identities/splits do not match checkpoint'
        )
    actual_grid = grid_signature(index.grid)
    expected_grid = config['grid_signature']
    if (
        actual_grid['crs'] != expected_grid['crs']
        or actual_grid['width'] != expected_grid['width']
        or actual_grid['height'] != expected_grid['height']
        or not np.allclose(
            actual_grid['transform'],
            expected_grid['transform'],
            rtol=0.0,
            atol=1e-9,
        )
        or not np.allclose(
            actual_grid['peak_window'],
            expected_grid['peak_window'],
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError('reconstructed Kulsary grid does not match checkpoint')
    actual_fingerprints = source_fingerprints(index)
    if actual_fingerprints != config['source_fingerprints']:
        raise ValueError('Kulsary input content does not match checkpoint')


def main(argv=None):
    options = build_parser().parse_args(argv)
    device = _resolve_device(options.device)
    model = SwinTinyUNet(imagenet_pretrained=False)
    checkpoint = load_model_checkpoint(
        options.path,
        model,
        map_location='cpu',
        expected_architecture='SwinTinyUNet',
    )
    config = checkpoint['config']

    has_cli_sigma0 = options.sigma0_root is not None or any((
        options.sigma0_before is not None,
        options.sigma0_peak is not None,
        options.sigma0_after is not None,
    ))
    if has_cli_sigma0:
        sigma0_paths, _ = resolve_sigma0_paths(
            sigma0_root=options.sigma0_root,
            sigma0_before=options.sigma0_before,
            sigma0_peak=options.sigma0_peak,
            sigma0_after=options.sigma0_after,
        )
    else:
        sigma0_paths = {
            'before': Path(config['sigma0_before']),
            'peak': Path(config['sigma0_peak']),
            'after': Path(config['sigma0_after']),
        }
    mask_source = options.mask_source or Path(config['mask_source'])
    index = build_kulsary_scene_index(
        sigma0_paths['before'],
        sigma0_paths['peak'],
        sigma0_paths['after'],
        mask_source,
        db_min=config['db_min'],
        db_max=config['db_max'],
        block_tiles=config['block_tiles'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        split_seed=config['split_seed'],
    )
    _validate_reconstructed_index(index, config)
    model.to(device)

    test_loader = get_water_test_loader(
        index,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
    )
    try:
        metrics = run_epoch(
            model,
            test_loader,
            nn.CrossEntropyLoss(),
            device,
        )
    finally:
        close_loader(test_loader)
    output = {
        'checkpoint': str(Path(options.path).resolve()),
        'checkpoint_epoch': checkpoint['epoch'],
        'best_validation_water_iou': checkpoint['best_water_iou'],
        'test': metrics,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return metrics


if __name__ == '__main__':
    main()
