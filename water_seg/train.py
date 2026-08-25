import argparse
import datetime
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils.kulsary_raster import sampled_file_fingerprint
from utils.parser import (epoch_count, finite_float, nonnegative_float,
                          nonnegative_int, positive_int)
from water_seg.dataset import (build_kulsary_scene_index,
                               compute_train_vv_stats, get_water_loaders,
                               grid_signature, resolve_sigma0_paths,
                               source_fingerprints, tile_split_records)
from water_seg.engine import (INPUT_CONTRACT, NORMALIZATION_CONTRACT,
                              build_optimizer, checkpoint_payload,
                              close_loader, load_initial_model_weights,
                              run_epoch, save_checkpoint, seed_everything)
from water_seg.model import SwinTinyUNet


GEOID_REFERENCE_COMMIT = 'b0ab63540a2a331513be306a5cbdc4ba88c766f5'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train a single-temporal Kulsary VV Swin-T U-Net'
    )
    parser.add_argument('--sigma0-root', type=Path, default=None)
    parser.add_argument('--sigma0-before', type=Path, default=None)
    parser.add_argument('--sigma0-peak', type=Path, default=None)
    parser.add_argument('--sigma0-after', type=Path, default=None)
    parser.add_argument('--mask-source', type=Path, required=True)
    parser.add_argument('--db-min', type=finite_float, default=-25.0)
    parser.add_argument('--db-max', type=finite_float, default=0.0)
    parser.add_argument('--block-tiles', type=positive_int, default=2)
    parser.add_argument('--train-ratio', type=finite_float, default=0.8)
    parser.add_argument('--val-ratio', type=finite_float, default=0.1)
    parser.add_argument('--test-ratio', type=finite_float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--epochs', type=epoch_count, default=20)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--num-workers', type=nonnegative_int, default=0)
    parser.add_argument('--encoder-lr', type=nonnegative_float, default=5e-5)
    parser.add_argument('--decoder-lr', type=nonnegative_float, default=5e-4)
    parser.add_argument('--weight-decay', type=nonnegative_float, default=0.01)
    parser.add_argument('--eta-min', type=nonnegative_float, default=1e-6)
    parser.add_argument('--early-stopping-patience', type=nonnegative_int, default=5)
    parser.add_argument('--min-iou-improvement', type=nonnegative_float, default=0.0)
    parser.add_argument('--seed', type=nonnegative_int, default=42)
    parser.add_argument('--save-dir', default='.tmp/water_swin_tiny_unet')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--init-checkpoint',
        type=Path,
        default=None,
        help=(
            'initialize model weights from water_seg GEOID pretraining or a '
            'compatible format-2 Kulsary checkpoint'
        ),
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='show train and validation batch progress bars',
    )
    parser.add_argument(
        '--augmentation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='enable uniform D4 augmentation for training samples',
    )
    parser.add_argument(
        '--imagenet-pretrained',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='initialize the one-channel Swin-T encoder from ImageNet weights',
    )
    return parser


def _validate_data_options(options):
    if options.db_min >= options.db_max:
        raise ValueError('--db-min must be smaller than --db-max')
    ratios = (
        options.train_ratio,
        options.val_ratio,
        options.test_ratio,
    )
    if any(value <= 0 for value in ratios):
        raise ValueError('split ratios must be positive')
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError('train/val/test ratios must sum to 1')


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _samples_per_split(index):
    return {
        split: len(index.samples_for(split))
        for split in ('train', 'val', 'test')
    }


def _serializable_config(
    options,
    index,
    vv_mean,
    vv_std,
    input_provenance,
    initialization,
):
    return {
        'epochs': options.epochs,
        'batch_size': options.batch_size,
        'num_workers': options.num_workers,
        'encoder_lr': options.encoder_lr,
        'decoder_lr': options.decoder_lr,
        'weight_decay': options.weight_decay,
        'eta_min': options.eta_min,
        'early_stopping_patience': options.early_stopping_patience,
        'min_iou_improvement': options.min_iou_improvement,
        'seed': options.seed,
        'augmentation': options.augmentation,
        'progress': options.progress,
        'imagenet_pretrained': (
            options.imagenet_pretrained and options.init_checkpoint is None
        ),
        'imagenet_pretrained_requested': options.imagenet_pretrained,
        'initialization': initialization,
        'architecture': 'SwinTinyUNet',
        'input': INPUT_CONTRACT,
        'in_chans': 1,
        'normalization': NORMALIZATION_CONTRACT,
        'vv_mean': float(vv_mean),
        'vv_std': float(vv_std),
        'db_min': index.db_min,
        'db_max': index.db_max,
        'sigma0_before': str(index.sigma0_paths['before']),
        'sigma0_peak': str(index.sigma0_paths['peak']),
        'sigma0_after': str(index.sigma0_paths['after']),
        'sigma0_input_mode': input_provenance['input_mode'],
        'sigma0_root': input_provenance['sigma0_root'],
        'sigma0_manifest': input_provenance['sigma0_manifest'],
        'sigma0_manifest_version': input_provenance['sigma0_manifest_version'],
        'mask_source': str(index.mask_source),
        'split_seed': index.split_seed,
        'block_tiles': index.block_tiles,
        'train_ratio': index.train_ratio,
        'val_ratio': index.val_ratio,
        'test_ratio': index.test_ratio,
        'kept_tile_count': len(index.kept_tiles),
        'samples_per_split': _samples_per_split(index),
        'grid_signature': grid_signature(index.grid),
        'tile_splits': tile_split_records(index),
        'source_fingerprints': source_fingerprints(index),
        'decoder_channels': [512, 256, 128, 64],
        'num_classes': 2,
        'geoid_reference_commit': GEOID_REFERENCE_COMMIT,
    }


def _write_metrics(writer, split, metrics, epoch):
    for name, value in metrics.items():
        if name == 'samples':
            continue
        writer.add_scalar(f'{split}/{name}', value, epoch)


def _print_epoch(epoch, train_metrics, val_metrics, optimizer):
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    summary = {
        'epoch': epoch,
        'encoder_lr': learning_rates[0],
        'decoder_lr': learning_rates[1],
        'train': train_metrics,
        'val': val_metrics,
    }
    print(json.dumps(summary, sort_keys=True))


def main(argv=None):
    options = build_parser().parse_args(argv)
    _validate_data_options(options)
    save_dir = Path(options.save_dir)
    seed_everything(options.seed)
    device = _resolve_device(options.device)

    sigma0_paths, input_provenance = resolve_sigma0_paths(
        sigma0_root=options.sigma0_root,
        sigma0_before=options.sigma0_before,
        sigma0_peak=options.sigma0_peak,
        sigma0_after=options.sigma0_after,
    )
    index = build_kulsary_scene_index(
        sigma0_paths['before'],
        sigma0_paths['peak'],
        sigma0_paths['after'],
        options.mask_source,
        db_min=options.db_min,
        db_max=options.db_max,
        block_tiles=options.block_tiles,
        train_ratio=options.train_ratio,
        val_ratio=options.val_ratio,
        test_ratio=options.test_ratio,
        split_seed=options.split_seed,
    )
    vv_mean, vv_std = compute_train_vv_stats(index)
    use_imagenet_initialization = (
        options.imagenet_pretrained and options.init_checkpoint is None
    )
    model = SwinTinyUNet(
        imagenet_pretrained=use_imagenet_initialization,
    )
    initialization = None
    if options.init_checkpoint is not None:
        init_path = options.init_checkpoint.expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(
                f'initialization checkpoint is missing: {init_path}'
            )
        init_checkpoint = load_initial_model_weights(
            init_path,
            model,
            map_location='cpu',
            expected_architecture='SwinTinyUNet',
        )
        initialization = {
            'path': str(init_path),
            'fingerprint': sampled_file_fingerprint(init_path),
            'kind': init_checkpoint.get('kind', 'kulsary-water-checkpoint'),
            'format_version': init_checkpoint.get('format_version'),
            'epoch': init_checkpoint.get('epoch'),
        }
    model.set_vv_normalization(vv_mean, vv_std).to(device)
    train_loader, val_loader = get_water_loaders(
        index,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        augmentation=options.augmentation,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        encoder_lr=options.encoder_lr,
        decoder_lr=options.decoder_lr,
        weight_decay=options.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=options.epochs,
        eta_min=options.eta_min,
    )

    config = _serializable_config(
        options,
        index,
        vv_mean,
        vv_std,
        input_provenance,
        initialization,
    )
    print(json.dumps({
        'dataset': {
            'kept_tiles': config['kept_tile_count'],
            'samples_per_split': config['samples_per_split'],
            'vv_mean': config['vv_mean'],
            'vv_std': config['vv_std'],
        }
    }, sort_keys=True))

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(str(save_dir / 'log' / timestamp))
    best_water_iou = float('-inf')
    best_epoch = None
    checks_without_improvement = 0
    stop_reason = f'reached the maximum of {options.epochs} epochs'

    try:
        for epoch in range(1, options.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                progress_description=(
                    f'Kulsary train {epoch}/{options.epochs}'
                    if options.progress else None
                ),
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                progress_description=(
                    f'Kulsary val {epoch}/{options.epochs}'
                    if options.progress else None
                ),
            )
            _write_metrics(writer, 'train', train_metrics, epoch)
            _write_metrics(writer, 'val', val_metrics, epoch)
            _print_epoch(epoch, train_metrics, val_metrics, optimizer)

            current_iou = val_metrics['water_iou']
            improved = current_iou > (
                best_water_iou + options.min_iou_improvement
            )
            if improved:
                best_water_iou = current_iou
                best_epoch = epoch
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            scheduler.step()
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                best_water_iou,
                train_metrics,
                val_metrics,
                config,
            )
            save_checkpoint(save_dir / 'last.pth', payload)
            if improved:
                save_checkpoint(save_dir / 'best.pth', payload)
                print(
                    f'Validation water IoU improved to {best_water_iou:.6f}; '
                    f'saved {save_dir / "best.pth"}'
                )

            if (
                options.early_stopping_patience > 0
                and checks_without_improvement
                >= options.early_stopping_patience
            ):
                stop_reason = (
                    f'early stopping after {checks_without_improvement} '
                    'validation checks without water-IoU improvement'
                )
                break
    finally:
        writer.close()
        close_loader(train_loader)
        close_loader(val_loader)

    print(f'Training complete: {stop_reason}.')
    if best_epoch is not None:
        print(
            f'Best validation water IoU: {best_water_iou:.6f} '
            f'at epoch {best_epoch}; checkpoint: {save_dir / "best.pth"}'
        )
    return save_dir / 'best.pth' if best_epoch is not None else None


if __name__ == '__main__':
    main()
