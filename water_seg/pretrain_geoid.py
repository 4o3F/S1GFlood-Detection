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
from water_seg.engine import (INPUT_CONTRACT, NORMALIZATION_CONTRACT,
                              build_optimizer, checkpoint_payload,
                              close_loader, run_epoch, save_checkpoint,
                              seed_everything)
from water_seg.geoid_dataset import (GEOID_IGNORE_INDEX,
                                     GEOID_METADATA_FILENAME,
                                     GEOID_PRETRAINING_FORMAT_VERSION,
                                     GEOID_PRETRAINING_KIND,
                                     build_geoid_water_index,
                                     get_geoid_water_loaders,
                                     validate_geoid_files)
from water_seg.geoid_stats import GEOID_VV_STATS
from water_seg.model import SwinTinyUNet


GEOID_REFERENCE_COMMIT = 'b0ab63540a2a331513be306a5cbdc4ba88c766f5'


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Pretrain the single-temporal VV Swin-T U-Net on GEOID-Flood'
        )
    )
    parser.add_argument('--geoid-root', type=Path, required=True)
    parser.add_argument(
        '--metadata-filename',
        default=GEOID_METADATA_FILENAME,
    )
    parser.add_argument('--db-min', type=finite_float, default=-25.0)
    parser.add_argument('--db-max', type=finite_float, default=0.0)
    parser.add_argument(
        '--min-valid-proportion',
        type=finite_float,
        default=0.01,
    )
    parser.add_argument('--epochs', type=epoch_count, default=20)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--num-workers', type=nonnegative_int, default=0)
    parser.add_argument('--encoder-lr', type=nonnegative_float, default=5e-5)
    parser.add_argument('--decoder-lr', type=nonnegative_float, default=5e-4)
    parser.add_argument('--weight-decay', type=nonnegative_float, default=0.01)
    parser.add_argument('--eta-min', type=nonnegative_float, default=1e-6)
    parser.add_argument(
        '--early-stopping-patience',
        type=nonnegative_int,
        default=5,
    )
    parser.add_argument(
        '--min-iou-improvement',
        type=nonnegative_float,
        default=0.0,
    )
    parser.add_argument('--seed', type=nonnegative_int, default=42)
    parser.add_argument('--save-dir', default='.tmp/geoid_swin_tiny_unet')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='validate CSV selection and referenced files without training',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='show source and batch progress bars',
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


def _validate_options(options):
    if options.db_min >= options.db_max:
        raise ValueError('--db-min must be smaller than --db-max')
    if not 0.0 <= options.min_valid_proportion <= 1.0:
        raise ValueError('--min-valid-proportion must be between 0 and 1')


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _load_geoid_vv_constants(index):
    stats = GEOID_VV_STATS
    if not isinstance(stats, dict):
        raise RuntimeError(
            'GEOID VV constants are not generated; run '
            '`python -m water_seg.compute_geoid_stats --geoid-root PATH` first'
        )
    required = {
        'vv_mean',
        'vv_std',
        'db_min',
        'db_max',
        'min_valid_proportion',
        'train_samples',
        'metadata_fingerprint',
    }
    missing = sorted(required - stats.keys())
    if missing:
        raise ValueError(f'GEOID VV constants are missing fields: {missing}')
    counts = index.counts()
    expected = {
        'db_min': index.db_min,
        'db_max': index.db_max,
        'min_valid_proportion': index.min_valid_proportion,
        'train_samples': counts['train'],
        'metadata_fingerprint': sampled_file_fingerprint(index.metadata_path),
    }
    for key, expected_value in expected.items():
        if stats[key] != expected_value:
            raise ValueError(
                f'GEOID VV constants do not match current {key}; regenerate '
                'them with water_seg.compute_geoid_stats'
            )
    vv_mean = float(stats['vv_mean'])
    vv_std = float(stats['vv_std'])
    if not math.isfinite(vv_mean):
        raise ValueError('GEOID VV mean constant must be finite')
    if not math.isfinite(vv_std) or vv_std <= 0:
        raise ValueError('GEOID VV std constant must be finite and positive')
    return vv_mean, vv_std


def _serializable_config(options, index, vv_mean, vv_std, file_inventory):
    counts = index.counts()
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
        'imagenet_pretrained': options.imagenet_pretrained,
        'architecture': 'SwinTinyUNet',
        'input': INPUT_CONTRACT,
        'in_chans': 1,
        'normalization': NORMALIZATION_CONTRACT,
        'vv_mean': float(vv_mean),
        'vv_std': float(vv_std),
        'db_min': index.db_min,
        'db_max': index.db_max,
        'geoid_root': str(index.root),
        'metadata_path': str(index.metadata_path),
        'metadata_fingerprint': sampled_file_fingerprint(index.metadata_path),
        'file_inventory': dict(file_inventory),
        'min_valid_proportion': index.min_valid_proportion,
        'modality': 's1grd',
        'band': 'VV',
        'image_scope': ['pre', 'post'],
        'ignore_index': GEOID_IGNORE_INDEX,
        'label_mapping': {
            'pre': {'0': 0, '1': 1, '2': 0, '255': 255},
            'post': {'0': 0, '1': 1, '2': 1, '255': 255},
        },
        'samples_per_split': {
            'train': counts['train'],
            'val': counts['val'],
        },
        'samples_per_time': {
            'pre': counts['pre'],
            'post': counts['post'],
        },
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
    print(json.dumps({
        'epoch': epoch,
        'encoder_lr': learning_rates[0],
        'decoder_lr': learning_rates[1],
        'train': train_metrics,
        'val': val_metrics,
    }, sort_keys=True))


def _pretraining_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    best_water_iou,
    train_metrics,
    val_metrics,
    config,
):
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
    payload['kind'] = GEOID_PRETRAINING_KIND
    payload['format_version'] = GEOID_PRETRAINING_FORMAT_VERSION
    return payload


def main(argv=None):
    options = build_parser().parse_args(argv)
    _validate_options(options)
    save_dir = Path(options.save_dir)
    seed_everything(options.seed)
    device = _resolve_device(options.device)

    index = build_geoid_water_index(
        options.geoid_root,
        metadata_filename=options.metadata_filename,
        db_min=options.db_min,
        db_max=options.db_max,
        min_valid_proportion=options.min_valid_proportion,
    )
    file_inventory = validate_geoid_files(index)
    counts = index.counts()
    print(json.dumps({'geoid_index': {
        'samples_per_split': {
            'train': counts['train'],
            'val': counts['val'],
        },
        'samples_per_time': {
            'pre': counts['pre'],
            'post': counts['post'],
        },
        **file_inventory,
    }}, sort_keys=True))
    if options.validate_only:
        print('GEOID S1-GRD/label selection is complete.')
        return None
    vv_mean, vv_std = _load_geoid_vv_constants(index)
    model = SwinTinyUNet(
        imagenet_pretrained=options.imagenet_pretrained,
    ).set_vv_normalization(vv_mean, vv_std).to(device)
    train_loader, val_loader = get_geoid_water_loaders(
        index,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        augmentation=options.augmentation,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=GEOID_IGNORE_INDEX)
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
        file_inventory,
    )
    print(json.dumps({'dataset': {
        'samples_per_split': config['samples_per_split'],
        'samples_per_time': config['samples_per_time'],
        'vv_mean': config['vv_mean'],
        'vv_std': config['vv_std'],
        'ignored_label': config['ignore_index'],
    }}, sort_keys=True))

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
                    f'GEOID train {epoch}/{options.epochs}'
                    if options.progress else None
                ),
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                progress_description=(
                    f'GEOID val {epoch}/{options.epochs}'
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
            payload = _pretraining_payload(
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

    print(f'GEOID pretraining complete: {stop_reason}.')
    if best_epoch is not None:
        print(
            f'Best validation water IoU: {best_water_iou:.6f} '
            f'at epoch {best_epoch}; checkpoint: {save_dir / "best.pth"}'
        )
    return save_dir / 'best.pth' if best_epoch is not None else None


if __name__ == '__main__':
    main()
