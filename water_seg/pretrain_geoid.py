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
                              cleanup_distributed, close_loader,
                              collect_rng_states,
                              initialize_distributed, run_epoch,
                              load_checkpoint_file, restore_training_state,
                              save_checkpoint, seed_everything,
                              set_loader_epoch, synchronize_model_buffers,
                              unwrap_model, wrap_distributed_model)
from water_seg.geoid_dataset import (GEOID_IGNORE_INDEX,
                                     GEOID_METADATA_FILENAME,
                                     GEOID_PRETRAINING_FORMAT_VERSION,
                                     GEOID_PRETRAINING_KIND,
                                     build_geoid_water_index,
                                     get_geoid_water_loaders,
                                     validate_geoid_files)
from water_seg.geoid_stats import GEOID_CHANNEL_STATS
from utils.kulsary_raster import POLARIZATIONS
from water_seg.model import SwinTinyUNet


GEOID_REFERENCE_COMMIT = 'b0ab63540a2a331513be306a5cbdc4ba88c766f5'


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Pretrain the single-temporal VV+VH Swin-T U-Net on GEOID-Flood'
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
    parser.add_argument(
        '--save-dir',
        default='.tmp/geoid_swin_tiny_unet_vv_vh',
    )
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help=(
            'resume complete training state from a format-2 GEOID checkpoint; '
            'all training/data options, including --epochs, must match'
        ),
    )
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
        help='initialize the two-channel Swin-T encoder from ImageNet weights',
    )
    return parser


def _validate_options(options):
    if options.db_min >= options.db_max:
        raise ValueError('--db-min must be smaller than --db-max')
    if not 0.0 <= options.min_valid_proportion <= 1.0:
        raise ValueError('--min-valid-proportion must be between 0 and 1')


def _load_geoid_channel_constants(index):
    stats = GEOID_CHANNEL_STATS
    if not isinstance(stats, dict):
        raise RuntimeError(
            'GEOID VV+VH constants are not generated; run '
            '`python -m water_seg.compute_geoid_stats --geoid-root PATH` first'
        )
    required = {
        'polarizations',
        'channel_mean',
        'channel_std',
        'db_min',
        'db_max',
        'min_valid_proportion',
        'train_samples',
        'metadata_fingerprint',
    }
    missing = sorted(required - stats.keys())
    if missing:
        raise ValueError(
            f'GEOID VV+VH constants are missing fields: {missing}'
        )
    if tuple(stats['polarizations']) != POLARIZATIONS:
        raise ValueError(
            f'GEOID polarizations must be {POLARIZATIONS}'
        )
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
                f'GEOID VV+VH constants do not match current {key}; regenerate '
                'them with water_seg.compute_geoid_stats'
            )
    channel_mean = [float(value) for value in stats['channel_mean']]
    channel_std = [float(value) for value in stats['channel_std']]
    if len(channel_mean) != len(POLARIZATIONS) or not all(
        math.isfinite(value) for value in channel_mean
    ):
        raise ValueError('GEOID channel mean constants are invalid')
    if len(channel_std) != len(POLARIZATIONS) or not all(
        math.isfinite(value) and value > 0 for value in channel_std
    ):
        raise ValueError('GEOID channel std constants are invalid')
    return channel_mean, channel_std


def _serializable_config(
    options,
    index,
    channel_mean,
    channel_std,
    file_inventory,
    distributed_context,
):
    counts = index.counts()
    return {
        'epochs': options.epochs,
        'batch_size': options.batch_size,
        'batch_size_per_rank': options.batch_size,
        'global_batch_size': (
            options.batch_size * distributed_context.world_size
        ),
        'distributed': distributed_context.distributed,
        'world_size': distributed_context.world_size,
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
            options.imagenet_pretrained and options.resume is None
        ),
        'imagenet_pretrained_requested': options.imagenet_pretrained,
        'architecture': 'SwinTinyUNet',
        'input': INPUT_CONTRACT,
        'in_chans': len(POLARIZATIONS),
        'polarizations': list(POLARIZATIONS),
        'normalization': NORMALIZATION_CONTRACT,
        'channel_mean': [float(value) for value in channel_mean],
        'channel_std': [float(value) for value in channel_std],
        'db_min': index.db_min,
        'db_max': index.db_max,
        'geoid_root': str(index.root),
        'metadata_path': str(index.metadata_path),
        'metadata_fingerprint': sampled_file_fingerprint(index.metadata_path),
        'file_inventory': dict(file_inventory),
        'min_valid_proportion': index.min_valid_proportion,
        'modality': 's1grd',
        'bands': list(POLARIZATIONS),
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
    best_epoch,
    checks_without_improvement,
    rng_states,
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
        best_epoch=best_epoch,
        checks_without_improvement=checks_without_improvement,
        rng_states=rng_states,
        format_version=GEOID_PRETRAINING_FORMAT_VERSION,
    )
    payload['kind'] = GEOID_PRETRAINING_KIND
    return payload


def _run(options, distributed_context):
    _validate_options(options)
    save_dir = Path(options.save_dir)
    seed_everything(options.seed + distributed_context.rank)
    device = distributed_context.device

    index = build_geoid_water_index(
        options.geoid_root,
        metadata_filename=options.metadata_filename,
        db_min=options.db_min,
        db_max=options.db_max,
        min_valid_proportion=options.min_valid_proportion,
    )
    file_inventory = validate_geoid_files(index)
    counts = index.counts()
    if distributed_context.is_main:
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
        if distributed_context.is_main:
            print('GEOID S1-GRD/label selection is complete.')
        return None
    channel_mean, channel_std = _load_geoid_channel_constants(index)
    model = SwinTinyUNet(
        imagenet_pretrained=(
            options.imagenet_pretrained
            and options.resume is None
            and distributed_context.is_main
        ),
    ).set_channel_normalization(channel_mean, channel_std).to(device)
    train_loader, val_loader = get_geoid_water_loaders(
        index,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        augmentation=options.augmentation,
        distributed_context=distributed_context,
        sampler_seed=options.seed,
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
    training_model = wrap_distributed_model(model, distributed_context)
    config = _serializable_config(
        options,
        index,
        channel_mean,
        channel_std,
        file_inventory,
        distributed_context,
    )
    if distributed_context.is_main:
        print(json.dumps({'dataset': {
            'samples_per_split': config['samples_per_split'],
            'samples_per_time': config['samples_per_time'],
            'channel_mean': config['channel_mean'],
            'channel_std': config['channel_std'],
            'ignored_label': config['ignore_index'],
            'world_size': config['world_size'],
            'global_batch_size': config['global_batch_size'],
        }}, sort_keys=True))

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = (
        SummaryWriter(str(save_dir / 'log' / timestamp))
        if distributed_context.is_main else None
    )
    start_epoch = 1
    best_water_iou = float('-inf')
    best_epoch = None
    checks_without_improvement = 0
    if options.resume is not None:
        resume_path = options.resume.expanduser().resolve()
        resume_checkpoint = load_checkpoint_file(resume_path)
        resume_state = restore_training_state(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            config,
            distributed_context,
            expected_format_version=GEOID_PRETRAINING_FORMAT_VERSION,
            expected_kind=GEOID_PRETRAINING_KIND,
            expected_architecture='SwinTinyUNet',
        )
        start_epoch = resume_state['start_epoch']
        best_water_iou = resume_state['best_water_iou']
        best_epoch = resume_state['best_epoch']
        checks_without_improvement = resume_state[
            'checks_without_improvement'
        ]
        config['resume'] = {
            'path': str(resume_path),
            'fingerprint': sampled_file_fingerprint(resume_path),
            'epoch': int(resume_checkpoint['epoch']),
        }
        if start_epoch > options.epochs:
            raise ValueError(
                f'--epochs={options.epochs} must be at least the next resume '
                f'epoch {start_epoch}'
            )
        if distributed_context.is_main:
            print(
                f'Resuming GEOID pretraining at epoch {start_epoch} from '
                f'{resume_path}'
            )
    stop_reason = f'reached the maximum of {options.epochs} epochs'

    try:
        for epoch in range(start_epoch, options.epochs + 1):
            set_loader_epoch(train_loader, epoch)
            train_metrics = run_epoch(
                training_model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                progress_description=(
                    f'GEOID train {epoch}/{options.epochs}'
                    if options.progress and distributed_context.is_main
                    else None
                ),
            )
            synchronize_model_buffers(training_model, distributed_context)
            val_metrics = run_epoch(
                unwrap_model(training_model),
                val_loader,
                criterion,
                device,
                progress_description=(
                    f'GEOID val {epoch}/{options.epochs}'
                    if options.progress and distributed_context.is_main
                    else None
                ),
            )
            if distributed_context.is_main:
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
            rng_states = collect_rng_states(distributed_context)
            if distributed_context.is_main:
                payload = _pretraining_payload(
                    unwrap_model(training_model),
                    optimizer,
                    scheduler,
                    epoch,
                    best_water_iou,
                    train_metrics,
                    val_metrics,
                    config,
                    best_epoch,
                    checks_without_improvement,
                    rng_states,
                )
                save_checkpoint(save_dir / 'last.pth', payload)
                if improved:
                    save_checkpoint(save_dir / 'best.pth', payload)
                    print(
                        'Validation water IoU improved to '
                        f'{best_water_iou:.6f}; '
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
        if writer is not None:
            writer.close()
        close_loader(train_loader)
        close_loader(val_loader)

    if distributed_context.is_main:
        print(f'GEOID pretraining complete: {stop_reason}.')
        if best_epoch is not None:
            print(
                f'Best validation water IoU: {best_water_iou:.6f} '
                f'at epoch {best_epoch}; checkpoint: {save_dir / "best.pth"}'
            )
    return save_dir / 'best.pth' if best_epoch is not None else None


def main(argv=None):
    options = build_parser().parse_args(argv)
    distributed_context = initialize_distributed(options.device)
    try:
        return _run(options, distributed_context)
    finally:
        cleanup_distributed(distributed_context)


if __name__ == '__main__':
    main()
