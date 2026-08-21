import os
from pathlib import Path
import random

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2
INPUT_CONTRACT = 'single VV channel, clipped dB'
NORMALIZATION_CONTRACT = 'train-split clipped-dB mean/std'
_REQUIRED_CONFIG_KEYS = {
    'architecture',
    'input',
    'in_chans',
    'normalization',
    'vv_mean',
    'vv_std',
    'db_min',
    'db_max',
    'sigma0_before',
    'sigma0_peak',
    'sigma0_after',
    'mask_source',
    'split_seed',
    'block_tiles',
    'train_ratio',
    'val_ratio',
    'test_ratio',
    'kept_tile_count',
    'samples_per_split',
    'grid_signature',
    'tile_splits',
    'source_fingerprints',
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _batch_confusion(targets, predictions):
    encoded = targets.reshape(-1) * 2 + predictions.reshape(-1)
    counts = torch.bincount(encoded, minlength=4)
    return counts.reshape(2, 2).detach().cpu().numpy().astype(np.int64)


def metrics_from_confusion(confusion):
    tn, fp, fn, tp = confusion.ravel()
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        'overall_accuracy': float(
            _safe_divide(tp + tn, tp + tn + fp + fn)
        ),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'water_iou': float(_safe_divide(tp, tp + fp + fn)),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0

    with torch.set_grad_enabled(training):
        for images, targets, _ in loader:
            images = images.float().to(device, non_blocking=True)
            targets = targets.long().to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            batch_size = targets.size(0)
            sample_count += batch_size
            loss_sum += loss.item() * batch_size
            confusion += _batch_confusion(targets, predictions)

    if sample_count == 0:
        raise ValueError('water segmentation loader is empty')

    metrics = metrics_from_confusion(confusion)
    metrics['loss'] = loss_sum / sample_count
    metrics['samples'] = sample_count
    return metrics


def close_loader(loader):
    if loader is None:
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None and hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    dataset = getattr(loader, 'dataset', None)
    if dataset is not None and hasattr(dataset, 'close'):
        dataset.close()


def build_optimizer(model, encoder_lr, decoder_lr, weight_decay):
    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith('encoder.')
    ]
    if not encoder_parameters or not decoder_parameters:
        raise ValueError('model must expose non-empty encoder and decoder parameters')

    return torch.optim.AdamW(
        (
            {'params': encoder_parameters, 'lr': encoder_lr},
            {'params': decoder_parameters, 'lr': decoder_lr},
        ),
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    best_water_iou,
    train_metrics,
    val_metrics,
    config,
):
    return {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_water_iou': best_water_iou,
        'train_metrics': dict(train_metrics),
        'val_metrics': dict(val_metrics),
        'config': dict(config),
    }


def save_checkpoint(path, payload):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def _validate_checkpoint_config(config, expected_architecture):
    if not isinstance(config, dict):
        raise ValueError('water checkpoint config must be a dictionary')
    missing = sorted(_REQUIRED_CONFIG_KEYS - config.keys())
    if missing:
        raise ValueError(
            f'water checkpoint config is missing required keys: {missing}'
        )
    if expected_architecture is not None:
        architecture = config.get('architecture')
        if architecture != expected_architecture:
            raise ValueError(
                f'checkpoint architecture must be {expected_architecture}, '
                f'got {architecture}'
            )
    if config.get('input') != INPUT_CONTRACT:
        raise ValueError(
            f'checkpoint input contract must be {INPUT_CONTRACT}'
        )
    if config.get('normalization') != NORMALIZATION_CONTRACT:
        raise ValueError(
            'checkpoint normalization contract must be '
            f'{NORMALIZATION_CONTRACT}'
        )
    if config.get('in_chans') != 1:
        raise ValueError('water checkpoint must use one VV input channel')

    vv_mean = float(config['vv_mean'])
    vv_std = float(config['vv_std'])
    if not np.isfinite(vv_mean):
        raise ValueError('checkpoint vv_mean must be finite')
    if not np.isfinite(vv_std) or vv_std <= 0:
        raise ValueError('checkpoint vv_std must be finite and positive')

    db_min = float(config['db_min'])
    db_max = float(config['db_max'])
    if not np.isfinite(db_min) or not np.isfinite(db_max) or db_min >= db_max:
        raise ValueError('checkpoint dB range is invalid')

    samples_per_split = config['samples_per_split']
    if not isinstance(samples_per_split, dict):
        raise ValueError('checkpoint samples_per_split must be a dictionary')
    for split in ('train', 'val', 'test'):
        value = samples_per_split.get(split)
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f'checkpoint samples_per_split[{split}] must be non-negative'
            )

    grid = config['grid_signature']
    if not isinstance(grid, dict):
        raise ValueError('checkpoint grid_signature must be a dictionary')
    required_grid = {'crs', 'transform', 'width', 'height', 'peak_window'}
    if required_grid - grid.keys():
        raise ValueError('checkpoint grid_signature is incomplete')
    if len(grid['transform']) != 6 or len(grid['peak_window']) != 4:
        raise ValueError('checkpoint grid_signature has invalid dimensions')

    kept_tile_count = config['kept_tile_count']
    if not isinstance(kept_tile_count, int) or kept_tile_count <= 0:
        raise ValueError('checkpoint kept_tile_count must be positive')
    tile_splits = config['tile_splits']
    if not isinstance(tile_splits, list) or len(tile_splits) != kept_tile_count:
        raise ValueError(
            'checkpoint tile_splits must contain one record per kept tile'
        )
    for record in tile_splits:
        if not isinstance(record, dict):
            raise ValueError('checkpoint tile_splits records must be dictionaries')
        if set(record) != {'row', 'col', 'split'}:
            raise ValueError('checkpoint tile_splits record has invalid fields')
        if record['split'] not in {'train', 'val', 'test'}:
            raise ValueError('checkpoint tile_splits has an invalid split')

    fingerprints = config['source_fingerprints']
    if not isinstance(fingerprints, dict) or set(fingerprints) != {'sigma0', 'masks'}:
        raise ValueError('checkpoint source_fingerprints has invalid groups')
    for group in ('sigma0', 'masks'):
        records = fingerprints[group]
        if not isinstance(records, dict) or set(records) != {'before', 'peak', 'after'}:
            raise ValueError(
                f'checkpoint source_fingerprints[{group}] has invalid roles'
            )
        for record in records.values():
            if not isinstance(record, dict) or set(record) != {
                'size',
                'sampled_sha256',
            }:
                raise ValueError('checkpoint source fingerprint is invalid')


def load_model_checkpoint(
    path,
    model,
    map_location='cpu',
    expected_architecture=None,
):
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError('water checkpoint must be a dictionary')
    format_version = checkpoint.get('format_version')
    if format_version != CHECKPOINT_FORMAT_VERSION:
        if format_version == 1:
            raise ValueError(
                'water checkpoint format 1 used quantized 0-255 VV with an '
                'RGB ImageNet stem; retrain for format 2 clipped-dB input'
            )
        raise ValueError(
            'unsupported water checkpoint format: '
            f'{format_version}'
        )

    required = {'epoch', 'best_water_iou', 'model_state_dict', 'config'}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            f'water checkpoint is missing required keys: {missing}'
        )
    _validate_checkpoint_config(
        checkpoint['config'],
        expected_architecture,
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if hasattr(model, 'vv_mean') and hasattr(model, 'vv_std'):
        config = checkpoint['config']
        model_mean = float(model.vv_mean.detach().cpu().item())
        model_std = float(model.vv_std.detach().cpu().item())
        if not np.isclose(model_mean, float(config['vv_mean'])):
            raise ValueError('checkpoint vv_mean disagrees with model state')
        if not np.isclose(model_std, float(config['vv_std'])):
            raise ValueError('checkpoint vv_std disagrees with model state')
    return checkpoint
