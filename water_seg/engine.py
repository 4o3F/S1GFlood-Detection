import os
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


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


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool = False

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0


def initialize_distributed(requested_device=None):
    """Initialize a torchrun environment, or return a single-process context."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        if requested_device:
            device = torch.device(requested_device)
        else:
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu'
            )
        return DistributedContext(0, 1, 0, device)

    if not dist.is_available():
        raise RuntimeError('torch.distributed is unavailable')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    requested = torch.device(requested_device) if requested_device else None
    force_cpu = requested is not None and requested.type == 'cpu'
    if torch.cuda.is_available() and not force_cpu:
        if local_rank >= torch.cuda.device_count():
            raise ValueError(
                f'LOCAL_RANK {local_rank} exceeds visible CUDA device count '
                f'{torch.cuda.device_count()}'
            )
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
    else:
        if requested is not None and requested.type != 'cpu':
            raise ValueError('CUDA was requested but is unavailable')
        device = torch.device('cpu')
        backend = 'gloo'

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend=backend, init_method='env://')
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError('torchrun rank metadata disagrees with process group')
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        initialized_here=initialized_here,
    )


def cleanup_distributed(context):
    if (
        context is not None
        and context.initialized_here
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def wrap_distributed_model(model, context):
    if not context.distributed:
        return model
    if context.device.type == 'cuda':
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model)


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def synchronize_model_buffers(model, context):
    if not context.distributed:
        return
    for buffer in unwrap_model(model).buffers():
        dist.broadcast(buffer, src=0)


def set_loader_epoch(loader, epoch):
    sampler = getattr(loader, 'sampler', None)
    if sampler is not None and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)


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


def _batch_confusion(targets, predictions, ignore_index=None):
    flat_targets = targets.reshape(-1)
    flat_predictions = predictions.reshape(-1)
    if flat_targets.shape != flat_predictions.shape:
        raise ValueError('target and prediction shapes must match')
    if ignore_index is not None:
        valid = flat_targets != ignore_index
        flat_targets = flat_targets[valid]
        flat_predictions = flat_predictions[valid]
    if flat_targets.numel() == 0:
        return np.zeros((2, 2), dtype=np.int64)
    if not bool(((flat_targets == 0) | (flat_targets == 1)).all()):
        raise ValueError('water targets must contain only 0, 1, or ignore_index')
    if not bool(((flat_predictions == 0) | (flat_predictions == 1)).all()):
        raise ValueError('water predictions must contain only classes 0 and 1')
    encoded = flat_targets * 2 + flat_predictions
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


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    progress_description=None,
):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((2, 2), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0
    valid_pixel_count = 0
    ignore_index = getattr(criterion, 'ignore_index', None)

    batches = tqdm(
        loader,
        desc=progress_description,
        unit='batch',
        dynamic_ncols=True,
        leave=False,
        disable=progress_description is None,
    )
    with torch.set_grad_enabled(training):
        for images, targets, _ in batches:
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
            if progress_description is not None:
                batches.set_postfix(loss=f'{loss.item():.4f}')
            if ignore_index is None:
                valid_pixel_count += targets.numel()
            else:
                valid_pixel_count += int((targets != ignore_index).sum().item())
            confusion += _batch_confusion(
                targets,
                predictions,
                ignore_index=ignore_index,
            )

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor(
            [
                loss_sum,
                sample_count,
                valid_pixel_count,
                *confusion.reshape(-1).tolist(),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sum = float(totals[0].item())
        sample_count = int(totals[1].item())
        valid_pixel_count = int(totals[2].item())
        confusion = (
            totals[3:].detach().cpu().numpy().astype(np.int64).reshape(2, 2)
        )

    if sample_count == 0:
        raise ValueError('water segmentation loader is empty')
    if valid_pixel_count == 0:
        raise ValueError('water segmentation loader has no supervised pixels')

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


def load_initial_model_weights(
    path,
    model,
    map_location='cpu',
    expected_architecture=None,
):
    """Load a compatible GEOID pretraining or Kulsary model state.

    This deliberately restores model tensors only. The caller must set the
    target dataset's VV normalization after loading.
    """
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError('initialization checkpoint must be a dictionary')
    required = {'model_state_dict', 'config'}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            f'initialization checkpoint is missing required keys: {missing}'
        )

    config = checkpoint['config']
    if not isinstance(config, dict):
        raise ValueError('initialization checkpoint config must be a dictionary')
    if checkpoint.get('kind') == 'geoid-water-pretraining':
        if checkpoint.get('format_version') != 1:
            raise ValueError('unsupported GEOID pretraining checkpoint format')
        for key, expected in (
            ('input', INPUT_CONTRACT),
            ('normalization', NORMALIZATION_CONTRACT),
            ('in_chans', 1),
        ):
            if config.get(key) != expected:
                raise ValueError(
                    f'GEOID pretraining checkpoint has incompatible {key}'
                )
    else:
        if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION:
            raise ValueError('unsupported initialization checkpoint format')
        _validate_checkpoint_config(config, expected_architecture)

    if expected_architecture is not None:
        architecture = config.get('architecture')
        if architecture != expected_architecture:
            raise ValueError(
                f'checkpoint architecture must be {expected_architecture}, '
                f'got {architecture}'
            )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return checkpoint
