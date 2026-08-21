import os
from pathlib import Path
import random

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


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
    if checkpoint.get('format_version') != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            'unsupported water checkpoint format: '
            f"{checkpoint.get('format_version')}"
        )

    required = {'epoch', 'best_water_iou', 'model_state_dict', 'config'}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(
            f'water checkpoint is missing required keys: {missing}'
        )
    if not isinstance(checkpoint['config'], dict):
        raise ValueError('water checkpoint config must be a dictionary')
    if expected_architecture is not None:
        architecture = checkpoint['config'].get('architecture')
        if architecture != expected_architecture:
            raise ValueError(
                f'checkpoint architecture must be {expected_architecture}, '
                f'got {architecture}'
            )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return checkpoint
