import datetime
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.helpers import (get_criterion, get_loaders, get_mean_metrics,
                           initialize_metrics, load_model, set_metrics)
from utils.parser import parser_with_args


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_scheduler(optimizer, opt, lr_policy):
    if lr_policy == 'linear':
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 - epoch / float(opt.epochs + 1),
        )
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=opt.epochs // 3,
            gamma=0.1,
        )
    else:
        raise NotImplementedError(f'LR policy [{lr_policy}] is not implemented')
    return scheduler


def should_validate(epoch_number, total_epochs, interval):
    return epoch_number % interval == 0 or epoch_number == total_epochs


def is_significant_improvement(current_f1, best_f1, min_delta):
    return current_f1 > best_f1 + min_delta


def should_stop_early(checks_without_improvement, patience):
    """Return True when early stopping is enabled and patience is exhausted.

    patience <= 0 disables early stopping so full epoch budgets (e.g. paper
    100-epoch runs) can complete without changing the default workflow.
    """
    return patience > 0 and checks_without_improvement >= patience


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def binary_metrics_from_confusion(confusion):
    tn, fp, fn, tp = confusion.ravel()
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    return {
        'overall_accuracy': accuracy,
        'precisions': precision,
        'recalls': recall,
        'f1_scores': f1_score,
    }


def binary_iou_from_confusion(confusion):
    _, fp, fn, tp = confusion.ravel()
    return safe_divide(tp, tp + fp + fn)


def move_targets_to_device(targets, device):
    return {
        name: value.to(device)
        for name, value in targets.items()
    }


def _change_logits(outputs):
    return outputs['change_logits'] if isinstance(outputs, dict) else outputs


def compute_multitask_loss(
    outputs,
    targets,
    criterion,
    water_loss_weight,
):
    change_loss = criterion(
        _change_logits(outputs),
        targets['change'].long(),
    )
    valid = targets['water_valid'].bool()
    supervised_count = (
        int(valid.sum().item())
        if water_loss_weight > 0
        else 0
    )

    if supervised_count:
        if not isinstance(outputs, dict):
            raise ValueError('auxiliary logits are required for water-supervised samples')
        water_logits = torch.cat(
            (
                outputs['water_a_logits'][valid],
                outputs['water_b_logits'][valid],
            ),
            dim=0,
        )
        water_targets = torch.cat(
            (
                targets['water_a'][valid].long(),
                targets['water_b'][valid].long(),
            ),
            dim=0,
        )
        water_loss = criterion(water_logits, water_targets)
    else:
        water_loss = change_loss.new_zeros(())

    return {
        'total': change_loss + water_loss_weight * water_loss,
        'change': change_loss,
        'water': water_loss,
        'water_supervised_samples': supervised_count,
    }


def _add_confusion(confusion, targets, predictions):
    confusion += confusion_matrix(
        targets.detach().cpu().numpy().ravel(),
        predictions.detach().cpu().numpy().ravel(),
        labels=[0, 1],
    )


def _add_water_metrics(metrics, prefix, confusion):
    binary = binary_metrics_from_confusion(confusion)
    metrics[f'{prefix}_precision'] = binary['precisions']
    metrics[f'{prefix}_recall'] = binary['recalls']
    metrics[f'{prefix}_f1'] = binary['f1_scores']
    metrics[f'{prefix}_iou'] = binary_iou_from_confusion(confusion)


def validate(
    model,
    val_loader,
    criterion,
    device,
    learning_rate,
    water_loss_weight=0.2,
):
    change_confusion = np.zeros((2, 2), dtype=np.int64)
    water_a_confusion = np.zeros((2, 2), dtype=np.int64)
    water_b_confusion = np.zeros((2, 2), dtype=np.int64)
    change_loss_sum = 0.0
    water_loss_sum = 0.0
    water_supervised_samples = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for img1, img2, targets, _ in val_loader:
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            targets = move_targets_to_device(targets, device)
            valid = targets['water_valid'].bool()
            request_aux = water_loss_weight > 0 and bool(valid.any())
            outputs = model(img1, img2, return_aux=True) if request_aux else model(img1, img2)
            losses = compute_multitask_loss(
                outputs,
                targets,
                criterion,
                water_loss_weight,
            )

            change_logits = _change_logits(outputs)
            change_predictions = torch.argmax(change_logits, dim=1)
            batch_size = targets['change'].size(0)
            change_loss_sum += losses['change'].item() * batch_size
            total_samples += batch_size
            _add_confusion(
                change_confusion,
                targets['change'],
                change_predictions,
            )

            valid_count = losses['water_supervised_samples']
            if valid_count:
                water_supervised_samples += valid_count
                water_loss_sum += losses['water'].item() * valid_count
                _add_confusion(
                    water_a_confusion,
                    targets['water_a'][valid],
                    torch.argmax(outputs['water_a_logits'][valid], dim=1),
                )
                _add_confusion(
                    water_b_confusion,
                    targets['water_b'][valid],
                    torch.argmax(outputs['water_b_logits'][valid], dim=1),
                )

    if total_samples == 0:
        raise ValueError('validation loader is empty')

    change_loss = change_loss_sum / total_samples
    water_loss = (
        water_loss_sum / water_supervised_samples
        if water_supervised_samples
        else 0.0
    )
    metrics = binary_metrics_from_confusion(change_confusion)
    metrics['change_losses'] = change_loss
    metrics['water_losses'] = water_loss
    metrics['weighted_water_losses'] = water_loss_weight * water_loss
    metrics['losses'] = change_loss + water_loss_weight * water_loss
    metrics['learning_rate'] = learning_rate
    metrics['water_supervised_samples'] = water_supervised_samples
    metrics['water_supervision_fraction'] = (
        water_supervised_samples / total_samples
    )

    if water_supervised_samples:
        _add_water_metrics(metrics, 'water_a', water_a_confusion)
        _add_water_metrics(metrics, 'water_b', water_b_confusion)
        _add_water_metrics(
            metrics,
            'water_pooled',
            water_a_confusion + water_b_confusion,
        )
    return metrics


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, opt,
                    device, writer, epoch_number, total_step):
    train_metrics = initialize_metrics()
    model.train()
    logging.info('Starting training phase')
    batch_iter = 0

    tbar = tqdm(train_loader)
    for img1, img2, targets, _ in tbar:
        tbar.set_description(
            f'epoch {epoch_number} info {batch_iter} - {batch_iter + opt.batch_size}'
        )
        batch_iter += opt.batch_size
        total_step += 1

        img1 = img1.float().to(device)
        img2 = img2.float().to(device)
        targets = move_targets_to_device(targets, device)
        valid = targets['water_valid'].bool()
        request_aux = opt.water_loss_weight > 0 and bool(valid.any())

        optimizer.zero_grad()
        outputs = model(img1, img2, return_aux=True) if request_aux else model(img1, img2)
        losses = compute_multitask_loss(
            outputs,
            targets,
            criterion,
            opt.water_loss_weight,
        )
        losses['total'].backward()
        optimizer.step()

        labels = targets['change'].long()
        predictions = torch.argmax(_change_logits(outputs), dim=1)
        overall_accuracy = (predictions == labels).float().mean()
        train_report = precision_recall_fscore_support(
            labels.detach().cpu().numpy().ravel(),
            predictions.detach().cpu().numpy().ravel(),
            average='binary',
            pos_label=1,
            zero_division=0,
        )
        train_metrics = set_metrics(
            train_metrics,
            losses['total'],
            overall_accuracy,
            train_report,
            scheduler.get_last_lr(),
            change_loss=losses['change'],
            water_loss=losses['water'],
            water_loss_weight=opt.water_loss_weight,
            water_supervised_samples=losses['water_supervised_samples'],
            batch_size=labels.size(0),
        )
        mean_train_metrics = get_mean_metrics(train_metrics)
        for name, value in mean_train_metrics.items():
            writer.add_scalars(str(name), {'train': value}, total_step)

    if not train_metrics['losses']:
        raise ValueError('training loader is empty')
    return get_mean_metrics(train_metrics), total_step


# DAM-Net paper training uses AdamW for all supported backbones.
# Keep an explicit SGD path only for optional non-paper experiments.
ADAMW_DEFAULTS = {
    'lr': 6e-5,
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,
}
SGD_DEFAULTS = {
    'lr': 0.001,
    'momentum': 0.99,
    'weight_decay': 0.0005,
}


def resolve_optimizer_type(backbone, optimizer_type=None):
    """Select optimizer type for DAM-Net training.

    Default is AdamW for every backbone (including vitae). Pass
    optimizer_type='sgd' only when an explicit SGD experiment is requested.
    """
    if optimizer_type is None:
        return 'adamw'
    normalized = optimizer_type.lower()
    if normalized not in {'adamw', 'sgd'}:
        raise ValueError(f'unsupported optimizer_type: {optimizer_type}')
    return normalized


def build_optimizer(model, backbone, optimizer_type=None):
    selected = resolve_optimizer_type(backbone, optimizer_type)
    if selected == 'sgd':
        return torch.optim.SGD(model.parameters(), **SGD_DEFAULTS)
    return torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)


def main():
    parser, _ = parser_with_args()
    opt = parser.parse_args()
    opt.dataset_dir = os.path.join(os.path.abspath(opt.dataset_dir), '')

    seed_torch(seed=42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_name = f'{opt.dataset}_{opt.backbone}_{opt.mode}'
    save_path = f'.tmp/{run_name}'
    writer = SummaryWriter(
        f'.tmp/log/{run_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    model = load_model(opt, device)
    train_loader, val_loader = get_loaders(opt)
    criterion = get_criterion(opt)
    optimizer = build_optimizer(model, opt.backbone)
    scheduler = get_scheduler(optimizer, opt, 'linear')

    best_f1 = float('-inf')
    best_epoch = None
    best_checkpoint = None
    checks_without_improvement = 0
    total_step = -1
    stop_reason = f'reached the maximum of {opt.epochs} epochs'

    try:
        for epoch_index in range(opt.epochs):
            epoch_number = epoch_index + 1
            train_metrics, total_step = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                opt,
                device,
                writer,
                epoch_number,
                total_step,
            )
            scheduler.step()
            logging.info(f'EPOCH {epoch_number} TRAIN METRICS: {train_metrics}')

            if not should_validate(
                epoch_number,
                opt.epochs,
                opt.validation_interval,
            ):
                print(f'Epoch {epoch_number} completed without validation.')
                continue

            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                scheduler.get_last_lr()[0],
                opt.water_loss_weight,
            )
            logging.info(f'EPOCH {epoch_number} VALIDATION METRICS: {val_metrics}')
            for name, value in val_metrics.items():
                writer.add_scalars(str(name), {'val': value}, epoch_number)

            current_f1 = val_metrics['f1_scores']
            print(f'Epoch {epoch_number} validation metrics: {val_metrics}')
            if is_significant_improvement(
                current_f1,
                best_f1,
                opt.min_f1_improvement,
            ):
                previous_best = best_f1
                best_f1 = current_f1
                best_epoch = epoch_number
                checks_without_improvement = 0
                os.makedirs(save_path, exist_ok=True)
                best_checkpoint = (
                    f'{save_path}/checkpoint_epoch_{epoch_number}.pth'
                )
                torch.save(model, best_checkpoint)
                print(
                    f'Validation F1 improved from {previous_best:.6f} '
                    f'to {best_f1:.6f}; saved {best_checkpoint}'
                )
            else:
                checks_without_improvement += 1
                print(
                    'Validation F1 did not improve by more than '
                    f'{opt.min_f1_improvement:.6f}: '
                    f'{checks_without_improvement}/'
                    f'{opt.early_stopping_patience}'
                )

            if should_stop_early(
                checks_without_improvement,
                opt.early_stopping_patience,
            ):
                stop_reason = (
                    f'early stopping after {checks_without_improvement} '
                    'validation checks without significant F1 improvement'
                )
                break
    finally:
        writer.close()

    print(f'Training complete: {stop_reason}.')
    if best_checkpoint:
        print(
            f'Best validation F1: {best_f1:.6f} at epoch {best_epoch}; '
            f'checkpoint: {best_checkpoint}'
        )
    else:
        print('No checkpoint was saved.')


if __name__ == '__main__':
    main()
