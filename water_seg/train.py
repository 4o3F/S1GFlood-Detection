import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from utils.parser import epoch_count, nonnegative_float, nonnegative_int, positive_int
from water_seg.dataset import get_water_loaders
from water_seg.engine import (build_optimizer, checkpoint_payload, run_epoch,
                              save_checkpoint, seed_everything)
from water_seg.model import SwinTinyUNet


GEOID_REFERENCE_COMMIT = 'b0ab63540a2a331513be306a5cbdc4ba88c766f5'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train a single-temporal VV Swin-T U-Net water segmenter'
    )
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--epochs', type=epoch_count, default=20)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--num-workers', type=nonnegative_int, default=4)
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
        '--augmentation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='enable uniform D4 augmentation for training samples',
    )
    parser.add_argument(
        '--imagenet-pretrained',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='initialize the Swin-T encoder from timm ImageNet weights',
    )
    return parser


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _serializable_config(options):
    return {
        'dataset_dir': str(Path(options.dataset_dir).resolve()),
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
        'imagenet_pretrained': options.imagenet_pretrained,
        'architecture': 'SwinTinyUNet',
        'input': 'single VV channel, raw 0-255',
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
    dataset_dir = str(Path(options.dataset_dir).resolve())
    save_dir = Path(options.save_dir)
    seed_everything(options.seed)
    device = _resolve_device(options.device)

    model = SwinTinyUNet(
        imagenet_pretrained=options.imagenet_pretrained,
    ).to(device)
    train_loader, val_loader = get_water_loaders(
        dataset_dir,
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

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(str(save_dir / 'log' / timestamp))
    config = _serializable_config(options)
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
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
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

    print(f'Training complete: {stop_reason}.')
    if best_epoch is not None:
        print(
            f'Best validation water IoU: {best_water_iou:.6f} '
            f'at epoch {best_epoch}; checkpoint: {save_dir / "best.pth"}'
        )
    return save_dir / 'best.pth' if best_epoch is not None else None


if __name__ == '__main__':
    main()
