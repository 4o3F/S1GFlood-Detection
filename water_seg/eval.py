import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from utils.parser import nonnegative_int, positive_int
from water_seg.dataset import get_water_test_loader
from water_seg.engine import load_model_checkpoint, run_epoch
from water_seg.model import SwinTinyUNet


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a single-temporal VV water-segmentation checkpoint'
    )
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--num-workers', type=nonnegative_int, default=4)
    parser.add_argument('--device', default=None)
    return parser


def _resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    model.to(device)

    test_loader = get_water_test_loader(
        str(Path(options.dataset_dir).resolve()),
        batch_size=options.batch_size,
        num_workers=options.num_workers,
    )
    metrics = run_epoch(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )
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
