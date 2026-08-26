import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from water_seg.dataset import _water_loader
from water_seg.engine import (build_optimizer, cleanup_distributed,
                              initialize_distributed, run_epoch,
                              set_loader_epoch, synchronize_model_buffers,
                              unwrap_model, wrap_distributed_model)


class TinyDistributedWaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.head = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, image):
        return self.head(self.encoder(image))


def _records(count):
    records = []
    for index in range(count):
        image = torch.full((1, 2, 2), float(index % 2))
        target = torch.full((2, 2), index % 2, dtype=torch.long)
        records.append((image, target, f'sample-{index}'))
    return records


def _distributed_worker(rank, world_size, master_port, output_directory):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    torch.set_num_threads(1)

    context = initialize_distributed('cpu')
    try:
        torch.manual_seed(13)
        base_model = TinyDistributedWaterModel().to(context.device)
        optimizer = build_optimizer(
            base_model,
            encoder_lr=1e-2,
            decoder_lr=1e-2,
            weight_decay=0.0,
        )
        model = wrap_distributed_model(base_model, context)
        train_loader = _water_loader(
            _records(4),
            batch_size=1,
            num_workers=0,
            shuffle=True,
            distributed_context=context,
            sampler_seed=17,
        )
        set_loader_epoch(train_loader, 1)
        train_metrics = run_epoch(
            model,
            train_loader,
            nn.CrossEntropyLoss(),
            context.device,
            optimizer=optimizer,
        )
        synchronize_model_buffers(model, context)

        val_loader = _water_loader(
            _records(3),
            batch_size=1,
            num_workers=0,
            shuffle=False,
            distributed_context=context,
        )
        val_metrics = run_epoch(
            unwrap_model(model),
            val_loader,
            nn.CrossEntropyLoss(),
            context.device,
        )
        state_keys = list(unwrap_model(model).state_dict())
        weight_sum = sum(
            float(parameter.detach().sum())
            for parameter in unwrap_model(model).parameters()
        )
        result = {
            'rank': context.rank,
            'world_size': context.world_size,
            'train_samples': train_metrics['samples'],
            'val_samples': val_metrics['samples'],
            'weight_sum': weight_sum,
            'state_has_module_prefix': any(
                key.startswith('module.') for key in state_keys
            ),
            'train_indices': list(iter(train_loader.sampler)),
            'val_indices': list(iter(val_loader.sampler)),
        }
        output = Path(output_directory) / f'rank-{rank}.json'
        output.write_text(json.dumps(result), encoding='utf-8')
    finally:
        cleanup_distributed(context)


class WaterDistributedTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.distributed.is_available(),
        'torch.distributed is unavailable',
    )
    def test_two_process_training_and_exact_validation_sharding(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            master_port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_worker,
                args=(2, master_port, directory),
                nprocs=2,
                join=True,
            )
            results = [
                json.loads(
                    (Path(directory) / f'rank-{rank}.json').read_text(
                        encoding='utf-8'
                    )
                )
                for rank in range(2)
            ]

        self.assertEqual([result['rank'] for result in results], [0, 1])
        self.assertTrue(all(result['world_size'] == 2 for result in results))
        self.assertTrue(all(result['train_samples'] == 4 for result in results))
        self.assertTrue(all(result['val_samples'] == 3 for result in results))
        self.assertFalse(any(
            result['state_has_module_prefix'] for result in results
        ))
        self.assertAlmostEqual(
            results[0]['weight_sum'],
            results[1]['weight_sum'],
            places=6,
        )
        train_indices = results[0]['train_indices'] + results[1]['train_indices']
        val_indices = results[0]['val_indices'] + results[1]['val_indices']
        self.assertEqual(sorted(train_indices), [0, 1, 2, 3])
        self.assertEqual(sorted(val_indices), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
