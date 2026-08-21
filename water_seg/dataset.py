import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils.dataloaders import _build_split_dataset, _open_binary_mask, _open_rgb


_D4_OPS = (
    None,
    Image.ROTATE_90,
    Image.ROTATE_180,
    Image.ROTATE_270,
    Image.FLIP_LEFT_RIGHT,
    Image.FLIP_TOP_BOTTOM,
    Image.TRANSPOSE,
    Image.TRANSVERSE,
)


def flatten_water_records(full_load):
    if isinstance(full_load, dict):
        source = (full_load[key] for key in sorted(full_load))
    else:
        source = full_load

    records = []
    for record in source:
        water_labels = record.get('water_labels')
        if not water_labels:
            continue
        directory, name = record['image']
        water_a, water_b = water_labels
        records.append({
            'image': os.path.join(directory, 'A', name),
            'mask': water_a,
            'name': f'{name}#A',
        })
        records.append({
            'image': os.path.join(directory, 'B', name),
            'mask': water_b,
            'name': f'{name}#B',
        })

    if not records:
        raise FileNotFoundError(
            'water dataset requires WATER_GT_A and WATER_GT_B labels; none remain'
        )
    return records


def _as_single_channel_vv(image, path):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'expected RGB image for single-channel VV: {path}')
    if not (
        np.array_equal(array[..., 0], array[..., 1])
        and np.array_equal(array[..., 0], array[..., 2])
    ):
        raise ValueError(
            f'RGB channels must be identical for single-channel VV: {path}'
        )
    return Image.fromarray(array[..., 0], mode='L')


def _to_image_tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)


def _to_mask_tensor(mask):
    return torch.from_numpy(np.asarray(np.asarray(mask) > 0, dtype=np.int64))


class RandomD4(object):
    def __call__(self, image, mask):
        op = _D4_OPS[random.randrange(8)]
        if op is None:
            return image, mask
        return image.transpose(op), mask.transpose(op)


class SingleTemporalWaterDataset(data.Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
        self._d4 = RandomD4() if augment else None

    def __getitem__(self, index):
        record = self.records[index]
        path = record['image']
        name = record['name']
        image = _as_single_channel_vv(_open_rgb(path), path)
        mask = _open_binary_mask(record['mask'])
        if image.size != mask.size:
            raise ValueError(
                f'image/mask size mismatch for {name}: '
                f'{image.size} vs {mask.size}'
            )
        if self.augment and image.width != image.height:
            raise ValueError(
                f'D4 augmentation requires a square VV patch for {name}: '
                f'{image.size}'
            )
        if self.augment:
            image, mask = self._d4(image, mask)
        return _to_image_tensor(image), _to_mask_tensor(mask), name

    def __len__(self):
        return len(self.records)


def build_water_dataset(data_dir, split, augment=False):
    records = flatten_water_records(_build_split_dataset(data_dir, split))
    return SingleTemporalWaterDataset(records, augment=augment)


def _water_loader(dataset, batch_size, num_workers, shuffle):
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def get_water_loaders(data_dir, batch_size, num_workers, augmentation=True):
    train_dataset = build_water_dataset(
        data_dir,
        'train',
        augment=augmentation,
    )
    val_dataset = build_water_dataset(data_dir, 'val', augment=False)
    return (
        _water_loader(train_dataset, batch_size, num_workers, shuffle=True),
        _water_loader(val_dataset, batch_size, num_workers, shuffle=False),
    )


def get_water_test_loader(data_dir, batch_size, num_workers):
    test_dataset = build_water_dataset(data_dir, 'test', augment=False)
    return _water_loader(test_dataset, batch_size, num_workers, shuffle=False)
