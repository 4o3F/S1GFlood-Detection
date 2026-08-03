import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils import transforms as trans


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
_REQUIRED_SUBDIRECTORIES = ('A', 'B', 'GT')
_WATER_SUBDIRECTORIES = ('WATER_GT_A', 'WATER_GT_B')
_ALLOWED_MASK_VALUES = {0, 1, 255}


def _list_images(directory):
    return sorted(
        name for name in os.listdir(directory)
        if not name.startswith('.') and name.lower().endswith(_IMAGE_EXTENSIONS)
    )


def _image_names(directory):
    return set(_list_images(directory)) if os.path.isdir(directory) else set()


def _validate_required_names(split_root, sample_names):
    reference = set(sample_names)
    for subdirectory in _REQUIRED_SUBDIRECTORIES[1:]:
        directory = os.path.join(split_root, subdirectory)
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f'missing required dataset directory: {directory}'
            )
        names = _image_names(directory)
        missing = sorted(reference - names)
        extra = sorted(names - reference)
        if missing or extra:
            raise ValueError(
                f'{split_root}/{subdirectory} basename mismatch; '
                f'missing={missing[:1]}, extra={extra[:1]}'
            )


def _water_label_names(split_root, sample_names):
    water_a_dir = os.path.join(split_root, _WATER_SUBDIRECTORIES[0])
    water_b_dir = os.path.join(split_root, _WATER_SUBDIRECTORIES[1])
    water_a_exists = os.path.isdir(water_a_dir)
    water_b_exists = os.path.isdir(water_b_dir)

    if water_a_exists != water_b_exists:
        missing = water_b_dir if water_a_exists else water_a_dir
        raise FileNotFoundError(
            f'water-label directories must exist as a pair; missing: {missing}'
        )
    if not water_a_exists:
        return set()

    water_a_names = _image_names(water_a_dir)
    water_b_names = _image_names(water_b_dir)
    if water_a_names != water_b_names:
        only_a = sorted(water_a_names - water_b_names)
        only_b = sorted(water_b_names - water_a_names)
        raise FileNotFoundError(
            f'water labels must exist as A/B pairs under {split_root}; '
            f'only WATER_GT_A={only_a[:1]}, only WATER_GT_B={only_b[:1]}'
        )

    orphan_names = sorted(water_a_names - set(sample_names))
    if orphan_names:
        raise ValueError(
            f'orphan water label under {split_root}: {orphan_names[0]}'
        )
    return water_a_names


def _build_split_dataset(data_dir, split):
    split_root = os.path.join(data_dir, split)
    a_directory = os.path.join(split_root, 'A')
    if not os.path.isdir(a_directory):
        raise FileNotFoundError(f'missing required dataset directory: {a_directory}')

    sample_names = _list_images(a_directory)
    _validate_required_names(split_root, sample_names)
    water_names = _water_label_names(split_root, sample_names)

    dataset = {}
    for index, name in enumerate(sample_names):
        water_labels = None
        if name in water_names:
            water_labels = (
                os.path.join(split_root, 'WATER_GT_A', name),
                os.path.join(split_root, 'WATER_GT_B', name),
            )
        dataset[index] = {
            'image': [split_root + os.sep, name],
            'label': os.path.join(split_root, 'GT', name),
            'water_labels': water_labels,
        }
    return dataset


def train_path(data_dir):
    return (
        _build_split_dataset(data_dir, 'train'),
        _build_split_dataset(data_dir, 'val'),
    )


def test_path(data_dir):
    return _build_split_dataset(data_dir, 'test')


def _open_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def _open_binary_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 3:
        if array.shape[2] != 3 or not (
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 0], array[..., 2])
        ):
            raise ValueError(f'mask RGB channels must be identical: {path}')
        array = array[..., 0]
    elif array.ndim != 2:
        raise ValueError(f'unsupported mask shape {array.shape}: {path}')

    values = set(np.unique(array).tolist())
    if not values.issubset(_ALLOWED_MASK_VALUES):
        raise ValueError(
            f'mask must contain only {{0,1,255}}, found {sorted(values)}: {path}'
        )
    normalized = np.where(array > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(normalized, mode='L')


def _validate_sample_sizes(name, image_a, image_b, targets):
    expected = image_a.size
    sizes = {'A': image_a.size, 'B': image_b.size}
    sizes.update({target_name: mask.size for target_name, mask in targets.items()})
    if any(size != expected for size in sizes.values()):
        raise ValueError(f'image/label size mismatch for {name}: {sizes}')


def FloodChange(
    img_path,
    label_path,
    aug,
    water_label_paths=None,
    include_water=False,
    load_water_labels=True,
):
    directory, name = img_path
    image_a = _open_rgb(directory + 'A/' + name)
    image_b = _open_rgb(directory + 'B/' + name)
    targets = {'change': _open_binary_mask(label_path)}
    water_valid = (
        load_water_labels
        and water_label_paths is not None
    )

    if include_water and load_water_labels:
        if water_valid:
            targets['water_a'] = _open_binary_mask(water_label_paths[0])
            targets['water_b'] = _open_binary_mask(water_label_paths[1])
        else:
            targets['water_a'] = Image.new('L', targets['change'].size, color=0)
            targets['water_b'] = Image.new('L', targets['change'].size, color=0)

    _validate_sample_sizes(name, image_a, image_b, targets)
    sample = {'image': (image_a, image_b), 'targets': targets}
    sample = trans.train_transforms(sample) if aug else trans.test_transforms(sample)

    if not include_water:
        return (
            sample['image'][0],
            sample['image'][1],
            sample['targets']['change'],
            name,
        )

    transformed_targets = sample['targets']
    transformed_targets['water_valid'] = torch.tensor(
        water_valid,
        dtype=torch.bool,
    )
    return sample['image'][0], sample['image'][1], transformed_targets, name


class FloodDetection(data.Dataset):
    def __init__(
        self,
        full_load,
        flag='train',
        aug=False,
        include_water=False,
        load_water_labels=True,
    ):
        self.full_load = full_load
        self.loader = FloodChange
        self.aug = aug
        self.include_water = include_water
        self.load_water_labels = load_water_labels

    def __getitem__(self, index):
        record = self.full_load[index]
        return self.loader(
            record['image'],
            record['label'],
            self.aug,
            water_label_paths=record.get('water_labels'),
            include_water=self.include_water,
            load_water_labels=self.load_water_labels,
        )

    def __len__(self):
        return len(self.full_load)
