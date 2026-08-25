import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.utils.data as data

from utils.kulsary_raster import PATCH_SIZE
from water_seg.dataset import (_validate_db_range, _water_loader,
                               _welford_combine, RandomD4)


GEOID_METADATA_FILENAME = 'data_tiles_s256_st128.csv'
GEOID_IGNORE_INDEX = 255
GEOID_PRETRAINING_KIND = 'geoid-water-pretraining'
GEOID_PRETRAINING_FORMAT_VERSION = 1
_GEOID_SPLITS = ('train', 'val')
_GEOID_IMAGE_TIMES = ('pre', 'post')
_REQUIRED_COLUMNS = {
    'tile_id',
    'modality',
    'x',
    'y',
    'size',
    'image_time',
    'positive_proportion',
    'valid_proportion',
    'label_id',
    'split',
}


def _parse_int(row, field, line_number):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f'GEOID metadata line {line_number} has invalid {field}'
        ) from exc


def _parse_float(row, field, line_number):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f'GEOID metadata line {line_number} has invalid {field}'
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f'GEOID metadata line {line_number} has non-finite {field}'
        )
    return value


def _safe_component(value, field, line_number):
    component = str(value).strip()
    if (
        not component
        or component in {'.', '..'}
        or Path(component).name != component
        or '/' in component
        or '\\' in component
    ):
        raise ValueError(
            f'GEOID metadata line {line_number} has unsafe {field}'
        )
    return component


def _event_folder_from_label_id(label_id):
    chip_prefix = label_id.removesuffix('_label')
    if '-' not in chip_prefix:
        raise ValueError(f'cannot derive GEOID event folder from {label_id!r}')
    return chip_prefix.rsplit('-', 1)[0]


@dataclass(frozen=True)
class GEOIDWaterSample:
    tile_id: str
    label_id: str
    event_folder: str
    image_time: str
    split: str
    x: int
    y: int
    size: int
    image_path: Path
    label_path: Path
    positive_proportion: float
    valid_proportion: float

    @property
    def name(self):
        return (
            f'{self.tile_id}__{self.image_time}'
            f'__x{self.x}_y{self.y}_s{self.size}'
        )


@dataclass
class GEOIDWaterIndex:
    root: Path
    metadata_path: Path
    samples: list[GEOIDWaterSample]
    db_min: float
    db_max: float
    min_valid_proportion: float

    def samples_for(self, split):
        if split not in _GEOID_SPLITS:
            raise ValueError(f'unsupported GEOID split: {split}')
        return [sample for sample in self.samples if sample.split == split]

    def counts(self):
        counts = {
            split: len(self.samples_for(split))
            for split in _GEOID_SPLITS
        }
        counts['samples'] = len(self.samples)
        counts['pre'] = sum(
            sample.image_time == 'pre' for sample in self.samples
        )
        counts['post'] = sum(
            sample.image_time == 'post' for sample in self.samples
        )
        return counts


def build_geoid_water_index(
    root,
    *,
    metadata_filename=GEOID_METADATA_FILENAME,
    db_min=-25.0,
    db_max=0.0,
    min_valid_proportion=0.01,
):
    """Build the official CSV-windowed, single-temporal S1-GRD VV index."""
    _validate_db_range(db_min, db_max)
    if not math.isfinite(min_valid_proportion) or not (
        0.0 <= min_valid_proportion <= 1.0
    ):
        raise ValueError('min_valid_proportion must be between 0 and 1')

    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'GEOID-Flood root is missing: {data_root}')
    metadata_name = _safe_component(metadata_filename, 'metadata filename', 0)
    metadata_path = data_root / metadata_name
    if not metadata_path.is_file():
        raise FileNotFoundError(f'GEOID metadata is missing: {metadata_path}')

    samples = []
    seen = set()
    with metadata_path.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f'GEOID metadata is missing required columns: {missing}'
            )
        for line_number, row in enumerate(reader, start=2):
            if str(row['modality']).strip().lower() != 's1grd':
                continue
            split = str(row['split']).strip().lower()
            if split not in _GEOID_SPLITS:
                continue
            image_time = str(row['image_time']).strip().lower()
            if image_time not in _GEOID_IMAGE_TIMES:
                raise ValueError(
                    f'GEOID metadata line {line_number} has invalid image_time'
                )
            valid_proportion = _parse_float(
                row,
                'valid_proportion',
                line_number,
            )
            if valid_proportion < min_valid_proportion:
                continue
            positive_proportion = _parse_float(
                row,
                'positive_proportion',
                line_number,
            )
            if not 0.0 <= valid_proportion <= 1.0:
                raise ValueError(
                    f'GEOID metadata line {line_number} has invalid '
                    'valid_proportion'
                )
            if not 0.0 <= positive_proportion <= 1.0:
                raise ValueError(
                    f'GEOID metadata line {line_number} has invalid '
                    'positive_proportion'
                )

            x = _parse_int(row, 'x', line_number)
            y = _parse_int(row, 'y', line_number)
            size = _parse_int(row, 'size', line_number)
            if x < 0 or y < 0 or size != PATCH_SIZE:
                raise ValueError(
                    f'GEOID metadata line {line_number} must describe a '
                    f'non-negative {PATCH_SIZE}x{PATCH_SIZE} window'
                )
            tile_id = _safe_component(row['tile_id'], 'tile_id', line_number)
            label_id = _safe_component(row['label_id'], 'label_id', line_number)
            event_folder = _safe_component(
                _event_folder_from_label_id(label_id),
                'event folder',
                line_number,
            )
            image_path = data_root / event_folder / 's1grd' / f'{tile_id}.tif'
            label_path = data_root / event_folder / 'label' / f'{label_id}.tif'
            key = (tile_id, label_id, image_time, split, x, y, size)
            if key in seen:
                raise ValueError(
                    f'duplicate GEOID sample at metadata line {line_number}: {key}'
                )
            seen.add(key)
            samples.append(GEOIDWaterSample(
                tile_id=tile_id,
                label_id=label_id,
                event_folder=event_folder,
                image_time=image_time,
                split=split,
                x=x,
                y=y,
                size=size,
                image_path=image_path,
                label_path=label_path,
                positive_proportion=positive_proportion,
                valid_proportion=valid_proportion,
            ))

    index = GEOIDWaterIndex(
        root=data_root,
        metadata_path=metadata_path,
        samples=samples,
        db_min=float(db_min),
        db_max=float(db_max),
        min_valid_proportion=float(min_valid_proportion),
    )
    for split in _GEOID_SPLITS:
        if not index.samples_for(split):
            raise ValueError(f'GEOID metadata has no selected {split} samples')
    return index


def validate_geoid_files(index):
    """Check every unique CSV-referenced S1-GRD and label path cheaply."""
    image_paths = {sample.image_path for sample in index.samples}
    label_paths = {sample.label_path for sample in index.samples}
    missing_images = sorted(path for path in image_paths if not path.is_file())
    missing_labels = sorted(path for path in label_paths if not path.is_file())
    if missing_images or missing_labels:
        examples = [
            str(path)
            for path in (missing_images[:3] + missing_labels[:3])
        ]
        raise FileNotFoundError(
            'GEOID partial download is incomplete for selected S1-GRD rows: '
            f'{len(missing_images)} images and {len(missing_labels)} labels '
            f'are missing; examples={examples}'
        )
    return {
        's1grd_files': len(image_paths),
        'label_files': len(label_paths),
    }


def _read_geoid_arrays(sample, db_min, db_max):
    if not sample.image_path.is_file():
        raise FileNotFoundError(f'GEOID S1-GRD image is missing: {sample.image_path}')
    if not sample.label_path.is_file():
        raise FileNotFoundError(f'GEOID label is missing: {sample.label_path}')

    window = Window(sample.x, sample.y, sample.size, sample.size)
    with rasterio.open(sample.image_path) as image_source:
        if image_source.count < 1:
            raise ValueError(f'GEOID image has no VV band: {sample.image_path}')
        if (
            sample.x + sample.size > image_source.width
            or sample.y + sample.size > image_source.height
        ):
            raise ValueError(f'GEOID image window is out of bounds: {sample.name}')
        sigma0 = image_source.read(1, window=window, out_dtype='float32')
    with rasterio.open(sample.label_path) as label_source:
        if label_source.count < 1:
            raise ValueError(f'GEOID label has no band: {sample.label_path}')
        if (
            sample.x + sample.size > label_source.width
            or sample.y + sample.size > label_source.height
        ):
            raise ValueError(f'GEOID label window is out of bounds: {sample.name}')
        raw_mask = label_source.read(1, window=window)

    expected_shape = (sample.size, sample.size)
    if sigma0.shape != expected_shape or raw_mask.shape != expected_shape:
        raise ValueError(f'GEOID window has an unexpected shape: {sample.name}')
    raw_values = set(np.unique(raw_mask).tolist())
    unexpected = sorted(raw_values - {0, 1, 2, GEOID_IGNORE_INDEX})
    if unexpected:
        raise ValueError(
            f'GEOID label has unsupported values {unexpected}: {sample.label_path}'
        )

    image_valid = np.isfinite(sigma0) & (sigma0 > 0) & (sigma0 <= 1e3)
    image = np.full(expected_shape, np.float32(db_min), dtype=np.float32)
    if bool(image_valid.any()):
        image[image_valid] = np.clip(
            np.float32(10.0) * np.log10(sigma0[image_valid]),
            np.float32(db_min),
            np.float32(db_max),
        )

    mask = np.full(expected_shape, GEOID_IGNORE_INDEX, dtype=np.int64)
    mask[raw_mask == 0] = 0
    mask[raw_mask == 1] = 1
    if sample.image_time == 'pre':
        mask[raw_mask == 2] = 0
    else:
        mask[raw_mask == 2] = 1
    mask[~image_valid] = GEOID_IGNORE_INDEX
    return image, mask


def compute_geoid_train_vv_stats(index):
    """Population VV dB statistics over valid train pixels only."""
    samples = index.samples_for('train')
    if not samples:
        raise ValueError('no GEOID training samples are available for VV stats')
    count = 0
    mean = 0.0
    second_moment = 0.0
    for sample in samples:
        image, mask = _read_geoid_arrays(
            sample,
            index.db_min,
            index.db_max,
        )
        count, mean, second_moment = _welford_combine(
            count,
            mean,
            second_moment,
            image[mask != GEOID_IGNORE_INDEX],
        )
    if count <= 0:
        raise ValueError('GEOID training VV stats received no valid pixels')
    std = math.sqrt(second_moment / count)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
        raise ValueError('GEOID training VV std must be finite and positive')
    return float(mean), float(std)


class GEOIDRawWaterDataset(data.Dataset):
    def __init__(self, index, split, augment=False):
        if split not in _GEOID_SPLITS:
            raise ValueError(f'unsupported GEOID split: {split}')
        self.index = index
        self.split = split
        self.samples = list(index.samples_for(split))
        self._d4 = RandomD4() if augment else None

    def __getitem__(self, sample_index):
        sample = self.samples[sample_index]
        image, mask = _read_geoid_arrays(
            sample,
            self.index.db_min,
            self.index.db_max,
        )
        if self._d4 is not None:
            image, mask = self._d4(image, mask)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask, dtype=np.int64)
        )
        return image_tensor, mask_tensor, sample.name

    def __len__(self):
        return len(self.samples)


def get_geoid_water_loaders(
    index,
    batch_size,
    num_workers,
    augmentation=True,
):
    train_dataset = GEOIDRawWaterDataset(
        index,
        'train',
        augment=augmentation,
    )
    val_dataset = GEOIDRawWaterDataset(index, 'val', augment=False)
    return (
        _water_loader(train_dataset, batch_size, num_workers, shuffle=True),
        _water_loader(val_dataset, batch_size, num_workers, shuffle=False),
    )
