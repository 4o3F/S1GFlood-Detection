from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from utils.kulsary_raster import (
    PATCH_SIZE,
    POLARIZATIONS,
    CommonGrid,
    LazySigma0Stack,
    Sigma0Stack,
    linear_sigma0_to_clipped_db,
    plan_valid_tiles,
    sampled_file_fingerprint,
    tile_slice,
    tile_window,
    warp_masks,
)
from utils.kulsary_temporal import (
    OUTPUT_SPLITS,
    ROLE_DATES,
    AssignedRoleSample,
    TileKey,
    assign_spatial_blocks,
    discover_mask_refs,
    expand_role_samples,
)


_ROLE_ORDER = tuple(ROLE_DATES)
_D4_IDENTITY = 0
_D4_ROT90 = 1
_D4_ROT180 = 2
_D4_ROT270 = 3
_D4_FLIP_LR = 4
_D4_FLIP_UD = 5
_D4_TRANSPOSE = 6
_D4_TRANSVERSE = 7
SIGMA0_ROOT_FILENAMES = {
    'before': 'before_sigma0_vv_vh.tif',
    'peak': 'peak_sigma0_vv_vh.tif',
    'after': 'after_sigma0_vv_vh.tif',
}
SIGMA0_MANIFEST_FILENAME = 'sigma0_manifest.json'


def _require_path(path, *, kind, role=None):
    resolved = Path(path).expanduser().resolve()
    label = kind if role is None else f'{kind} for {role}'
    if not resolved.is_file():
        raise FileNotFoundError(f'{label} is missing: {resolved}')
    return resolved


def _require_directory(path, *, kind):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f'{kind} is missing: {resolved}')
    return resolved


def resolve_sigma0_paths(
    *,
    sigma0_root=None,
    sigma0_before=None,
    sigma0_peak=None,
    sigma0_after=None,
):
    explicit = {
        'before': sigma0_before,
        'peak': sigma0_peak,
        'after': sigma0_after,
    }
    has_explicit = [value is not None for value in explicit.values()]
    if sigma0_root is not None and any(has_explicit):
        raise ValueError(
            '--sigma0-root cannot be combined with explicit Sigma0 paths'
        )
    if sigma0_root is None:
        if not all(has_explicit):
            raise ValueError(
                'provide --sigma0-root or all of --sigma0-before, '
                '--sigma0-peak, and --sigma0-after'
            )
        paths = {
            role: _require_path(path, kind='Sigma0 raster', role=role)
            for role, path in explicit.items()
        }
        return paths, {
            'input_mode': 'explicit',
            'sigma0_root': None,
            'sigma0_manifest': None,
            'sigma0_manifest_version': None,
            'polarizations': list(POLARIZATIONS),
        }

    root = _require_directory(sigma0_root, kind='Sigma0 dataset directory')
    manifest_path = root / SIGMA0_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Sigma0 manifest is missing: {manifest_path}')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'could not read Sigma0 manifest: {manifest_path}') from exc
    if not isinstance(manifest, dict) or manifest.get('format') != 'kulsary-sigma0':
        raise ValueError(f'invalid Kulsary Sigma0 manifest: {manifest_path}')
    if manifest.get('polarizations') != list(POLARIZATIONS):
        raise ValueError(
            f'Sigma0 manifest must declare polarizations {POLARIZATIONS}'
        )
    expected_band_order = {
        polarization: index
        for index, polarization in enumerate(POLARIZATIONS, start=1)
    }
    if manifest.get('band_order') != expected_band_order:
        raise ValueError(
            f'Sigma0 manifest band order must be {expected_band_order}'
        )
    roles = manifest.get('roles')
    if not isinstance(roles, dict) or set(roles) != set(SIGMA0_ROOT_FILENAMES):
        raise ValueError('Sigma0 manifest must contain before, peak, and after roles')

    paths = {}
    for role, filename in SIGMA0_ROOT_FILENAMES.items():
        record = roles[role]
        if not isinstance(record, dict) or record.get('output_filename') != filename:
            raise ValueError(f'Sigma0 manifest filename mismatch for {role}')
        path = _require_path(root / filename, kind='Sigma0 raster', role=role)
        if sampled_file_fingerprint(path) != record.get('fingerprint'):
            raise ValueError(f'Sigma0 manifest fingerprint mismatch for {role}')
        paths[role] = path
    return paths, {
        'input_mode': 'sigma0-root',
        'sigma0_root': str(root),
        'sigma0_manifest': str(manifest_path.resolve()),
        'sigma0_manifest_version': manifest.get('version'),
        'polarizations': list(POLARIZATIONS),
    }


def _validate_db_range(db_min, db_max):
    if not math.isfinite(db_min) or not math.isfinite(db_max):
        raise ValueError('db_min and db_max must be finite')
    if db_min >= db_max:
        raise ValueError('db_min must be smaller than db_max')


def _apply_d4(array, op):
    spatial_axes = (-2, -1)
    if op == _D4_IDENTITY:
        transformed = array
    elif op == _D4_ROT90:
        transformed = np.rot90(array, 1, axes=spatial_axes)
    elif op == _D4_ROT180:
        transformed = np.rot90(array, 2, axes=spatial_axes)
    elif op == _D4_ROT270:
        transformed = np.rot90(array, 3, axes=spatial_axes)
    elif op == _D4_FLIP_LR:
        transformed = np.flip(array, axis=-1)
    elif op == _D4_FLIP_UD:
        transformed = np.flip(array, axis=-2)
    elif op == _D4_TRANSPOSE:
        transformed = np.swapaxes(array, -2, -1)
    elif op == _D4_TRANSVERSE:
        transformed = np.flip(
            np.rot90(array, 1, axes=spatial_axes),
            axis=-1,
        )
    else:
        raise ValueError(f'unsupported D4 op: {op}')
    return np.ascontiguousarray(transformed)


def _welford_combine(count, mean, second_moment, values):
    """Combine a tile into a running Welford/Chan population accumulator."""
    batch = np.asarray(values, dtype=np.float64).reshape(-1)
    batch_count = int(batch.size)
    if batch_count == 0:
        return count, mean, second_moment
    batch_mean = float(batch.mean())
    batch_second = float(np.square(batch - batch_mean).sum())
    if count == 0:
        return batch_count, batch_mean, batch_second
    delta = batch_mean - mean
    total = count + batch_count
    combined_mean = mean + delta * batch_count / total
    combined_second = (
        second_moment
        + batch_second
        + (delta * delta) * count * batch_count / total
    )
    return total, combined_mean, combined_second


@dataclass
class KulsarySceneIndex:
    sigma0_paths: dict[str, Path]
    mask_source: Path
    mask_files: dict[str, Path]
    grid: CommonGrid
    water_masks: dict[str, np.ndarray]
    kept_tiles: list[TileKey]
    split_by_tile: dict[TileKey, str]
    samples: list[AssignedRoleSample]
    skips: list[dict]
    db_min: float
    db_max: float
    block_tiles: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_seed: int

    def samples_for(self, split):
        if split not in OUTPUT_SPLITS:
            raise ValueError(f'unsupported output split: {split}')
        return [sample for sample in self.samples if sample.output_split == split]

    def counts(self):
        split_counts = {split: 0 for split in OUTPUT_SPLITS}
        for sample in self.samples:
            split_counts[sample.output_split] += 1
        split_counts['tiles'] = len(self.kept_tiles)
        split_counts['samples'] = len(self.samples)
        return split_counts


def grid_signature(grid):
    return {
        'crs': str(grid.crs),
        'transform': [
            float(grid.transform.a),
            float(grid.transform.b),
            float(grid.transform.c),
            float(grid.transform.d),
            float(grid.transform.e),
            float(grid.transform.f),
        ],
        'width': int(grid.width),
        'height': int(grid.height),
        'peak_window': [
            float(grid.peak_window.col_off),
            float(grid.peak_window.row_off),
            float(grid.peak_window.width),
            float(grid.peak_window.height),
        ],
    }


def tile_split_records(index):
    return [
        {
            'row': tile.row,
            'col': tile.col,
            'split': index.split_by_tile[tile],
        }
        for tile in sorted(index.kept_tiles)
    ]


def source_fingerprints(index):
    return {
        'sigma0': {
            role: sampled_file_fingerprint(index.sigma0_paths[role])
            for role in _ROLE_ORDER
        },
        'masks': {
            role: sampled_file_fingerprint(index.mask_files[role])
            for role in _ROLE_ORDER
        },
    }


def build_kulsary_scene_index(
    sigma0_before,
    sigma0_peak,
    sigma0_after,
    mask_source,
    *,
    db_min=-25.0,
    db_max=0.0,
    block_tiles=2,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=42,
):
    """Index fully valid 256 tiles from linear Sigma0 GeoTIFFs and PNG/PGW masks."""
    _validate_db_range(db_min, db_max)
    sigma0_paths = {
        'before': _require_path(sigma0_before, kind='Sigma0 raster', role='before'),
        'peak': _require_path(sigma0_peak, kind='Sigma0 raster', role='peak'),
        'after': _require_path(sigma0_after, kind='Sigma0 raster', role='after'),
    }
    mask_root = _require_directory(mask_source, kind='mask source directory')
    mask_refs = discover_mask_refs(mask_root)

    stack = Sigma0Stack(sigma0_paths, mask_ref=mask_refs['peak'])
    try:
        water_masks, coverage = warp_masks(mask_refs, stack.grid)
        kept_tiles, skips = plan_valid_tiles(stack, coverage)
        grid = stack.grid
    finally:
        stack.close()

    if not kept_tiles:
        raise ValueError('no fully valid 256x256 Kulsary tiles were found')

    split_by_tile = assign_spatial_blocks(
        kept_tiles,
        block_tiles=block_tiles,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    samples = expand_role_samples(kept_tiles, split_by_tile)
    _assert_role_cardinality(kept_tiles, samples)

    return KulsarySceneIndex(
        sigma0_paths=sigma0_paths,
        mask_source=mask_root,
        mask_files={role: mask_refs[role].png_path for role in _ROLE_ORDER},
        grid=grid,
        water_masks=water_masks,
        kept_tiles=list(kept_tiles),
        split_by_tile=dict(split_by_tile),
        samples=list(samples),
        skips=list(skips),
        db_min=float(db_min),
        db_max=float(db_max),
        block_tiles=int(block_tiles),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        split_seed=int(split_seed),
    )


def _assert_role_cardinality(kept_tiles, samples):
    kept = list(kept_tiles)
    if len(samples) != 3 * len(kept):
        raise ValueError(
            'expected exactly 3 role samples per kept tile, '
            f'got {len(samples)} samples for {len(kept)} tiles'
        )
    roles_by_tile = {}
    for sample in samples:
        roles_by_tile.setdefault(sample.tile, Counter())[sample.role] += 1
    if set(roles_by_tile) != set(kept):
        raise ValueError('role samples do not cover the kept tiles exactly once')
    expected = {role: 1 for role in _ROLE_ORDER}
    for tile, role_counts in roles_by_tile.items():
        if dict(role_counts) != expected:
            raise ValueError(
                f'tile {tile} must have exactly one before, peak, and after sample, '
                f'found {dict(role_counts)}'
            )


def compute_train_channel_stats(index):
    """Per-channel population statistics over Kulsary train samples only.

    Each train role-tile is read through ``LazySigma0Stack.read_role`` and
    converted with ``linear_sigma0_to_clipped_db``. Accumulators use a
    streaming Welford/Chan update so the three scenes are never loaded at
    once. The returned standard deviation is the population value
    ``sqrt(M2 / n)``.
    """
    samples = index.samples_for('train')
    if not samples:
        raise ValueError('no training samples are available for channel stats')

    counts = np.zeros(len(POLARIZATIONS), dtype=np.int64)
    means = np.zeros(len(POLARIZATIONS), dtype=np.float64)
    second_moments = np.zeros(len(POLARIZATIONS), dtype=np.float64)
    stack = LazySigma0Stack(index.sigma0_paths, index.grid)
    try:
        for sample in samples:
            sigma0, valid = stack.read_role(sample.role, tile_window(sample.tile))
            clipped_db = linear_sigma0_to_clipped_db(
                sigma0,
                valid,
                index.db_min,
                index.db_max,
            )
            for channel in range(len(POLARIZATIONS)):
                (
                    counts[channel],
                    means[channel],
                    second_moments[channel],
                ) = _welford_combine(
                    int(counts[channel]),
                    float(means[channel]),
                    float(second_moments[channel]),
                    clipped_db[channel],
                )
    finally:
        stack.close()

    if bool((counts <= 0).any()):
        raise ValueError('training channel stats received no pixels')
    stds = np.sqrt(second_moments / counts)
    if not bool(np.isfinite(means).all()):
        raise ValueError('training channel means must be finite')
    if not bool(np.isfinite(stds).all()) or bool((stds <= 0.0).any()):
        raise ValueError('training channel stds must be finite and positive')
    return means.tolist(), stds.tolist()


class RandomD4(object):
    def __call__(self, image, mask):
        image_array = np.asarray(image)
        mask_array = np.asarray(mask)
        if image_array.ndim not in {2, 3} or mask_array.ndim != 2:
            raise ValueError(
                'D4 augmentation requires a 2-D/CHW image and 2-D mask'
            )
        if image_array.shape[-2:] != mask_array.shape:
            raise ValueError(
                'image/mask shape mismatch for D4: '
                f'{image_array.shape} vs {mask_array.shape}'
            )
        if image_array.shape[-2] != image_array.shape[-1]:
            raise ValueError(
                'D4 augmentation requires square spatial dimensions: '
                f'{image_array.shape}'
            )
        op = random.randrange(8)
        return _apply_d4(image_array, op), _apply_d4(mask_array, op)


class KulsaryRawWaterDataset(data.Dataset):
    def __init__(self, index, split, augment=False):
        if split not in OUTPUT_SPLITS:
            raise ValueError(f'unsupported output split: {split}')
        self.index = index
        self.split = split
        self.augment = bool(augment)
        self.samples = list(index.samples_for(split))
        self._d4 = RandomD4() if self.augment else None
        self._stack = LazySigma0Stack(index.sigma0_paths, index.grid)

    def __getitem__(self, sample_index):
        sample = self.samples[sample_index]
        sigma0, valid = self._stack.read_role(sample.role, tile_window(sample.tile))
        expected_image_shape = (len(POLARIZATIONS), PATCH_SIZE, PATCH_SIZE)
        if sigma0.shape != expected_image_shape:
            raise ValueError(
                f'expected {POLARIZATIONS} Sigma0 tile with shape '
                f'{expected_image_shape} for {sample.name}, '
                f'got {sigma0.shape}'
            )
        if valid.shape != (PATCH_SIZE, PATCH_SIZE) or not bool(valid.all()):
            raise ValueError(f'Sigma0 tile is not fully valid for {sample.name}')
        image = linear_sigma0_to_clipped_db(
            sigma0,
            valid,
            self.index.db_min,
            self.index.db_max,
        )
        mask = np.asarray(
            self.index.water_masks[sample.role][tile_slice(sample.tile)],
            dtype=bool,
        )
        if mask.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(
                f'expected {PATCH_SIZE}x{PATCH_SIZE} water mask for {sample.name}, '
                f'got {mask.shape}'
            )
        if self.augment:
            image, mask = self._d4(image, mask)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image, dtype=np.float32)
        )
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        return image_tensor, mask_tensor, sample.name

    def __len__(self):
        return len(self.samples)

    def close(self):
        stack = getattr(self, '_stack', None)
        if stack is not None:
            stack.close()

    def __getstate__(self):
        self.close()
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, '_stack', None) is None:
            self._stack = LazySigma0Stack(self.index.sigma0_paths, self.index.grid)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def water_worker_init_fn(worker_id):
    """Close inherited raster handles and reseed per DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, 'close'):
        info.dataset.close()
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(int(seed))


class DistributedEvalSampler(data.Sampler):
    """Shard evaluation without DistributedSampler's duplicate padding."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError('num_replicas must be positive')
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError('rank must be within num_replicas')

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


def _water_loader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    distributed_context=None,
    sampler_seed=0,
):
    sampler = None
    if (
        distributed_context is not None
        and distributed_context.distributed
    ):
        if shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=distributed_context.world_size,
                rank=distributed_context.rank,
                shuffle=True,
                seed=int(sampler_seed),
                drop_last=False,
            )
        else:
            sampler = DistributedEvalSampler(
                dataset,
                num_replicas=distributed_context.world_size,
                rank=distributed_context.rank,
            )
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle if sampler is None else False,
        'sampler': sampler,
        'num_workers': num_workers,
        'drop_last': False,
        'pin_memory': torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs['worker_init_fn'] = water_worker_init_fn
        kwargs['multiprocessing_context'] = 'spawn'
    return data.DataLoader(dataset, **kwargs)


def get_water_loaders(
    index,
    batch_size,
    num_workers,
    augmentation=True,
    distributed_context=None,
    sampler_seed=0,
):
    train_dataset = KulsaryRawWaterDataset(
        index,
        'train',
        augment=augmentation,
    )
    val_dataset = KulsaryRawWaterDataset(index, 'val', augment=False)
    return (
        _water_loader(
            train_dataset,
            batch_size,
            num_workers,
            shuffle=True,
            distributed_context=distributed_context,
            sampler_seed=sampler_seed,
        ),
        _water_loader(
            val_dataset,
            batch_size,
            num_workers,
            shuffle=False,
            distributed_context=distributed_context,
            sampler_seed=sampler_seed,
        ),
    )


def get_water_test_loader(index, batch_size, num_workers):
    test_dataset = KulsaryRawWaterDataset(index, 'test', augment=False)
    return _water_loader(test_dataset, batch_size, num_workers, shuffle=False)
