# Single-Temporal VV+VH Water Segmentation

The target workflow remains Kulsary-only training/evaluation. An optional
GEOID-Flood stage provides source-domain pretraining before Kulsary fine-tuning.
The two datasets keep separate loaders and checkpoints so their label, split,
and normalization provenance cannot be mixed accidentally.

## Optional GEOID-Flood pretraining

Point the pretraining command directly at the directory that contains
`data_tiles_s256_st128.csv`, `tile_catalog.parquet`, and the `EMSR*` folders:

```shell
uv run python -m water_seg.pretrain_geoid \
  --geoid-root /data/lhx/datasets/GEOID/data/geoid-flood \
  --validate-only

uv run python -m water_seg.compute_geoid_stats \
  --geoid-root /data/lhx/datasets/GEOID/data/geoid-flood

CUDA_VISIBLE_DEVICES=0 uv run python -m water_seg.pretrain_geoid \
  --geoid-root /data/lhx/datasets/GEOID/data/geoid-flood \
  --batch-size 8 \
  --num-workers 4 \
  --save-dir .tmp/geoid_swin_tiny_unet_vv_vh_unclipped
```

For one machine with two GPUs, launch one process per visible GPU:

```shell
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m water_seg.pretrain_geoid \
  --geoid-root /data/lhx/datasets/GEOID/data/geoid-flood \
  --batch-size 8 \
  --num-workers 2 \
  --save-dir .tmp/geoid_swin_tiny_unet_vv_vh_unclipped
```

In DDP mode, `--batch-size` and `--num-workers` are per process/GPU. The command
above therefore uses global batch size 16 and four loader workers in total.
Learning rates are not scaled automatically. Train metrics and validation
confusion counts are reduced across both ranks; validation shards contain no
padding duplicates. Only rank 0 displays progress, writes TensorBoard events,
and saves `best.pth`/`last.pth`. Checkpoints retain ordinary unwrapped model
keys and are interchangeable with single-GPU runs.

`compute_geoid_stats` is a one-time offline pass. It displays source-level
progress, computes exact train-window-weighted VV and VH mean/std values, and
atomically replaces `water_seg/geoid_stats.py` with Python constants plus the
metadata fingerprint, radiometry contract, validity threshold, and train sample
count. Pretraining never scans the imagery for statistics; it reads those
constants and rejects them if their provenance does not match the current CSV
selection. Regenerate the constants after changing the dataset metadata,
radiometry implementation, or `--min-valid-proportion`.

This is not compatible with the Kulsary DataLoader. The GEOID adapter reads
each official CSV row as a window into:

```text
geoid-flood/<EMSR-event-AoI>/s1grd/<tile_id>.tif
geoid-flood/<EMSR-event-AoI>/label/<label_id>.tif
```

Only S1-GRD bands 1 and 2 in `[VV,VH]` order, `train`/`val` rows, and
`pre`/`post` images are used. The adapter does not read `s2l2a`, `s1rtc`,
`dem`, `cloudmask`, or
`tile_catalog.parquet`, and it does not create another tiled dataset. A partial
download is therefore valid if the CSV plus every referenced `s1grd` and
`label` file is present. No GEOID redownload or data conversion is required;
the existing `s1grd` GeoTIFFs already provide the two source bands.

GEOID labels use `0=background`, `1=permanent water`, `2=flood`, and
`255=ignore`. For a single-temporal complete-water target, class 2 maps to
background on `pre` images and to water on `post` images; 255 and pixels invalid
in either polarization are excluded from both cross-entropy and metrics. GEOID
uses `10*log10(max(linear Sigma0, float32 eps))` without fixed-range clipping,
matching the official loader's radiometry step, then applies exact per-channel
mean/std computed from valid GEOID training pixels. Invalid pixels are filled
with those means before model normalization, so their normalized input is zero
while their labels remain ignored. This source-domain contract is intentionally
separate from Kulsary's clipped-dB contract. The official 256-pixel windows are
retained instead of resized to 224.

Pretraining and fine-tuning remain epoch-based and validate after every epoch.
Their train/validation loops display batch progress and current loss by default;
use `--no-progress` only for non-interactive logging.

Both commands write `last.pth` after every completed validation epoch. Resume
restores model, optimizer, scheduler, best metric, early-stopping counter, and
per-rank random-number state, then starts at the next epoch. Repeat the original
data and training options (including the original `--epochs` value):

```shell
CUDA_VISIBLE_DEVICES=0 uv run python -m water_seg.pretrain_geoid \
  --geoid-root /data/lhx/datasets/GEOID/data/geoid-flood \
  --batch-size 8 \
  --num-workers 4 \
  --save-dir .tmp/geoid_swin_tiny_unet_vv_vh_unclipped \
  --resume .tmp/geoid_swin_tiny_unet_vv_vh_unclipped/last.pth

CUDA_VISIBLE_DEVICES=0 uv run python -m water_seg.train \
  --sigma0-root /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0_vv_vh \
  --mask-source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks \
  --batch-size 8 \
  --num-workers 0 \
  --save-dir .tmp/water_swin_tiny_unet_vv_vh \
  --resume .tmp/water_swin_tiny_unet_vv_vh/last.pth
```

Resume is intentionally epoch-boundary only: work in an interrupted partial
epoch is repeated. Changing the dataset, world size, batch size, scheduler,
augmentation, split, or epoch budget is rejected instead of silently creating
a non-equivalent run. Old single-VV checkpoints and format-2 clipped-dB GEOID
checkpoints cannot initialize or resume the format-3 unclipped model.

Fine-tune the resulting model on Kulsary:

```shell
CUDA_VISIBLE_DEVICES=0 uv run python -m water_seg.train \
  --sigma0-root /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0_vv_vh \
  --mask-source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks \
  --init-checkpoint .tmp/geoid_swin_tiny_unet_vv_vh_unclipped/best.pth
```

The same target fine-tuning can use both GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m water_seg.train \
  --sigma0-root /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0_vv_vh \
  --mask-source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks \
  --init-checkpoint .tmp/geoid_swin_tiny_unet_vv_vh_unclipped/best.pth \
  --batch-size 4 \
  --num-workers 2
```

`--init-checkpoint` restores model weights only. Kulsary train-split per-channel
normalization is then applied, and optimizer/scheduler state starts fresh.

The Kulsary workflow has two explicit stages:

1. restored dual-pol Sentinel-1 GRD SAFE → stable two-band linear Sigma0
   GeoTIFF dataset in `[VV,VH]` order;
2. Sigma0 dataset + original PNG/PGW water masks → Swin-T U-Net training/evaluation.

SNAP runs only in stage 1. The training DataLoader never reads SAFE products or invokes SNAP.

## Stage 1: prepare the Sigma0 dataset

The restored root may contain top-level standard SAFEs and duplicate `products/` links into `*_COG` wrapper directories. Discovery collapses copies by product identifier and prefers restore-managed `products/` targets. For the current Kulsary scene the selected identifiers are `_5249.SAFE`, `_75FD.SAFE`, and `_779A.SAFE`.

```shell
uv run python prepare_kulsary_sigma0.py \
  --safe-root /home/ubuntu/lhx/Sentinel1-SAR/restored_grd \
  --output /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0_vv_vh \
  --gpt /usr/local/esa-snap/bin/gpt
```

The script uses the existing content-addressed SNAP cache. Cache hits are reused;
a missing date is preprocessed once in the parent process. VV-only cache entries
and the old one-band output directory remain untouched, while the first dual-pol
run creates distinct cache entries. Defaults match the Kulsary pair converter:
Precise orbit, Copernicus 30 m DEM, `EPSG:32639`, and 10 m pixels.

Output:

```text
kulsary_sigma0_vv_vh/
  before_sigma0_vv_vh.tif
  peak_sigma0_vv_vh.tif
  after_sigma0_vv_vh.tif
  sigma0_manifest.json
```

The GeoTIFFs are hardlinked from immutable cache generations when possible and copied across filesystems. Publication is atomic. Use `--dry-run` to inspect product bindings/cache status and `--refresh-snap-cache` to rebuild cache entries deliberately.

## Masks

The mask root is used only in stage 2:

```text
kulsary_masks/
  1_water_before_20240402.png
  1_water_before_20240402.pgw
  2_water_during_20240414.png
  2_water_during_20240414.pgw
  3_water_after_20240426.png
  3_water_after_20240426.pgw
  _preview_3panel.png
```

The three formal PNG+PGW pairs are discovered uniquely; files beginning with `_`, including the preview, are ignored. Masks are interpreted as EPSG:4326.

## Stage 2: training

```shell
uv run python -m water_seg.train \
  --sigma0-root /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0_vv_vh \
  --mask-source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks
```

The Sigma0 root manifest and sampled fingerprints are verified before loading. Advanced/test use may supply all three files explicitly instead:

```shell
uv run python -m water_seg.train \
  --sigma0-before /path/to/before_sigma0_vv_vh.tif \
  --sigma0-peak /path/to/peak_sigma0_vv_vh.tif \
  --sigma0-after /path/to/after_sigma0_vv_vh.tif \
  --mask-source /path/to/kulsary_masks
```

`--sigma0-root` and explicit paths are mutually exclusive.

## Spatial and temporal sampling

Peak Sigma0 defines the common grid. Before/after are read through bilinear
`WarpedVRT`; masks are nearest-neighbor reprojected. Only non-overlapping
256×256 windows with complete mask coverage and finite positive VV and VH in
all dates are retained.

Each retained tile contributes exactly:

```text
before
peak
after
```

The old prepared pair path exposed peak twice; direct loading produces `3N` unique-date samples. Tiles use deterministic spatial super-block splitting (default 2×2 blocks, 80/10/10, split seed 42).

## Radiometry and model

Linear Sigma0 is converted in memory:

```text
10 * log10(Sigma0) → clip [-25,0] dB
```

No uint8 quantization or RGB ImageNet normalization is applied. Mean and
population standard deviation are computed independently for VV and VH from the
train split only. Swin-T uses a two-channel stem initialized from adapted
ImageNet weights and a four-scale U-Net decoder `[512,256,128,64]`.

Important defaults:

| Setting | Default |
|---|---:|
| Epochs | 20 |
| Batch size | 8 |
| DataLoader workers | 0 |
| Encoder LR | `5e-5` |
| Decoder LR | `5e-4` |
| Weight decay | `0.01` |
| dB range | `[-25,0]` |
| Split | `0.8 / 0.1 / 0.1` |
| Split seed | 42 |
| Augmentation | uniform D4 |
| Early-stop patience | 5 |
| Checkpoint metric | water-class IoU |

Higher worker counts use multiprocessing `spawn` and per-process lazy raster reopening.

## Evaluation

When checkpoint paths still exist:

```shell
uv run python -m water_seg.eval \
  --path .tmp/water_swin_tiny_unet_vv_vh/best.pth
```

To relocate the same Sigma0 dataset:

```shell
uv run python -m water_seg.eval \
  --sigma0-root /new/path/kulsary_sigma0_vv_vh \
  --mask-source /new/path/kulsary_masks \
  --path .tmp/water_swin_tiny_unet_vv_vh/best.pth
```

Evaluation verifies file fingerprints, common grid, exact tile identities, and
split membership. Kulsary checkpoint format 3 stores the dual-channel contract,
per-channel clipped-dB normalization, and resumable state. Legacy single-VV
formats 1 and 2 are rejected and require preprocessing and retraining.

## Relationship to DAM-Net

`prepare_kulsary_pairs.py` remains the bi-temporal DAM-Net converter and still emits both pair variants plus prepared `A/B/GT/WATER_GT_*` trees. `water_seg` does not consume those trees.
