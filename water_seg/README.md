# Kulsary Single-Temporal VV Water Segmentation

The Kulsary workflow has two explicit stages:

1. restored Sentinel-1 GRD SAFE → stable linear `Sigma0_VV` GeoTIFF dataset;
2. Sigma0 dataset + original PNG/PGW water masks → Swin-T U-Net training/evaluation.

SNAP runs only in stage 1. The training DataLoader never reads SAFE products or invokes SNAP.

## Stage 1: prepare the Sigma0 dataset

The restored root may contain top-level standard SAFEs and duplicate `products/` links into `*_COG` wrapper directories. Discovery collapses copies by product identifier and prefers restore-managed `products/` targets. For the current Kulsary scene the selected identifiers are `_5249.SAFE`, `_75FD.SAFE`, and `_779A.SAFE`.

```shell
uv run python prepare_kulsary_sigma0.py \
  --safe-root /home/ubuntu/lhx/Sentinel1-SAR/restored_grd \
  --output /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0 \
  --gpt /usr/local/esa-snap/bin/gpt
```

The script uses the existing content-addressed SNAP cache. Cache hits are reused; a missing date is preprocessed once in the parent process. Defaults match the Kulsary pair converter: Precise orbit, Copernicus 30 m DEM, `EPSG:32639`, and 10 m pixels.

Output:

```text
kulsary_sigma0/
  before_sigma0_vv.tif
  peak_sigma0_vv.tif
  after_sigma0_vv.tif
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
  --sigma0-root /home/ubuntu/lhx/Sentinel1-SAR/kulsary_sigma0 \
  --mask-source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks
```

The Sigma0 root manifest and sampled fingerprints are verified before loading. Advanced/test use may supply all three files explicitly instead:

```shell
uv run python -m water_seg.train \
  --sigma0-before /path/to/before_sigma0_vv.tif \
  --sigma0-peak /path/to/peak_sigma0_vv.tif \
  --sigma0-after /path/to/after_sigma0_vv.tif \
  --mask-source /path/to/kulsary_masks
```

`--sigma0-root` and explicit paths are mutually exclusive.

## Spatial and temporal sampling

Peak Sigma0 defines the common grid. Before/after are read through bilinear `WarpedVRT`; masks are nearest-neighbor reprojected. Only non-overlapping 256×256 windows with complete mask coverage and finite positive Sigma0 in all dates are retained.

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

No uint8 quantization or RGB ImageNet normalization is applied. Mean and population standard deviation are computed only from the train split. Swin-T uses a one-channel stem initialized from adapted ImageNet weights and a four-scale U-Net decoder `[512,256,128,64]`.

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
  --path .tmp/water_swin_tiny_unet/best.pth
```

To relocate the same Sigma0 dataset:

```shell
uv run python -m water_seg.eval \
  --sigma0-root /new/path/kulsary_sigma0 \
  --mask-source /new/path/kulsary_masks \
  --path .tmp/water_swin_tiny_unet/best.pth
```

Evaluation verifies file fingerprints, common grid, exact tile identities, and split membership. Checkpoint format 2 stores one-channel clipped-dB normalization; format 1 checkpoints are rejected and require retraining.

## Relationship to DAM-Net

`prepare_kulsary_pairs.py` remains the bi-temporal DAM-Net converter and still emits both pair variants plus prepared `A/B/GT/WATER_GT_*` trees. `water_seg` does not consume those trees.
