# Single-Temporal VV Water Segmentation

This package provides a standalone binary water-segmentation baseline for the prepared datasets used by this repository. It does not call or modify the bi-temporal DAM-Net change-detection path.

## Reference baseline

The architecture and training defaults follow the single-image SAR benchmark in [GEOID-Flood](https://github.com/links-ads/geoid-flood) at commit `b0ab63540a2a331513be306a5cbdc4ba88c766f5`:

- Swin-T encoder with four feature scales;
- U-Net decoder channels `[512, 256, 128, 64]`;
- cross-entropy loss, D4 augmentation, AdamW, cosine scheduling, and validation water-IoU selection;
- ImageNet encoder initialization by default.

GEOID-Flood reports `F1_bin=0.929`, `IoU_bin=0.873`, and post-hoc `IoU_flood=0.469` for its Swin-T baseline. Those values are reference targets only: GEOID uses VV+VH dB rasters, different labels, event splits, and `255` as ignore. This implementation uses the current project's VV-only PNGs and treats `255` in `WATER_GT_*` as water, so its metrics are not directly comparable.

Reference files:

- [GEOID Swin-T configuration](https://github.com/links-ads/geoid-flood/blob/b0ab63540a2a331513be306a5cbdc4ba88c766f5/configs/backbone_benchmark/swin_tiny.yaml)
- [GEOID dataset implementation](https://github.com/links-ads/geoid-flood/blob/b0ab63540a2a331513be306a5cbdc4ba88c766f5/src/geoid_flood/datasets/geoid.py)
- [GEOID paper](https://arxiv.org/abs/2608.02315)

## Dataset contract

The root must use the existing prepared layout:

```text
<dataset>/
  train/
    A/ B/ GT/
    WATER_GT_A/ WATER_GT_B/
  val/
    A/ B/ GT/
    WATER_GT_A/ WATER_GT_B/
  test/
    A/ B/ GT/
    WATER_GT_A/ WATER_GT_B/
```

Each labeled temporal pair is flattened into two independent samples:

```text
(A/<name>, WATER_GT_A/<name>)
(B/<name>, WATER_GT_B/<name>)
```

Change-only records without `WATER_GT_A/B` are omitted rather than interpreted as dry pixels. A split with no complete-water labels fails explicitly. Plain S1GFloods `GT` is a flood-change mask and is not a valid target for this model; use prepared ETCI/Kulsary data or a merged root containing `WATER_GT_A/B`.

The source PNG must contain the same VV grayscale values in all three RGB channels. The loader verifies this and returns one raw `float32` channel in `[0,255]`. Training patches must be square when D4 augmentation is enabled (the prepared project patches are `256×256`). The model internally replicates that VV channel, scales it, and applies ImageNet normalization before Swin-T.

Masks may contain `{0,1}` or `{0,255}`. Both `1` and `255` mean water; there is no ignore class in this project-specific path.

## Training

```shell
uv run python -m water_seg.train \
  --dataset-dir /path/to/prepared-or-merged-dataset
```

Important defaults:

| Setting | Default |
|---|---:|
| Epochs | 20 |
| Batch size | 8 |
| Encoder LR | `5e-5` |
| Decoder LR | `5e-4` |
| Weight decay | `0.01` |
| Scheduler | cosine, `eta_min=1e-6` |
| Augmentation | uniform D4 |
| Early-stop patience | 5 validation checks |
| Checkpoint metric | water IoU |
| Save directory | `.tmp/water_swin_tiny_unet` |

ImageNet Swin-T weights are obtained through the pinned `timm==0.6.13`. For an offline or random-initialization run:

```shell
uv run python -m water_seg.train \
  --dataset-dir /path/to/dataset \
  --no-imagenet-pretrained
```

The trainer writes `best.pth`, `last.pth`, and TensorBoard logs. Checkpoints are state-dictionary bundles rather than pickled model objects.

## Evaluation

```shell
uv run python -m water_seg.eval \
  --dataset-dir /path/to/dataset \
  --path .tmp/water_swin_tiny_unet/best.pth
```

Evaluation uses both independently flattened dates from the test split and reports global loss, precision, recall, F1, overall accuracy, and water IoU.

## Scope

This package supports patch training and patch evaluation only. Single-SAFE preprocessing, sliding-window inference, probability mosaics, and GeoTIFF output are intentionally outside the first implementation.
