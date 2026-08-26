# Design

## Goal

Optimize single-temporal water segmentation for the Kulsary target domain through a reproducible target-data boundary, with optional source-domain pretraining on GEOID-Flood. Restored Kulsary GRD SAFE products are first published as a stable three-file Sigma0 dataset, then target training reads only that dataset and the original full-water masks. This eliminates prepared-pair peak duplication, uint8 quantization, and RGB ImageNet normalization without changing the legacy DAM-Net pipeline.

## Boundaries

```text
utils.kulsary_products    -> duplicate-safe restored SAFE discovery
prepare_kulsary_sigma0.py -> parent-only SNAP/cache and atomic Sigma0 publish
utils.kulsary_raster      -> common grid, raster windows, mask warp, tile validity
utils.kulsary_temporal    -> dates, spatial split, unique role samples
water_seg.dataset         -> Sigma0-root validation, scene index, stats, loaders
water_seg.geoid_dataset   -> official GEOID CSV windows and label remapping
water_seg.compute_geoid_stats -> one-time VV statistics constant generator
water_seg.geoid_stats     -> generated GEOID normalization and provenance
water_seg.model           -> one-channel Swin-T encoder and U-Net decoder
water_seg.engine          -> metrics, optimizer, checkpoint format 2
water_seg.pretrain_geoid  -> GEOID source-domain pretraining CLI
water_seg.train/eval      -> Kulsary Sigma0 training/evaluation CLIs
```

Training accepts a canonical precomputed Sigma0 root or three explicit GeoTIFFs. SAFE discovery and SNAP execution are confined to the standalone preprocessing script; DataLoader workers cannot reach them.

GEOID pretraining is a separate chain. It accepts the official
`data_tiles_s256_st128.csv`, lazily reads S1-GRD VV windows and labels, and emits
a transfer checkpoint. Kulsary training may restore its model tensors through
`--init-checkpoint`, then overwrites VV normalization with Kulsary train-split
statistics and creates a fresh optimizer/scheduler. Joint source/target batches
are intentionally not supported.

## Single-node distributed training

Both training entry points infer DDP from the environment populated by
`torchrun`. Each rank binds to `cuda:LOCAL_RANK`; the CLI batch size and worker
count are per rank. Training uses `DistributedSampler` with a shared seed and
epoch update. Validation uses an unpadded rank-strided sampler, then loss and
the global 2x2 confusion matrix are all-reduced before metric computation.

The DDP wrapper synchronizes gradients, while model buffers are explicitly
broadcast from rank 0 before unwrapped validation. Rank 0 alone emits progress,
TensorBoard data, console summaries, and atomic checkpoints. Payloads serialize
the unwrapped model so no `module.` prefix enters either GEOID transfer or
Kulsary format-2 checkpoints. A non-`torchrun` invocation remains the original
single-process path.

Before the first pretraining run, `compute_geoid_stats` scans the selected
training coverage once. It groups overlapping windows by source raster, derives
their exact coverage weights, and writes mean/std plus selection provenance to
`geoid_stats.py`. The pretraining entry point performs no statistics scan and
refuses constants whose metadata fingerprint, dB range, validity threshold, or
train sample count differs from the current index.

## GEOID label and raster contract

GEOID raw classes are background 0, permanent water 1, flood 2, and ignore
255. The complete-water remap is `{0:0,1:1,2:0}` for pre-event imagery and
`{0:0,1:1,2:1}` for post-event imagery. Ignore-label and invalid/nonpositive VV
pixels remain ignored in loss and confusion metrics. The index keeps only
official train/val S1-GRD rows with at least 1% valid label coverage.

Only band 1 VV is read. Each metadata row contributes its `(x,y,256,256)`
window; overlapping train windows and non-overlapping validation windows remain
as published. No prepared GEOID tile cache is written.

## Stage-one SAFE publication

`discover_kulsary_grd_products` scans top-level and `products/` SAFE entries, collapses duplicate copies by product identifier, and prefers restore-managed `products/` targets. `prepare_kulsary_sigma0.py` reuses or builds content-addressed SNAP cache entries in the parent process, then atomically publishes canonical before/peak/after GeoTIFFs and a fingerprinted manifest. Hardlinks preserve immutable cache bytes without duplication; copy is the cross-filesystem fallback.

## Spatial data flow

1. Discover the three PNG+PGW masks and validate identical size/affine.
2. Open the three one-band Sigma0 GeoTIFFs. Peak defines the common grid; before/after use bilinear `WarpedVRT`.
3. Clip the grid to the three raster extents and mask extent.
4. Reproject full-water masks from EPSG:4326 using nearest-neighbor.
5. Enumerate non-overlapping 256×256 windows, dropping partial edges, incomplete mask coverage, or any tile with invalid/nonpositive Sigma0 in any date.
6. Assign valid tiles to deterministic 2×2 spatial super-block train/val/test splits.
7. Expand each tile to exactly `before`, `peak`, and `after` samples. The pair converter's second peak occurrence is not used.

`KulsarySceneIndex` owns paths, immutable grid metadata, warped masks, tiles, splits, role samples, and provenance. It does not retain open GDAL handles.

## Worker-safe raster access

`LazySigma0Stack` stores only paths and `CommonGrid`. It opens rasterio datasets and VRTs on first access, records the process PID, drops handles during pickle, and reopens when the PID changes. Multi-worker DataLoaders use the `spawn` context, so workers receive a handle-free serialized dataset; worker initialization resets the lazy stack and reseeds Python/NumPy from `torch.initial_seed()`.

No worker performs grid planning, mask reprojection, or SNAP processing.

## Radiometry

Dataset samples remain float32 clipped dB:

```text
linear Sigma0 → 10*log10 → clip [db_min, db_max]
```

Train-split role tiles alone provide streaming population mean/std. Validation and test pixels never contribute to normalization statistics.

The local Swin-T patch embedding is one channel. ImageNet initialization adapts the timm RGB patch weights with `adapt_input_conv(1, weight)`, maps the final timm normalization to local `norm3`, and copies the remaining matching tensors. Forward normalization is `(VV_dB - vv_mean) / vv_std`.

## Model

The encoder produces four scales:

```text
/4  : 96 channels
/8  : 192 channels
/16 : 384 channels
/32 : 768 channels
```

The U-Net decoder uses channels `[512,256,128,64]` and returns two-class full-resolution logits. DAM-Net and its auxiliary water head are not instantiated.

## Checkpoint format 2

The checkpoint stores model/optimizer/scheduler state, metrics, and required provenance:

```text
input = single VV channel, clipped dB
normalization = train-split clipped-dB mean/std
vv_mean / vv_std
db_min / db_max
three Sigma0 source paths and mask source
split seed, block size, split ratios
common-grid signature
exact kept tile identities and split membership
sampled content fingerprints for all Sigma0 and mask files
samples per split
```

Evaluation rebuilds the index with checkpoint paths or explicit relocation paths, applies checkpoint split/dB parameters, and verifies the common grid plus every kept tile's split membership. Format 1 checkpoints are rejected because their three-channel quantized input stem is incompatible.

## Compatibility invariants

- `prepare_kulsary_pairs.py` still emits both change pair variants and all prepared directories/manifests.
- DAM-Net `train.py`, `eval.py`, loaders, ETCI conversion, and merge tooling remain unchanged.
- Water targets are the complete per-date masks; flood-change `GT` is never used.
- No hidden tile cache or prepared-water directory is written.
- `before`, `peak`, and `after` are each sampled once per spatial tile.

## Known limits

- Real training requires the three precomputed SNAP Sigma0 GeoTIFFs; masks alone are insufficient.
- The full warped mask arrays reside in memory.
- Spatial split quality is constrained by one Kulsary event and should be evaluated with additional split seeds when reporting final performance.
- Current metrics report water-class IoU, not two-class mean IoU.
- GEOID VV constants must be regenerated explicitly when its metadata or
  radiometry selection changes.
