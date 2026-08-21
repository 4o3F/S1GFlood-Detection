# Design

## Goal

Provide a single-temporal, VV-only water-segmentation baseline while preserving every public contract of the existing bi-temporal DAM-Net implementation.

## Module boundaries

```text
water_seg.dataset  -> prepared A/B/WATER_GT records and D4 transforms
water_seg.model    -> single-channel adapter, Swin-T encoder, U-Net decoder
water_seg.engine   -> shared train/eval loop, metrics, optimizer, checkpoints
water_seg.train    -> training CLI
water_seg.eval     -> patch-evaluation CLI
```

The package reuses the existing dataset validation functions, while keeping its runtime metrics, seeding, optimizer, and checkpoint code isolated from the DAM-Net training entry point. It does not modify or instantiate `DAMNet_New`.

## Data flow

1. `utils.dataloaders._build_split_dataset` validates the existing split layout and optional paired water masks.
2. `flatten_water_records` drops change-only records and emits A and B as independent samples.
3. `SingleTemporalWaterDataset` verifies replicated-VV RGB input, extracts one channel, applies a synchronized D4 transform during training, and returns `(image, mask, name)`.
4. `SwinTinyUNet` receives `[B,1,H,W]` raw VV intensity, scales to `[0,1]`, replicates to three channels, and applies ImageNet normalization.
5. The local hierarchical Swin-T produces `/4`, `/8`, `/16`, and `/32` features with channels `96`, `192`, `384`, and `768`.
6. The U-Net decoder fuses all four scales through channels `512`, `256`, `128`, and `64`, then returns two-class full-resolution logits.

## Encoder initialization

The legacy `swin_transformer.swin` constructor unconditionally loads `./PRETRAINED`. `SwinTinyEncoder` subclasses it and overrides only `init_weights`, preserving the encoder implementation while preventing that broken path from affecting this standalone model.

When ImageNet initialization is enabled, a classification-form Swin-T from the repository's pinned `timm==0.6.13` supplies matching encoder tensors. The final classification normalization is mapped to the local deepest-stage `norm3`; classification-head and fixed attention-mask-only keys are ignored. The runtime encoder remains the local size-flexible implementation, so 256-pixel project patches do not inherit timm's fixed classification forward.

## Optimization

The optimizer has two explicit parameter groups:

- encoder: `5e-5` by default;
- decoder and segmentation head: `5e-4` by default.

Both use AdamW with weight decay `0.01`; cosine annealing runs for the configured epoch budget. Cross-entropy is applied to complete-water targets. Validation water IoU controls the best checkpoint and early stopping.

## Checkpoint contract

Water checkpoints use format version `1` and contain:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
best_water_iou
train_metrics
val_metrics
config
```

This avoids the arbitrary-code and compatibility issues of full-module pickle checkpoints used by the legacy DAM-Net path.

## Compatibility invariants

- `DAMNet_New.forward(x1, x2)` remains unchanged.
- Existing auxiliary `WaterSegmentationHead` behavior remains unchanged.
- Existing change loaders, transforms, train/eval entry points, and SAFE inference remain unchanged.
- `255` remains a positive water value for `WATER_GT_*`; GEOID's `255=ignore` convention is not imported.
- Missing complete-water labels never become synthetic background supervision.
- The external model contract remains one VV tensor channel even though the encoder internally receives replicated normalized values for ImageNet compatibility.

## Known limits

- Current project PNG radiometry is not the same as GEOID's VV/VH dB normalization.
- The repository example root contains no `WATER_GT_A/B` and cannot train this task.
- No whole-scene or geospatial output path is included.
- GEOID benchmark values are contextual references, not expected scores on current project data.
