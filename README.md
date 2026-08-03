![demo](images/1.png)

<div align="center">
<h1 align="center">☀️S1GFloods Benchmark☀️</h1>

<h3 align="center">DAM-Net: Flood detection from SAR imagery using differential attention metric-based vision transformers</h3>


[Tamer Saleh](https://scholar.google.com/citations?user=KAmm5ZkAAAAJ&hl=en)<sup>1,2</sup>, 
[Xingxing Weng]()<sup>1</sup>, 
[Shimaa Holail](https://scholar.google.com/citations?user=WKKVqDgAAAAJ&hl=en)<sup>1</sup>, 
[Chen Hao]()<sup>1</sup>, 
[Gui Song-Xia](https://scholar.google.com/citations?user=SAUCVsEAAAAJ&hl=en)<sup>1</sup>


<sup>1</sup> Wuhan University, <sup>2</sup> Benha University

[![ISPRS paper](https://img.shields.io/badge/ISPRS-paper-cyan)](https://www.sciencedirect.com/science/article/abs/pii/S0924271624002168)  [![Google Drive Dataset](https://img.shields.io/badge/GoogleDrive-Dataset-blue)](https://drive.google.com/file/d/1bm_sFfJ05Fryj6Ib1niIidOywljzIEgo/view?usp=sharing)   [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tamer-Saleh/S1GFloods)  [![Baidu Drive Dataset](https://img.shields.io/badge/BaiduDrive-Dataset-green)](https://pan.baidu.com/s/1E4dEJtlQ6xeUDRPGO904KQ?pwd=m6gr)

</div>


## 🛎️Updates 
 <p align="center">
  <img src=./images/highlycited.png width="100%">
</p>

* **`🎉 February 2026 Achievement`**: S1GFloods has been selected as an 🔥 ESI Hot Paper and Highly Cited Paper, placing it among the top 1% of publications in the Geosciences field 🏆🌍
* **` 13 May 2024`**: DAM-Net has been accepted by [ISPRS JP&RS and online available](https://www.sciencedirect.com/science/article/abs/pii/S0924271624002168) now!!
* **` 02 July 2023`**: The S1GFloods benchmark related to our paper has now released. You are warmly welcome to use it!!
* **` 25 June 2023`**: DAM-Net has been submitted for publication at ISPRS Journal of Photogrammetry and Remote Sensing!!
* **` 01 Jun 2023`**: The [arXiv paper](https://arxiv.org/abs/2306.00704v1) of DAM-Net is now online.


## 🔭Dataset Overview

* [**S1GFloods**](https://www.sciencedirect.com/science/article/abs/pii/S0924271624002168) is the first open-access, globally distributed, event-diverse Sentinel-1 SAR dataset specifically designed to support AI-based flood response applications. The dataset comprises **5,360** image pairs with a spatial size of **256 × 256** pixels, covering **46** major flood events that occurred between **2015** and **2022** across **six** continents, with particular emphasis on developing countries. Its broad geographic and event diversity provides a comprehensive benchmark for developing and evaluating robust flood-mapping models.

* The accompanying animation illustrates pre- and post-event SAR imagery along with sample flood-mapping results for a rural region in Iran affected by severe flooding in **March 2019**. The figure on the right presents a magnified visualization of a **1 km × 1 km** area (highlighted by the yellow box in the larger scene), demonstrating the extent of flood impacts on buildings as identified by our model.

* **Dataset Statistics**

- **Training Set**: 4,300 image pairs  
- **Validation Set**: 530 image pairs  
- **Testing Set**: 530 image pairs 
- **Image Size**: 256 × 256 pixels
- **Spatial Resolution**: 10 meters  

* **Each image pair includes:**
- A manually annotated binary label map
- Pixel value `0`: Background / non-flooded area
- Pixel value `255`: Newly inundated flood area


 <p align="center">
  <img src="./images/test-map.gif" width="521.97" height="304.8" />
  <img src="./images/zoom-test-map.gif" width="304.8" height="304.8" />
</p>

 
 ![image1](./images/2.png)

 <div align="center">
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Bangladesh-Img.gif" width="144.78" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Iran-Img.gif" width="314.96" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Nigeria-Img.gif" width="197.485" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Nanchang-Img.gif" width="149.86" height="168.91" />
</div>

 <div align="center">
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Bangladesh-GT.gif" width="144.78" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Iran-GT.gif" width="314.96" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Nigeria-GT.gif" width="197.485" height="168.91" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Nanchang-GT.gif" width="149.86" height="168.91" />
</div>

 <div align="center">
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Wuhan-Img.gif" width="176.53" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Redrivernorth-Img.gif" width="188.595" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Sudan-Img.gif" width="219.075" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Florence-Img.gif" width="221.615" height="254" />
</div>

 <div align="center">
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Wuhan-GT.gif" width="176.53" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Redrivernorth-GT.gif" width="188.595" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Sudan-GT.gif" width="219.075" height="254" />
  <img src="https://github.com/Tamer-Saleh/GFlood-Detection/blob/Flood-Mapping/images/Florence-GT.gif" width="221.615" height="254" />
</div>


![image3](./images/4.png)


## Requirements

[![Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/) 
[![Pytorch 1.7.1](https://img.shields.io/badge/Pytorch-1.7.1-blue.svg)](https://pytorch.org/get-started/previous-versions/)
[![torchvision 0.8.2](https://img.shields.io/badge/torchvision-0.8.2-blue.svg)](https://pypi.org/project/torchvision/0.8.2/)
[![Opencv 4.5.5](https://img.shields.io/badge/Opencv-4.5.5-blue.svg)](https://opencv.org/opencv-4-5-5/)
[![CUDA Toolkit 10.1](https://img.shields.io/badge/CUDA-10.1-blue.svg)](https://developer.nvidia.com/cuda-10.1-download-archive-base)
[![Python-SNAPPY 8.0](https://img.shields.io/badge/PythonSNAPPY-8.0-blue.svg)](https://senbox.atlassian.net/wiki/spaces/SNAP/pages/50855941/Configure+Python+to+use+the+SNAP-Python+snappy+interface)
[![Wandb 0.13.10](https://img.shields.io/badge/Wandb-0.13.10-blue.svg)](https://pypi.org/project/wandb/)

### Reproducible environment with uv

The checked-in uv environment targets Linux x86_64, Python 3.10, and NVIDIA CUDA 12.4. PyTorch CUDA wheels are resolved from the official PyTorch wheel repository; all other packages use the TUNA PyPI mirror.

```shell
uv sync --locked
```

Verify the CUDA environment:

```shell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The repository does not include the optional `PRETRAINED/` backbone weights. When they are absent, the model starts with random initialization.

Whole-scene inference from downloaded Sentinel-1 GRD `.SAFE` products also requires the ESA SNAP command-line Graph Processing Tool (`gpt`). Install SNAP separately, add `gpt` to `PATH`, set `SNAP_GPT`, or pass `--gpt /path/to/gpt`. The Python environment uses Rasterio for windowed GeoTIFF access and final grid alignment; Python-SNAPPY is not required by `infer_safe.py`.


## Our model
An overview of the proposed DAM-Net. The feature maps of the pre-and post-event image pairs are extracted through a Siamese structure and pre-trained remote sensing. 
![Overall](./images/overall.png)


### 🔭 Baselines <a name="baselines"></a>

- :open_book:	:open_book:	 :open_book: DTCDSCN [[here](https://ieeexplore.ieee.org/abstract/document/9311793)]
- :open_book:	:open_book:	 :open_book: UNet [[here](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIV-4-W3-2020/215/2020/)]
- :open_book:	:open_book:	 :open_book: FC-Siam [[here](https://ieeexplore.ieee.org/abstract/document/8451652)]
- :open_book:	:open_book:	 :open_book: SNUNet–ECAM [[here](https://ieeexplore.ieee.org/abstract/document/9355573)]
- :open_book:	:open_book:	 :open_book: Siam-Nested-UNet [[here](https://dl.acm.org/doi/abs/10.1145/3437802.3437810)]
- :open_book:	:open_book:	 :open_book: ResNet50-IMP [[here](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
- :open_book:	:open_book:	 :open_book: ResNet50-RSP [[here](https://ieeexplore.ieee.org/abstract/document/9782149)]
- :open_book:	:open_book:	 :open_book: Swin–T-RSP [[here](https://ieeexplore.ieee.org/abstract/document/9782149)]
- :open_book:	:open_book:	 :open_book: Swin-T-IMP [[here](https://ieeexplore.ieee.org/abstract/document/9736956)]
- :open_book:	:open_book:	 :open_book: ViTAEv2 [[here](https://arxiv.org/pdf/2202.10108.pdf)]


## 📒 Dataset Preparation

  Prepare the following folders to organize this repo:

     For the S1GFloods dataset, clip the images to 256 × 256 patches. Please, respect the following structure: 
              ├── SIGFloods
              │   ├── train
              │   │   ├── A                           Images of Time 1 before the flood event
              │   │   │   └── <region><year><XY>.png
              │   │   ├── B                           Images of Time 2 after the flood event
              │   │   │   └── <region><year><XY>.png
              │   │   └── GT                          Ground truth labels
              │   │       └── <region><year><XY>.png
              │   ├── val 
              │   │   ├── A                           
              │   │   │   └── <region><year><XY>.png
              │   │   ├── B                           
              │   │   │   └── <region><year><XY>.png
              │   │   └── GT                          
              │   │       └── <region><year><XY>.png
              │   ├── test
              │   │   ├── A                           
              │   │   │   └── <region><year><XY>.png
              │   │   ├── B                           
              │   │   │   └── <region><year><XY>.png
              │   │   └── GT                          
              │   │       └── <region><year><XY>.png
              │  

Each split may optionally provide paired full-water masks for auxiliary
supervision:

```text
<split>/
├── A/<name>.png
├── B/<name>.png
├── GT/<name>.png
├── WATER_GT_A/<name>.png   # all water visible in A
└── WATER_GT_B/<name>.png   # all water visible in B
```

`WATER_GT_A` and `WATER_GT_B` are sparse but strictly paired: both directories
must exist together, and a supervised basename must exist in both. Samples
without either water mask remain valid change-only samples; the loader uses an
explicit validity flag and never interprets a missing mask as all-background.
Masks may use binary `{0, 1}` or `{0, 255}` values and must match the A/B image
size. S1GFloods can therefore be mixed with datasets that provide full-water
labels without fabricating auxiliary targets for S1GFloods.

If the downloaded dataset contains flat `A/`, `B/`, and `Label/` directories, prepare the required layout with:

```shell
uv run python prepare_dataset.py \
  --source /path/to/S1GFloods \
  --output /path/to/S1GFloods_prepared \
  --mode copy
```

The default deterministic split uses seed `42` and creates `4,300` training, `530` validation, and `530` test samples. Use `--dry-run` to validate file pairing before creating output. The generated split is reproducible but is not a substitute for an official benchmark split manifest.

For large datasets, `--mode hardlink` avoids duplicating file data when source and output are on the same filesystem. `--mode symlink` is also available.

### ETCI-2021 temporal pairs

The [ETCI-2021 Flood Detection](https://huggingface.co/datasets/blanchon/ETCI-2021-Flood-Detection) dataset ships per-date Sentinel-1 VV/VH tiles plus RGB flood masks. Several regions are captured on multiple dates, so a single offline converter, `prepare_etci_pairs.py`, reorganizes them into the same bi-temporal `train/val/test` `{A, B, GT}` layout as S1GFloods:

- **A** = earlier-date VV tile for a `(region, x, y)` coordinate
- **B** = later-date VV tile for the same coordinate
- **GT** = later-date `flood_label`, rewritten to single-channel mode-L `{0, 255}` (always re-encoded, never hardlinked from the RGB source)

```shell
uv run python prepare_etci_pairs.py \
  --download \
  --output /path/to/ETCI_2021_prepared \
  --pair-policy nearest-flood-free \
  --val-ratio 0.1 \
  --seed 42 \
  --mode hardlink
```

`--download` fetches the pinned Hugging Face revision into the cache; pass `--source /path/to/etci_repo` instead to convert an already-downloaded copy. Use `--dry-run` to inspect the planned pair counts and skip reasons before creating output.

If `huggingface.co` is unreachable or slow, route the download through a mirror with `--hf-endpoint https://hf-mirror.com` (also honored from the `HF_ENDPOINT` environment variable). The endpoint is passed explicitly to `snapshot_download`, so it does not depend on the variable being exported into the process.

Pairing and split policy:

- `--pair-policy nearest-flood-free` (default): for each later tile, pair it with the most recent earlier tile whose own flood label is clean (`0` flood pixels); `adjacent-any` pairs with the immediately preceding tile regardless of its label.
- Train pairs are group-split into `train`/`val` by `(region, x, y)` using a seeded SHA-256 hash, so all dates of one tile coordinate stay in the same split (leakage-free).
- Source `test` pairs (Florence) are written to output `test`; source `test_internal` tiles carry no `flood_label` and are excluded.
- Lightweight QC drops tiles that are too small (`--min-vv-bytes`), uniform, or saturated (`--max-saturated-fraction`); use `--no-keep-negative-post` to drop tiles whose post-event label has no flood.

Caveat: tile alignment is inferred from matching `(region, x, y)` keys and equal image shapes. The PNG mirror carries no Sentinel-1 product, orbit, or geotransform metadata, so coordinate equality does not prove strict geographic registration. The output is a drop-in root for `train.py`/`eval.py`:

```shell
uv run python train.py --dataset-dir /path/to/ETCI_2021_prepared
```

### Kulsary Orbit 159 temporal pairs

`prepare_kulsary_pairs.py` converts the three Kulsary 2024 Sentinel-1
acquisitions and their PNG+PGW water masks into the same
`{train,val,test}/{A,B,GT}` contract. It creates two semantic variants at each
valid tile coordinate:

| Variant | A (baseline) | B (flood peak) | GT | Chronology |
|---|---|---|---|---|
| `before_to_peak` | 2024-04-02 | 2024-04-14 | peak water excluding 04-02 water | chronological |
| `after_to_peak` | 2024-04-26 | 2024-04-14 | peak water excluding 04-26 water | deliberately reversed |

The converter requires three restored, standard GRD SAFE products under
`--safe-root` or its managed `products/` directory. Products are assigned to
roles from acquisition dates in their `manifest.safe`, so the restored filename
CRC is irrelevant. COG SAFE products are rejected; restoration remains a
separate workflow.

Upload the three mask PNGs and sibling PGW files to the server, then run the
static validation first:

```shell
uv run python prepare_kulsary_pairs.py \
  --source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks \
  --safe-root /home/ubuntu/lhx/Sentinel1-SAR/restored_grd \
  --output /home/ubuntu/lhx/Sentinel1-SAR/kulsary_prepared \
  --dry-run
```

The dry run validates mask geometry and SAFE metadata, prints source-grid GT
pixel counts and both pair definitions, and probes the existing SNAP cache. It
does not require `gpt` and does not create output, work, staging, or cache
directories.

Run the full conversion on the GPU server with the same SNAP parameters used
by `predict_safe_pair.sh`:

```shell
uv run python prepare_kulsary_pairs.py \
  --source /home/ubuntu/lhx/Sentinel1-SAR/kulsary_masks \
  --safe-root /home/ubuntu/lhx/Sentinel1-SAR/restored_grd \
  --output /home/ubuntu/lhx/Sentinel1-SAR/kulsary_prepared \
  --work-dir "${HOME}/scratch/damnet-safe" \
  --snap-cache-dir "${HOME}/scratch/damnet-safe/snap-cache" \
  --gpt /usr/local/esa-snap/bin/gpt \
  --target-crs EPSG:32639 \
  --pixel-spacing 10
```

The peak-date Sigma0 raster defines the common grid. Before/after Sigma0 is
warped bilinearly, masks are warped nearest-neighbor from their WGS84 PGW
geometry, and linear Sigma0 is mapped through `[-25,0] dB` before being saved
as RGB PNG. Tiles are non-overlapping 256×256 windows. Incomplete edges and
windows containing an invalid pixel in any acquisition are dropped for both
variants, while all-background GT tiles are retained.

Unique 2×2-tile spatial super-blocks are deterministically assigned
80/10/10 to train, validation, and test. Both variants at one coordinate
always share a split. The output includes split, pair, QC, and skipped-record
manifests and can be passed directly to `train.py` or merged with another
prepared root:

```shell
uv run python merge_datasets.py \
  --input /path/to/S1GFloods_prepared \
  --input /home/ubuntu/lhx/Sentinel1-SAR/kulsary_prepared \
  --output /path/to/merged \
  --mode hardlink
```

### Joint training on multiple datasets

`train.py` trains on a single prepared root. To mix several datasets (e.g. S1GFloods and the ETCI-2021 temporal pairs above), merge their prepared roots into one with `merge_datasets.py`, then point training at the merged root. The training loader shuffles, so samples from every source are interleaved each epoch.

```shell
uv run python merge_datasets.py \
  --input /path/to/S1GFloods_prepared \
  --input /path/to/ETCI_2021_prepared \
  --output /path/to/merged \
  --mode hardlink
```

Each `--input` must be a prepared root with `{train,val,test}/{A,B,GT}`. Optional paired `WATER_GT_A/WATER_GT_B` masks are propagated sparsely, and the merge manifest records supervised counts per split and source; no placeholder label files are created. `--mode hardlink` combines sources without duplicating file data (falls back to copy across filesystems). Cross-source filename collisions are rejected by default; ETCI pairs are prefixed `etci_`, so S1GFloods and ETCI never collide. Use `--on-collision rename` (with a unique `--tag NAME` per input) to re-namespace all sample files, including water masks, when sources share names. Rename mode uses an unambiguous length-prefixed namespace such as `2__s1__<name>`. Use `--dry-run` to preview sample and paired-water counts first. The merged root is a drop-in `--dataset-dir`:

```shell
uv run python train.py --dataset-dir /path/to/merged
```

              
## :truck: Datasets <a name="dataset"></a>

You can download our novel public S1GFloods dataset through the following link:

- [x] [S1GFloods][baidu drive](https://pan.baidu.com/s/1E4dEJtlQ6xeUDRPGO904KQ?pwd=m6gr)
- [x] [S1GFloods][Google Drive Link](https://drive.google.com/file/d/1bm_sFfJ05Fryj6Ib1niIidOywljzIEgo/view?usp=sharing)
- [x] [S1GFloods][HuggingFace Link](https://huggingface.co/datasets/Tamer-Saleh/S1GFloods)


## 📚 Use example

- Training

  ```shell
  bash train_s1gfloods.sh /path/to/S1GFloods
  ```

  Training runs for at most `1,000` epochs and validates every `10` completed epochs. Checkpoints are saved only when validation F1 improves by more than `0.001`. Training stops after `3` consecutive validation checks without significant improvement.

  ```shell
  bash train_s1gfloods.sh /path/to/S1GFloods_prepared \
    --epochs 1000 \
    --validation-interval 10 \
    --early-stopping-patience 3 \
    --min-f1-improvement 0.001
  ```

  Checkpoint epoch numbers are one-based, for example `checkpoint_epoch_10.pth`.

  When paired `WATER_GT_A/WATER_GT_B` masks are available, one shared lightweight
  segmentation head predicts all water independently for A and B. The head
  branches directly from each single-temporal backbone feature map and bypasses
  tokenization, Transformer, CTCA, TACE, TDF, and other multi-temporal fusion.
  The main change loss still uses every sample, while the auxiliary loss uses
  only samples whose paired water masks are present:

  ```text
  total_loss = change_loss + water_loss_weight * water_loss
  ```

  The default auxiliary weight is `0.2`; set it to `0` to disable auxiliary
  computation and recover main-task-only training. Best-checkpoint selection and
  early stopping remain controlled exclusively by validation flood-change F1,
  not by the auxiliary water metrics.

  The dataset directory can also be passed directly to the Python entry point:

  ```shell
  uv run python train.py \
    --dataset-dir /path/to/merged \
    --water-loss-weight 0.2
  ```

- Testing

  ```shell
  uv run python eval.py \
    --dataset-dir /path/to/S1GFloods_prepared \
    --path .tmp/S1GFloods_vitae_rsp/checkpoint_epoch_<N>.pth
  ```

- Whole-scene inference from Sentinel-1 GRD SAFE products

  Pass an earlier pre-event product followed by a later post-event product. Both products must contain VV polarization and use the same IW acquisition geometry, orbit direction, and relative orbit. The inference entry point applies SNAP orbit correction, GRD border-noise removal, thermal-noise removal, Sigma0 calibration, and terrain correction before sliding-window inference.

  Products ending in `_COG.SAFE` contain ZSTD-compressed Cloud Optimized GeoTIFF measurements. Direct SNAP support remains SNAP/GDAL-version dependent. If SNAP fails while reading COG metadata, including `noiseVectorListElem is null` in `Remove-GRD-Border-Noise`, restore the products to standard GRD SAFE before prediction.

  Build the official CDSE utilities image once:

  ```shell
  docker build "https://github.com/eu-cdse/utilities.git#main" -t cdse_utilities
  ```

  The host must provide `docker`, `zip`, and `unzip`, and the Docker daemon must be accessible to the invoking user. Conversion temporarily needs space for the staged COG ZIP, restored GRD ZIP, and extracted SAFE; the launcher enforces the configurable `MIN_FREE_GIB` threshold before starting.

  Edit the paths and three product names at the top of `restore_cog_safe_to_grd.sh`, then run:

  ```shell
  bash restore_cog_safe_to_grd.sh
  ```

  This restoration launcher does not run SNAP or model inference. It creates store-only `_COG.zip` staging archives, runs the official `COG2GRD.sh` inside Docker, validates the regenerated standard SAFE products, and caches them under `restored_grd/`. The original COG SAFE directories are never mounted into Docker or modified. Completed products are reused on later runs.

  The regenerated SAFE names contain newly calculated CRC values and therefore cannot be predicted in advance. The launcher prints the actual COG-to-GRD mappings and creates a common `restored_grd/products/` directory containing links to the validated products. Copy the printed `DATA_ROOT`, `PRE_SAFE_NAME`, and `POST_SAFE_NAME` values into `predict_safe_pair.sh`.

  For already restored or original standard GRD SAFE products, inference can also be invoked directly:

  ```shell
  uv run python infer_safe.py \
    /path/to/pre_event.SAFE \
    /path/to/post_event.SAFE \
    --checkpoint /path/to/checkpoint.pth \
    --output /path/to/flood_map.tif \
    --trust-checkpoint
  ```

  For repeatable GPU-server runs, edit the configuration block at the top of `predict_safe_pair.sh`, then execute:

  ```shell
  bash predict_safe_pair.sh
  ```

  The prediction launcher validates all paths, optionally synchronizes the locked uv environment, prints the effective configuration, and saves console output beside the prediction as a `.log` file.

  SNAP preprocessing outputs are cached per SAFE product under `${WORK_DIR}/snap-cache` by the launcher. The cache key includes the SAFE manifest, sampled VV measurement content, annotation-file hashes, SNAP graph, GPT launcher content, orbit type, DEM, target CRS, and pixel spacing. Unrelated preview or note files inside the SAFE do not invalidate the cache. Checkpoint, device, threshold, dB mapping, batch size, and stride changes also do not invalidate it. Consequently, a second run of the same pair reuses both Sigma0 rasters, while changing only the post-event product reuses the pre-event raster and preprocesses only the new post-event product.

  The launcher exposes `USE_SNAP_CACHE` and `REFRESH_SNAP_CACHE` in its configuration block. Set `REFRESH_SNAP_CACHE=true` after changing SNAP modules, orbit/DEM infrastructure, or when intentionally rebuilding matching entries. Direct CLI users can select a persistent location with `--snap-cache-dir` or `SNAP_CACHE_DIR`; otherwise `<work-dir>/snap-cache` is used when `--work-dir` is supplied, and caching is disabled when neither is configured. `--no-snap-cache` explicitly overrides the environment default. Cache entries are validated before reuse, incomplete or corrupt entries are rebuilt, and refreshes atomically publish an immutable generation so concurrent readers keep a valid raster. Cached generations persist independently of `--keep-intermediate` and are not automatically evicted; refreshes retain older generations for reader safety, so monitor cache disk usage and remove obsolete entries only when no inference process is using them.

  The primary output is a georeferenced `uint8` mask (`0` background, `255` flood). A `float32` flood-probability GeoTIFF is written beside it as `<output>_probability.tif`. Overlapping `256×256` predictions are blended before thresholding, and `tqdm` reports SNAP, inference, and output progress.

  The model was trained on VV-like 8-bit grayscale images replicated into three channels, but the paper does not document the exact raw-SAR intensity conversion. Deployment therefore maps calibrated linear Sigma0 through a configurable fixed dB range, `[-25, 0] dB` by default. Adjust `--db-min` and `--db-max` only when a validated preprocessing recipe is available. Inputs are not divided by 255 or ImageNet-normalized.

  Current checkpoints must include the registered TACE Q/K/V projections introduced in commit `11c309a`. Older full-model checkpoints cannot be repaired faithfully and are rejected with an explicit error. Because checkpoints are loaded as complete Python objects, use `--trust-checkpoint` only for files from a trusted source.
  
## Results

<p align="center">
<img src=./images/R1.png width="100%">
</P>


## Visualization

<p align="center">
<img src=./images/R2.png width="100%">
</P>


### :page_with_curl: Citing <a name="citing"></a>

```bibtex
@article{saleh2024dam,
  title={DAM-Net: Flood detection from SAR imagery using differential attention metric-based vision transformers},
  author={Saleh, Tamer and Weng, Xingxing and Holail, Shimaa and Hao, Chen and Xia, Gui-Song},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={212},
  pages={440--453},
  year={2024},
  publisher={Elsevier}
}
```

  
## Contact Information

If you have any questions or would like to collaborate, please reach out to me at tamersaleh@whu.edu.cn.

## License
The datasets are released for non-commercial and research purposes only. For commercial purposes, please contact the authors.


## Acknowledgment

Appreciate the work from the following repositories:

- [wenhwu/awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
- [SNUNet-CD](https://github.com/RSCD-Lab/Siam-NestedUNet)
- [BIT-CD](https://github.com/justchenhao/BIT_CD)

## Related resources
- [ASF-Dataset](https://search.asf.alaska.edu/)
- [Sentinel-Hub](https://scihub.copernicus.eu/)
- [SNAP Toolbox](http://step.esa.int/main/download/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Tamer-Saleh/S1GFlood-Detection&type=Date)](https://star-history.com/#Tamer-Saleh/S1GFlood-Detection&Date)
