# Senegal Land Cover & Crop Classification Pipeline

A unified, config-driven pipeline for pixel-level land cover and crop-type classification of Senegal satellite imagery. Supports five feature extraction approaches, five classifier types, multi-seed ensemble aggregation, and whole-map inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [Feature Approaches](#feature-approaches)
7. [Classification Tasks](#classification-tasks)
8. [Models](#models)
9. [Training Strategies](#training-strategies)
10. [Analysis Scripts](#analysis-scripts)
11. [Visualization](#visualization)
12. [Output Files](#output-files)
13. [Data Requirements](#data-requirements)

---

## Overview

This pipeline was refactored from a collection of ~30 single-purpose scripts into a modular package. All behaviour is controlled through a single `config.yaml` file — no source edits are required to switch approaches, years, or models.

The pipeline supports:
- **Two classification tasks**: land cover (8 classes) and main crop type (7 classes)
- **Five feature approaches**: raw Sentinel-2 + SAR time series, Tessera representations, spectral-temporal metrics (STM), AlphaEarth EFMs, and spectral matching NDVI
- **Five classifier types**: Random Forest, Logistic Regression, XGBoost, SVM, MLP
- **Multi-seed ensemble aggregation**: train N models with different random seeds, average their probability outputs
- **Whole-map inference**: classify every pixel in the scene, not just labelled fields
- **Three analysis utilities**: year-on-year change detection, cropland combo mask generation, accuracy summarization

---

## Repository Structure

```
senegal_kinabalu/
│
├── classify.py              # Main pipeline entry point
├── config.yaml              # All run settings (edit this, not the source)
│
├── utils/
│   ├── data_loading.py      # Config loading, label loading, per-approach feature loaders
│   ├── geo_utils.py         # GeoTIFF conversion, raster clipping, chunk grid, save helpers
│   ├── preprocessing.py     # Train/val/test splitting, SMOTE, label remapping, safe stacking
│   └── reporting.py         # Classification report CSV writing and summarization
│
├── models/
│   ├── classifiers.py       # RF, LR, XGBoost, SVM trainers + ensemble predict_proba
│   └── neural.py            # PyTorch MLP with FocalLoss, BatchNorm, early stopping
│
├── analysis/
│   ├── average_agg_accuracies.py    # Summarize aggregated (multi-seed) report CSVs
│   ├── average_single_accuracies.py # Summarize single-model per-seed report CSVs
│   ├── common_cropland_map.py       # Build stable cropland mask across all years
│   ├── quantify_areas.py            # Per-class pixel area percentages across years
│   └── yeartoyear_comparison.py     # Year-on-year cropland change map
│
└── visualization/
    └── maps.py              # Plot prediction maps as publication-quality PNGs
```

---

## Installation

The pipeline requires Python 3.9+ and the following packages:

```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost torch \
            rasterio geopandas matplotlib pyyaml joblib
```

---

## Quick Start

1. **Edit `config.yaml`** — set `approach`, `classification`, `year`, and update all `paths` to point to your data.

2. **Run the pipeline**:
   ```bash
   python classify.py
   # or with an explicit config path:
   python classify.py --config config.yaml
   ```

3. **Check outputs** in the directories specified by `paths.landcoverclassification` / `paths.cropclassification` and `paths.classification_reports`.

---

## Configuration Reference

All settings live in `config.yaml`. The `{year}` token in path strings is automatically replaced at runtime with the value of `year`.

### Top-level settings

| Key | Values | Description |
|-----|--------|-------------|
| `approach` | `raw` \| `tessera` \| `stm` \| `alphaearth` \| `specmat` | Feature extraction method |
| `classification` | `landcover` \| `maincrop` | Classification task |
| `year` | `2018` \| `2019` \| `2021` | Data year |
| `models` | list of model names | Models to train, e.g. `["RandomForest", "MLP"]` |
| `num_seeds` | integer | Number of seeds when aggregation is disabled |
| `njobs` | integer | Parallel jobs for joblib (chunk processing) |
| `chunk_size` | integer | Spatial tile size in pixels (e.g. `1000` = 1000×1000 px tiles) |
| `wholemap` | bool | Run whole-map inference after training |
| `save_maps` | bool | Save prediction maps as `.npy` and `.tiff` |
| `save_report` | bool | Save per-seed classification report CSVs |
| `save_confmat` | bool | Save confusion matrix CSVs |
| `remap_2021_labels` | bool | Zero out pasture (7) and natural vegetation (8) for 2021 |

### `training` block

| Key | Values | Description |
|-----|--------|-------------|
| `ratio` | float (0–1) | Fraction of field area/count assigned to training |
| `sampling` | `bypercentage` \| `bypercentage_count` \| `bycount` | Field splitting strategy (see [Training Strategies](#training-strategies)) |
| `augment` | bool | Apply SMOTE oversampling to balance the training set |

### `aggregation` block

| Key | Values | Description |
|-----|--------|-------------|
| `enabled` | bool | Train `num_runs` models and average their probability outputs |
| `num_runs` | integer | Number of models in the ensemble |

### `mlp` block (only used when `MLP` is in `models`)

| Key | Description |
|-----|-------------|
| `batch_size` | Mini-batch size for training and inference |
| `learning_rate` | Adam optimizer learning rate |
| `num_epochs` | Maximum training epochs |
| `patience` | Early stopping patience (epochs without val loss improvement) |
| `hidden_sizes` | List of 3 hidden layer widths, e.g. `[2048, 1024, 512]` |
| `dropout_rate` | Dropout probability applied after each hidden layer |
| `focal_loss.alpha` | Focal loss scaling factor |
| `focal_loss.gamma` | Focal loss focusing parameter (higher = more focus on hard examples) |

### `paths` block

All paths support the `{year}` placeholder. Key paths:

| Key | Description |
|-----|-------------|
| `ref_tiff` | Reference GeoTIFF used to georeference all output rasters |
| `field_ids` | `.npy` raster mapping every pixel to its field polygon ID |
| `fielddata` | CSV with field metadata (`Id`, `Year`, `Area_ha`, class codes) |
| `label_landcover` / `label_maincrop` | Label rasters for each task |
| `s2_bands`, `s2_mask`, `sar_bands` | Raw approach inputs |
| `tessera_reps`, `tessera_scales` | Tessera int8 representations and per-pixel scale factors |
| `stm_group0`–`stm_group5` | STM band group files |
| `alphaearth_efm` | AlphaEarth EFM feature array |
| `specmat_ndvi` | Monthly NDVI time series array |
| `landcoverclassification` / `cropclassification` | Output folders for prediction maps |
| `classification_reports` | Output folder for accuracy report CSVs |
| `cropland_combo_mask` | Stable cropland mask used when `classification == maincrop` |

---

## Feature Approaches

### `raw` — Sentinel-2 + Sentinel-1 time series
Loads multi-date optical bands (10 bands × T timesteps) and SAR backscatter (VV, VH × T timesteps). Applies cloud masking and Z-score normalisation using pre-computed dataset statistics. Feature vector length: `T_s2 × 10 + T_sar × 2`.

### `tessera` — Tessera learned representations
Loads 128-dimensional int8 feature maps with per-pixel scale factors and dequantizes to float32. Feature vector length: `128`.

### `stm` — Spectral-Temporal Metrics
Loads 6 pre-computed band group files and concatenates along the band dimension. Feature vector length: sum of bands across all 6 groups.

### `alphaearth` — AlphaEarth EFM
Loads a `(H, W, N)` pre-computed feature array directly. Feature vector length: `N` (typically 64).

### `specmat` — Spectral Matching (NDVI)
Loads a `(T, H, W)` monthly NDVI array and transposes to `(H×W, T)`. Feature vector length: number of monthly timesteps.

---

## Classification Tasks

### `landcover` — Land Cover (8 classes)

| Code | Class |
|------|-------|
| 1 | Built-up surface |
| 2 | Bare soil |
| 3 | Water body |
| 4 | Wetland |
| 5 | Cropland |
| 6 | Shrub land |
| 7 | Pasture |
| 8 | Natural vegetation |

> **Note**: When `remap_2021_labels: true`, classes 7 and 8 are zeroed out for 2021 data because the ground-truth survey does not distinguish them reliably for that year.

### `maincrop` — Main Crop Type (7 classes)

| Code | Class |
|------|-------|
| 1 | Maize |
| 2 | Rice |
| 3 | Sorghum |
| 4 | Millet |
| 5 | Groundnut |
| 6 | Other |
| 7 | Fallow/Bare |

> **Note**: The maincrop pipeline requires a pre-computed cropland combo mask (`paths.cropland_combo_mask`). Only pixels inside this mask are classified — see [Analysis Scripts](#analysis-scripts) for how to generate it.

---

## Models

| Name in config | Algorithm | Notes |
|----------------|-----------|-------|
| `RandomForest` | sklearn `RandomForestClassifier` | 100 trees, no depth limit |
| `LogisticRegression` | sklearn `LogisticRegression` | Multinomial softmax, C=1e4 |
| `XGBOOST` | `XGBClassifier` | 400 estimators, depth 3, softprob objective |
| `SVM` | sklearn `SVC` | Probability estimates enabled via Platt scaling |
| `MLP` | PyTorch 3-layer MLP | BatchNorm + Dropout + FocalLoss + early stopping |

Multiple models can be listed:
```yaml
models: ["RandomForest", "MLP"]
```

---

## Training Strategies

Controlled by `training.sampling`:

| Strategy | Description |
|----------|-------------|
| `bypercentage` | Greedily selects fields by hectare area until `ratio` of total class area is in training |
| `bypercentage_count` | Selects `ratio` of fields by count per class |
| `bycount` | Reserves exactly 5 fields per class for val and test; all remaining go to train |

---

## Analysis Scripts

All scripts accept `--help` for a full argument list. Defaults reproduce the original hardcoded behaviour.

### `analysis/average_agg_accuracies.py`
Reads multi-seed aggregated classification report CSVs and prints mean ± std for accuracy, macro-avg F1, and weighted-avg F1.

```bash
python analysis/average_agg_accuracies.py \
    --years 2018 2019 2021 \
    --approaches tessera raw stm specmat \
    --classification landcover \
    --report_root /maps/mcl66/senegal/classification_reports
```

### `analysis/average_single_accuracies.py`
Same as above but for single-model (non-aggregated) report CSVs. Matches files by year and classification type using glob.

```bash
python analysis/average_single_accuracies.py \
    --years 2018 \
    --classification landcover \
    --csv_folder /maps/mcl66/senegal/classification_reports_singlemodel/
```

### `analysis/common_cropland_map.py`
Loads landcover prediction maps for 2018, 2019, and 2021, remaps to binary cropland/non-cropland, and identifies pixels that are cropland in **all three years**. The output is used as the cropland mask for maincrop classification.

```bash
python analysis/common_cropland_map.py \
    --approach tessera \
    --map_root /maps/mcl66/senegal/landcoverclassification \
    --num_agg 15 \
    --out_root /maps/mcl66/senegal/landcoverclassification
```

### `analysis/quantify_areas.py`
Computes the percentage of pixels in each land cover or crop class across all years (2018, 2019, 2021) and all three approaches (tessera, stm, raw). Optionally saves results to CSV.

```bash
python analysis/quantify_areas.py \
    --classification landcover \
    --data_root /maps/mcl66/senegal \
    --num_agg 15 \
    --save          # use --no-save to print only
```

### `analysis/yeartoyear_comparison.py`
Remaps two prediction maps to binary cropland and subtracts them to produce a change map: `-1` = crop loss, `0` = no change, `+1` = crop gain (saved shifted to `0/1/2` for visualization).

```bash
python analysis/yeartoyear_comparison.py \
    --data_source tessera \
    --year1 2019 \
    --year2 2021 \
    --num_agg 15 \
    --save
```

---

## Visualization

```bash
python visualization/maps.py path/to/prediction_map.npy \
    --type landcover \
    --year 2021 \
    --method tessera
```

Saves a 600 DPI PNG alongside the `.npy` file and converts to GeoTIFF using the reference raster. Use `--type cropcover` for crop maps; the 2021 extended palette (9 classes) is selected automatically when `--year 2021`.

---

## Output Files

| File pattern | Description |
|-------------|-------------|
| `senegal_{approach}_prediction_map_whole_{year}_{N}agg.npy/.tiff` | Whole-map prediction raster |
| `senegal_{approach}_prediction_map_whole_{year}_remapped_{N}agg.npy/.tiff` | Same, with 2021 label remapping |
| `senegal_{approach}_classification_report_{year}_{model}_{task}.csv` | Per-seed accuracy report |
| `senegal_{approach}_classification_report_{year}_agg_{task}.csv` | Aggregated ensemble accuracy report |
| `senegal_{approach}_confmat_{year}_{model}_{task}.csv` | Confusion matrix |
| `senegal_{approach}_croplandcombo_map.npy/.tiff` | Stable cropland mask |
| `senegal_{approach}_change_map_{year1}_{year2}.npy/.tiff` | Year-on-year change map |
| `senegal_unique_counts.csv` | Per-class pixel percentages across years |

---

## Data Requirements

The pipeline expects pre-processed `.npy` arrays. All spatial arrays must share the same extent and resolution (10 m, UTM 28N, clipped to central Senegal). Required inputs per approach:

| Approach | Required files |
|----------|---------------|
| `raw` | `s2_bands`, `s2_mask`, `sar_bands` |
| `tessera` | `tessera_reps`, `tessera_scales` |
| `stm` | `stm_group0` … `stm_group5` |
| `alphaearth` | `alphaearth_efm` |
| `specmat` | `specmat_ndvi` |

All approaches also require: `field_ids`, `fielddata`, `label_landcover` or `label_maincrop`, and `ref_tiff`.
