"""
classify.py — unified pipeline entry point for Senegal land cover / crop classification.

Usage:
    python classify.py [--config config.yaml]

All run settings are controlled via config.yaml.  The approach-specific feature
loading is handled by FEATURE_LOADERS dispatch table in utils/data_loading.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from joblib import Parallel, delayed
from models.classifiers import ensemble_predict_proba, train_model
from sklearn.metrics import classification_report
from utils.data_loading import (
    FEATURE_LOADERS,
    load_config,
    load_labels,
    validate_config,
)
from utils.geo_utils import get_chunk_grid, save_prediction
from utils.preprocessing import (
    identify_valid_classes,
    mask_classes_2021,
    safe_hstack,
    safe_vstack,
    split_field_ids,
)
from utils.reporting import save_classification_report, save_confusion_matrix

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Human-readable names for the 8 landcover classes (codes 1–8).
LANDCOVER_CLASS_NAMES = [
    "Built-up surface",
    "Bare soil",
    "Water body",
    "Wetland",
    "Cropland",
    "Shrub land",
    "Pasture",
    "Natural vegetation",
]
# Human-readable names for the 7 main crop classes (codes 1–7).
MAINCROP_CLASS_NAMES = [
    "Maize",
    "Rice",
    "Sorghum",
    "Millet",
    "Groundnut",
    "Sesame",
    "Cotton",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_pipeline_data(cfg: dict) -> tuple[
    npt.NDArray,  # labels
    npt.NDArray,  # field_ids
    int,  # H
    int,  # W
    pd.DataFrame,  # fielddata_df
    set[int],  # valid_classes
    list[tuple[int, int, int, int]],  # chunks
    npt.NDArray | None,  # agg_pred_map_mask
    str,  # outfolder
    list[str],  # class_names
    str,  # classcode
    bool,  # reduce_labels
]:
    """Set up classification-specific paths, load labels, and prepare chunk grid."""
    year = cfg["year"]
    classification = cfg["classification"]

    # Select the label raster, output folder, and class names based on task type.
    if classification == "landcover":
        classcode = "landcover_code"
        cfg["paths"]["label"] = cfg["paths"]["label_landcover"]
        outfolder = cfg["paths"]["landcoverclassification"]
        class_names = LANDCOVER_CLASS_NAMES
    else:
        classcode = "maincrop_code"
        cfg["paths"]["label"] = cfg["paths"]["label_maincrop"]
        outfolder = cfg["paths"]["cropclassification"]
        class_names = MAINCROP_CLASS_NAMES

    # Create output directories up-front so later save calls never fail mid-run.
    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(cfg["paths"]["classification_reports"], exist_ok=True)

    logging.info("Loading labels and field IDs...")
    labels, field_ids = load_labels(cfg)
    H, W = labels.shape
    logging.info(f"Data dimensions: {H}x{W}")

    # For 2021, pasture (7) and natural vegetation (8) are excluded because the
    # ground-truth dataset doesn't distinguish them reliably for that year.
    reduce_labels = cfg["remap_2021_labels"] and year == 2021
    if reduce_labels:
        logging.info("Masking pasture/natural-veg labels for 2021...")
        labels, field_ids = mask_classes_2021(labels, field_ids)

    # For maincrop mode, build a binary cropland mask from a prior landcover run.
    # Only pixels predicted as cropland are processed during maincrop inference.
    agg_pred_map_mask = None
    if classification == "maincrop":
        agg_pred_map = np.load(cfg["paths"]["cropland_combo_mask"])
        # Map landcover codes to binary: 4 (Cropland) → 1, everything else → 0.
        remap = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0}
        max_key = max(remap)
        lut = np.zeros(max_key + 1, dtype=np.float32)
        for k, v in remap.items():
            lut[k] = v
        # clip handles any out-of-range values; int64 cast is required for indexing.
        agg_pred_map_mask = lut[np.clip(agg_pred_map.astype(np.int64), 0, max_key)]

    # Exclude background (0) and any class with fewer than 2 pixels.
    valid_classes = identify_valid_classes(labels, "generic")
    logging.info(f"Valid classes: {sorted(valid_classes)}")

    # Split the raster into non-overlapping spatial tiles processed in parallel.
    chunks = get_chunk_grid(H, W, cfg["chunk_size"])

    # Load field-level metadata and filter to the current year.
    fielddata_df = pd.read_csv(cfg["paths"]["fielddata"])
    fielddata_df = fielddata_df[fielddata_df["Year"] == year]

    return (
        labels,
        field_ids,
        H,
        W,
        fielddata_df,
        valid_classes,
        chunks,
        agg_pred_map_mask,
        outfolder,
        class_names,
        classcode,
        reduce_labels,
    )


def _run_training_loop(
    cfg: dict,
    labels: npt.NDArray,
    field_ids: npt.NDArray,
    fielddata_df: pd.DataFrame,
    valid_classes: set[int],
    chunks: list[tuple[int, int, int, int]],
    loader: Callable,
    outfolder: str,
    class_names: list[str],
    classcode: str,
) -> list:
    """Iterate over models and seeds, train each, evaluate per-seed if not aggregating."""
    agg_enabled = cfg["aggregation"]["enabled"]
    # When aggregation is on, num_runs controls how many models form the ensemble.
    num_runs = (
        cfg["aggregation"]["num_runs"] if agg_enabled else cfg.get("num_seeds", 10)
    )
    approach = cfg["approach"]
    year = cfg["year"]
    classification = cfg["classification"]
    njobs = cfg["njobs"]

    trained_models = []
    seeds = list(range(1, num_runs + 1))

    for model_name in cfg["models"]:
        # XGBOOST and MLP need a validation set for early stopping / calibration,
        # so they use a larger val fraction (1/4) than simpler models (1/20).
        if model_name in ("XGBOOST", "MLP"):
            val_test_ratio = 1 / 4
        else:
            val_test_ratio = 1 / 20

        for seed in seeds:
            # Set all random seeds for reproducibility across frameworks.
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Expose seed via cfg so train_model can retrieve it without extra parameters.
            cfg["_current_seed"] = seed
            start = time.process_time()

            logging.info(f"\n{'='*60}")
            logging.info(f"Model: {model_name} | Seed: {seed}")

            # Deterministic field-level train/val/test split for this seed.
            train_fids, val_fids, test_fids = split_field_ids(
                fielddata_df,
                cfg["training"]["sampling"],
                cfg["training"]["ratio"],
                val_test_ratio,
                seed,
                classcode,
            )
            logging.info(
                f"Train FIDs: {len(train_fids)}, "
                f"Val FIDs: {len(val_fids)}, "
                f"Test FIDs: {len(test_fids)}"
            )

            # Inner function captures the current seed's split arrays via closure.
            # Each spatial chunk is processed independently; joblib spreads them
            # across CPU cores.
            def process_chunk(h0, h1, w0, w1):
                # Load features for this spatial tile (mmap avoids full-RAM load).
                X_chunk = loader(cfg, h0, h1, w0, w1)
                y_chunk = labels[h0:h1, w0:w1].ravel()
                fid_chunk = field_ids[h0:h1, w0:w1].ravel()

                # Remove pixels whose class code isn't in the valid set.
                valid_mask = np.isin(y_chunk, list(valid_classes))
                X_chunk = X_chunk[valid_mask]
                y_chunk = y_chunk[valid_mask]
                fid_chunk = fid_chunk[valid_mask]

                # Assign each pixel to train / val / test based on its field ID.
                tr = np.isin(fid_chunk, train_fids)
                va = np.isin(fid_chunk, val_fids)
                te = np.isin(fid_chunk, test_fids)

                return (
                    X_chunk[tr],
                    y_chunk[tr],
                    X_chunk[va],
                    y_chunk[va],
                    X_chunk[te],
                    y_chunk[te],
                )

            results = Parallel(n_jobs=njobs)(
                delayed(process_chunk)(h0, h1, w0, w1) for h0, h1, w0, w1 in chunks
            )

            # Infer feature dimension from the first non-empty chunk result.
            feature_dim = next((r[0].shape[1] for r in results if r[0].size > 0), None)
            if feature_dim is None:
                raise ValueError("No training data found to infer feature dimension!")
            logging.info(f"Feature dimension: {feature_dim}")

            # Concatenate all chunk results into full train/val/test arrays.
            # safe_vstack/hstack handle the case where some chunks returned nothing.
            X_train = safe_vstack([r[0] for r in results], (0, feature_dim))
            y_train = safe_hstack([r[1] for r in results], (0,))
            X_val = safe_vstack([r[2] for r in results], (0, feature_dim))
            y_val = safe_hstack([r[3] for r in results], (0,))
            X_test = safe_vstack([r[4] for r in results], (0, feature_dim))
            y_test = safe_hstack([r[5] for r in results], (0,))

            # Remove background pixels (class 0) from training only.
            # XGBoost handles background as a valid class internally (code -1).
            if model_name != "XGBOOST":
                bg_mask = y_train != 0
                X_train = X_train[bg_mask]
                y_train = y_train[bg_mask]

            logging.info(
                f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
            )

            # Train the model for this seed.
            model = train_model(model_name, X_train, y_train, X_val, y_val, cfg)
            trained_models.append(model)

            # Per-seed evaluation is only run when not in aggregation mode
            # (aggregation evaluates the full ensemble after all seeds are done).
            if not agg_enabled:
                y_pred = model.predict(X_test)
                logging.info(
                    "Classification Report:\n"
                    + classification_report(y_test, y_pred, digits=4)
                )
                cpu_time = time.process_time() - start

                if cfg["save_report"]:
                    report_path = os.path.join(
                        cfg["paths"]["classification_reports"],
                        f"senegal_{approach}_classification_report_{year}_"
                        f"{model_name.lower()}_{classification}.csv",
                    )
                    save_classification_report(
                        y_test, y_pred, seed, cpu_time, report_path
                    )

                if cfg["save_confmat"]:
                    confmat_path = os.path.join(
                        cfg["paths"]["classification_reports"],
                        f"senegal_{approach}_confmat_{year}_{model_name.lower()}_{classification}.csv",
                    )
                    save_confusion_matrix(y_test, y_pred, class_names, confmat_path)

    return trained_models


def _run_aggregated_evaluation(
    cfg: dict,
    labels: npt.NDArray,
    valid_classes: set[int],
    chunks: list[tuple[int, int, int, int]],
    loader: Callable,
    trained_models: list,
) -> None:
    """Run ensemble evaluation across all labelled pixels and optionally save report."""
    approach = cfg["approach"]
    year = cfg["year"]
    classification = cfg["classification"]
    njobs = cfg["njobs"]

    logging.info("\nRunning ensemble evaluation on all labelled pixels...")

    # Collect features and labels for every labelled pixel across all spatial chunks.
    def collect_chunk(h0, h1, w0, w1):
        X_chunk = loader(cfg, h0, h1, w0, w1)
        y_chunk = labels[h0:h1, w0:w1].ravel()
        # Keep only pixels with a valid (non-background) class code.
        valid_mask = np.isin(y_chunk, list(valid_classes))
        return X_chunk[valid_mask], y_chunk[valid_mask]

    all_results = Parallel(n_jobs=njobs)(
        delayed(collect_chunk)(h0, h1, w0, w1) for h0, h1, w0, w1 in chunks
    )

    # Stack all chunks into a single feature matrix and label vector.
    feature_dim = next((r[0].shape[1] for r in all_results if r[0].size > 0), None)
    X_all = safe_vstack([r[0] for r in all_results], (0, feature_dim))
    y_all = safe_hstack([r[1] for r in all_results], (0,))

    # Average predicted probabilities across all trained models, then take argmax.
    # +1 converts 0-based argmax back to 1-based class codes.
    y_probs = ensemble_predict_proba(trained_models, X_all)
    y_pred_agg = np.argmax(y_probs, axis=1) + 1

    logging.info(
        "Ensemble Classification Report:\n"
        + classification_report(y_all, y_pred_agg, digits=4)
    )

    if cfg["save_report"]:
        agg_report_path = os.path.join(
            cfg["paths"]["classification_reports"],
            f"senegal_{approach}_classification_report_{year}_agg_{classification}.csv",
        )
        # seed=0 and cpu_time=0 are sentinel values indicating this is the ensemble row.
        save_classification_report(y_all, y_pred_agg, 0, 0, agg_report_path)


def _predict_whole_map(
    cfg: dict,
    H: int,
    W: int,
    labels: npt.NDArray,
    valid_classes: set[int],
    chunks: list[tuple[int, int, int, int]],
    loader: Callable,
    trained_models: list,
    agg_pred_map_mask: npt.NDArray | None,
    outfolder: str,
    remapped_suffix: str,
    num_runs: int,
) -> npt.NDArray:
    """Generate whole-map prediction using the trained model ensemble and optionally save."""
    approach = cfg["approach"]
    year = cfg["year"]
    classification = cfg["classification"]
    agg_enabled = cfg["aggregation"]["enabled"]
    njobs = cfg["njobs"]

    logging.info("\nGenerating whole-map prediction...")
    # Initialise output map to zero (background / no prediction).
    pred_map_whole = np.zeros((H, W), dtype=np.int64)

    def batch_predict_chunk(h0, h1, w0, w1):
        # Start with an empty (all-zero) prediction for this tile.
        chunk_pred = np.zeros((h1 - h0, w1 - w0), dtype=np.int64)

        # For landcover, predict every pixel; for maincrop, skip non-cropland pixels
        # to avoid making predictions outside the area of interest.
        if classification == "landcover":
            predict_mask = np.ones((h1 - h0, w1 - w0), dtype=bool)
        else:
            predict_mask = agg_pred_map_mask[h0:h1, w0:w1].astype(bool)

        # Skip chunks where no pixels need predicting (e.g. fully non-cropland tiles).
        if not np.any(predict_mask):
            return h0, h1, w0, w1, chunk_pred

        X_chunk = loader(cfg, h0, h1, w0, w1)  # ((h1-h0)*(w1-w0), features)
        predict_flat = predict_mask.ravel()
        # Select only the pixels flagged for prediction to avoid wasted inference.
        X_pred = X_chunk[predict_flat]

        # Use ensemble averaging when multiple models are available; otherwise
        # use the last trained model's hard predictions directly.
        if len(trained_models) > 1 and agg_enabled:
            probs = ensemble_predict_proba(trained_models, X_pred)
            preds = np.argmax(probs, axis=1) + 1  # 0-based argmax → 1-based code
        else:
            preds = trained_models[-1].predict(X_pred)

        # Re-expand the flat predictions back to the tile's spatial shape.
        chunk_pred_flat = np.zeros((h1 - h0) * (w1 - w0), dtype=np.int64)
        chunk_pred_flat[predict_flat] = preds
        chunk_pred = chunk_pred_flat.reshape(h1 - h0, w1 - w0)

        return h0, h1, w0, w1, chunk_pred

    # Parallelise inference over all spatial tiles.
    pred_results = Parallel(n_jobs=njobs)(
        delayed(batch_predict_chunk)(h0, h1, w0, w1) for h0, h1, w0, w1 in chunks
    )

    # Assemble the per-tile predictions into the full-resolution output map.
    for h0, h1, w0, w1, chunk_pred in pred_results:
        pred_map_whole[h0:h1, w0:w1] = chunk_pred

    if cfg["save_maps"]:
        prefix = os.path.join(
            outfolder,
            f"senegal_{approach}_prediction_map_whole_{year}"
            f"{remapped_suffix}_{num_runs}agg",
        )
        save_prediction(pred_map_whole, prefix, cfg)
        logging.info(f"Saved prediction map: {prefix}.npy / .tiff")

    return pred_map_whole


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    validate_config(cfg)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    # Write logs to both a file (for later review) and stdout (for live monitoring).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("classify.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(
        f"Approach: {cfg['approach']}, "
        f"Classification: {cfg['classification']}, "
        f"Year: {cfg['year']}"
    )

    # Load all data and set up pipeline state in a single call.
    (
        labels,
        field_ids,
        H,
        W,
        fielddata_df,
        valid_classes,
        chunks,
        agg_pred_map_mask,
        outfolder,
        class_names,
        classcode,
        reduce_labels,
    ) = _load_pipeline_data(cfg)

    # Select the feature-loading function for the configured approach.
    loader = FEATURE_LOADERS[cfg["approach"]]
    agg_enabled = cfg["aggregation"]["enabled"]
    # Suffix appended to output filenames when 2021 labels have been remapped.
    remapped_suffix = "_remapped" if reduce_labels else ""
    num_runs = (
        cfg["aggregation"]["num_runs"] if agg_enabled else cfg.get("num_seeds", 10)
    )

    # Train all models (one per seed × model_name combination).
    trained_models = _run_training_loop(
        cfg,
        labels,
        field_ids,
        fielddata_df,
        valid_classes,
        chunks,
        loader,
        outfolder,
        class_names,
        classcode,
    )

    # Ensemble evaluation on the full labelled dataset (only in aggregation mode).
    if agg_enabled and trained_models:
        _run_aggregated_evaluation(
            cfg, labels, valid_classes, chunks, loader, trained_models
        )

    # Whole-map inference: predict every pixel in the scene (not just labelled ones).
    if cfg["wholemap"] or cfg["save_maps"]:
        _predict_whole_map(
            cfg,
            H,
            W,
            labels,
            valid_classes,
            chunks,
            loader,
            trained_models,
            agg_pred_map_mask,
            outfolder,
            remapped_suffix,
            num_runs,
        )

    logging.info("Pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Senegal land cover / crop classification"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
