from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.over_sampling import SMOTE


def split_field_ids(
    df: pd.DataFrame,
    sampling: str,
    ratio: float,
    val_test_ratio: float,
    seed: int,
    classcode: str,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Split field IDs into train/val/test sets using the specified sampling strategy."""
    # Shuffle rows so that greedy area-filling picks fields in a random order each seed.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Pre-compute per-class totals used by the area- and count-based strategies.
    area_summary = df.groupby(classcode)["Area_ha"].sum().reset_index()
    count_summary = df.groupby(classcode).size().reset_index(name="count")

    train_fids = []
    val_fids = []
    test_fids = []

    if sampling == "bypercentage":
        # Allocate training fields greedily by area until the target fraction is reached,
        # then split the remainder into val and test by count.
        for _, row in area_summary.iterrows():
            sn_code = row[classcode]
            # Target total hectares to assign to the training split for this class.
            target_train_area = row["Area_ha"] * ratio

            # Sort fields by size so small fields are added first, giving finer
            # control over how close we get to the target area.
            rows_sncode = (
                df[df[classcode] == sn_code]
                .sort_values(by="Area_ha")
                .reset_index(drop=True)
            )

            selected_fids = []
            selected_area = 0.0
            for _, r2 in rows_sncode.iterrows():
                if selected_area < target_train_area:
                    selected_fids.append(int(r2["Id"]))
                    selected_area += r2["Area_ha"]
                else:
                    break

            train_fids.extend(selected_fids)
            # Remaining fields (not selected for train) go to val / test.
            remaining = (
                rows_sncode[~rows_sncode["Id"].isin(selected_fids)]
                .copy()
                .sample(frac=1, random_state=42)  # re-shuffle so val/test are random
                .reset_index(drop=True)
            )
            # Ensure at least 1 field in val even for very small classes.
            val_count = max(1, int(len(remaining) * val_test_ratio))
            val_fids.extend(remaining.iloc[:val_count]["Id"].astype(int).tolist())
            test_fids.extend(remaining.iloc[val_count:]["Id"].astype(int).tolist())

    elif sampling == "bypercentage_count":
        # Same logic as bypercentage but uses field count rather than area as the budget.
        for _, row in count_summary.iterrows():
            sn_code = row[classcode]
            target_train_count = int(row["count"] * ratio)

            rows_sncode = (
                df[df[classcode] == sn_code]
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            selected_train = rows_sncode.iloc[:target_train_count]
            remaining = rows_sncode.iloc[target_train_count:]
            val_count = max(1, int(len(remaining) * val_test_ratio))

            train_fids.extend(selected_train["Id"].astype(int).tolist())
            val_fids.extend(remaining.iloc[:val_count]["Id"].astype(int).tolist())
            test_fids.extend(remaining.iloc[val_count:]["Id"].astype(int).tolist())

    elif sampling == "bycount":
        # Reserve a fixed number of fields per class for val and test, put the rest
        # in train.  Simple and class-balanced by construction.
        NUM_FIELDS = 5  # Fixed number of fields reserved per class for val and test splits (bycount strategy)
        for sn_code in df[classcode].unique():
            fids = df[df[classcode] == sn_code]["Id"].unique()
            fids = np.array(fids)
            np.random.shuffle(fids)  # in-place shuffle for random assignment

            if len(fids) >= NUM_FIELDS * 2:
                # Enough fields: reserve exactly NUM_FIELDS each for test and val.
                test_fids.extend(fids[:NUM_FIELDS])
                val_fids.extend(fids[NUM_FIELDS : NUM_FIELDS * 2])
                train_fids.extend(fids[NUM_FIELDS * 2 :])
            else:
                # Too few fields: use a smaller fraction to avoid leaving train empty.
                total = len(fids)
                n_val = min(5, total // 2)
                n_test = min(5, total - n_val)
                val_fids.extend(fids[:n_val])
                test_fids.extend(fids[n_val : n_val + n_test])
                train_fids.extend(fids[n_val + n_test :])

    # Deduplicate (a field ID should belong to exactly one split).
    return (
        np.array(list(set(train_fids))),
        np.array(list(set(val_fids))),
        np.array(list(set(test_fids))),
    )


def build_train_val_test_mask(
    field_ids: npt.NDArray,
    train_fids: npt.NDArray,
    val_fids: npt.NDArray,
    test_fids: npt.NDArray,
) -> npt.NDArray:
    """Build an int8 split map: 1=train, 2=val, 3=test, 0=unassigned."""
    # Start with all zeros (background / unlabelled pixels).
    mask = np.zeros_like(field_ids, dtype=np.int8)
    # Set each pixel according to which split its field ID belongs to.
    mask[np.isin(field_ids, train_fids)] = 1
    mask[np.isin(field_ids, val_fids)] = 2
    mask[np.isin(field_ids, test_fids)] = 3
    return mask


def identify_valid_classes(labels: npt.NDArray, model_name: str) -> set[int]:
    """Return the set of class codes with at least 2 pixels."""
    # Count occurrences of every unique value in the flattened label raster.
    class_counts = Counter(labels.ravel())
    # Require at least 2 pixels so a train/test split is always possible.
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
    # XGBoost uses -1 for its internal background; all others use 0.
    if model_name == "XGBOOST":
        valid_classes.discard(-1)
    else:
        valid_classes.discard(0)
    return valid_classes


def apply_smote(
    X: npt.NDArray, y: npt.NDArray, seed: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """Apply SMOTE oversampling and return balanced (X, y)."""
    logging.info("Applying SMOTE to balance training set...")
    sm = SMOTE(random_state=seed)
    # fit_resample generates synthetic minority-class samples until all classes
    # have the same count as the majority class.
    X_bal, y_bal = sm.fit_resample(X, y)
    logging.info(f"Training samples before SMOTE: {len(y)}, after: {len(y_bal)}")
    return X_bal, y_bal


def remap_labels(
    field_id_raster: npt.NDArray,
    id_to_code: dict,
    background: int = 0,
) -> npt.NDArray:
    """
    Remap field IDs to class codes using NumPy fancy indexing (fast C-level lookup).

    Replaces the slow np.vectorize pattern from remap_senegal_labels.py.
    """
    # Build a lookup table of size (max_field_id + 1,) so that
    # lookup[field_id] gives the corresponding class code directly.
    max_id = int(np.max(field_id_raster))
    # Initialise all entries to the background value; only known IDs get overwritten.
    lookup = np.full(max_id + 1, background, dtype=np.float32)
    for fid, code in id_to_code.items():
        if 0 <= fid <= max_id:
            lookup[fid] = code
    # Integer-index the lookup table with the full raster — O(n) and fully vectorised.
    return lookup[field_id_raster.astype(np.int64)]


def mask_classes_2021(
    labels: npt.NDArray,
    field_ids: npt.NDArray,
    classes: tuple[int, ...] = (7, 8),
) -> tuple[npt.NDArray, npt.NDArray]:
    """Zero out pasture (7) and natural vegetation (8) for 2021 data."""
    # Build a boolean mask for pixels that belong to the excluded classes.
    mask = np.isin(labels, list(classes))
    # Zero out both the label and the field-ID rasters so these pixels are treated
    # as background (unlabelled) in all downstream operations.
    labels[mask] = 0
    field_ids[mask] = 0
    return labels, field_ids


def safe_vstack(arrays: list, empty_shape: tuple | None = None) -> npt.NDArray:
    # Filter out empty arrays before stacking to avoid numpy errors on zero-row inputs.
    filtered = [arr for arr in arrays if arr.size > 0]
    if filtered:
        return np.vstack(filtered)
    # Return a correctly shaped empty array so downstream code can still check .shape.
    return np.empty(empty_shape) if empty_shape is not None else np.empty((0,))


def safe_hstack(arrays: list, empty_shape: tuple | None = None) -> npt.NDArray:
    # Same pattern as safe_vstack but for 1-D label/ID arrays.
    filtered = [arr for arr in arrays if arr.size > 0]
    if filtered:
        return np.hstack(filtered)
    return (
        np.empty(empty_shape, dtype=int)
        if empty_shape is not None
        else np.empty((0,), dtype=int)
    )
