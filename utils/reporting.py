from __future__ import annotations

import csv
import os
import statistics as stats

import numpy.typing as npt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_classification_report(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    seed: int,
    cpu_time: float,
    path: str,
) -> None:
    """Append a classification report block to a CSV file."""
    # output_dict=True returns a nested dict (label → metric → value) that pandas
    # can directly convert to a wide-format DataFrame.
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()          # rows = labels, cols = precision/recall/f1/support
        .reset_index()
        .rename(columns={"index": "label"})
    )
    # Prepend metadata columns so each row carries its seed and runtime context.
    report_df.insert(0, "cpu_time", cpu_time)
    report_df.insert(0, "seed", seed)

    # Append to an existing file rather than overwriting so that multiple seeds
    # accumulate into a single CSV for easy downstream summarization.
    if os.path.exists(path):
        existing_df = pd.read_csv(path)
        combined_df = pd.concat([existing_df, report_df], ignore_index=True)
    else:
        combined_df = report_df

    combined_df.to_csv(path, index=False, float_format="%.4f")


def save_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    class_names: list[str],
    path: str,
) -> None:
    """Save a confusion matrix to CSV."""
    # Use explicit label range to ensure rows/cols are ordered consistently even
    # if some classes are absent from y_test for this seed.
    labels = list(range(1, len(class_names) + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Wrap in a DataFrame with human-readable class names for both axes.
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(path, index=True)


def summarize_seed_results(
    report_csv_path: str,
) -> dict[str, tuple[float | None, float | None]]:
    """Read a multi-seed classification report CSV and return mean/std statistics."""
    accuracy_vals = []
    macro_vals = []
    weighted_vals = []

    # The CSV schema written by save_classification_report is:
    # seed, cpu_time, label, precision, recall, f1-score, support
    # Column indices:  0      1          2        3         4        5            6
    with open(report_csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header row
        for row in reader:
            label = row[2]   # class label or summary row name
            value = float(row[5])  # f1-score column (or accuracy for accuracy row)
            if label == "accuracy":
                accuracy_vals.append(value)
            elif label == "macro avg":
                macro_vals.append(value)
            elif label == "weighted avg":
                weighted_vals.append(value)

    def _summarize(values: list[float]) -> tuple[float | None, float | None]:
        # Return (None, None) if no data was found, so callers can handle it gracefully.
        if not values:
            return None, None
        mean = stats.mean(values)
        # stdev requires at least 2 data points; return 0 for single-seed runs.
        std = stats.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    return {
        "accuracy": _summarize(accuracy_vals),
        "macro_f1": _summarize(macro_vals),
        "weighted_f1": _summarize(weighted_vals),
    }
