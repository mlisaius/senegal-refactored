"""
analysis/average_single_accuracies.py — summarize single-model classification report CSVs.

Reads per-seed classification report CSVs from a folder (single-model, non-aggregation runs)
and prints mean ± std for accuracy, macro-avg F1, and weighted-avg F1 per matched file.
"""

from __future__ import annotations

import argparse
import glob
import os

from utils.reporting import summarize_seed_results


def main(
    years: list[int],
    classification: str,
    csv_folder: str,
) -> None:
    for year in years:
        # glob matches all CSVs in the folder that contain the year and classification
        # in their filename — handles multiple approaches and model names automatically.
        for path in glob.glob(
            os.path.join(csv_folder, f"*{year}*{classification}.csv")
        ):
            # summarize_seed_results reads all seed rows and returns mean/std per metric.
            results = summarize_seed_results(path)
            acc_mean, acc_std = results["accuracy"]
            macro_mean, macro_std = results["macro_f1"]
            wt_mean, wt_std = results["weighted_f1"]

            print(f"File: {path}")
            print(f"  Accuracy       → mean: {acc_mean:.4f}, std: {acc_std:.4f}")
            print(f"  Macro avg F1   → mean: {macro_mean:.4f}, std: {macro_std:.4f}")
            print(f"  Weighted avg F1 → mean: {wt_mean:.4f}, std: {wt_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize single-model classification report CSVs."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018],
        help="Years to process",
    )
    parser.add_argument(
        "--classification",
        choices=["landcover", "maincrop"],
        default="landcover",
        help="Classification type",
    )
    parser.add_argument(
        "--csv_folder",
        default="/maps/mcl66/senegal/classification_reports_singlemodel/",
        help="Folder containing per-seed classification report CSVs",
    )
    args = parser.parse_args()
    main(
        years=args.years,
        classification=args.classification,
        csv_folder=args.csv_folder,
    )
