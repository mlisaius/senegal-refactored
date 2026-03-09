"""
analysis/average_agg_accuracies.py — summarize aggregated classification report CSVs.

Reads multi-seed classification report CSVs produced by classify.py (aggregation mode)
and prints mean ± std for accuracy, macro-avg F1, and weighted-avg F1 per approach/year.
"""

from __future__ import annotations

import argparse

from utils.reporting import summarize_seed_results


def main(
    years: list[int],
    approaches: list[str],
    classification: str,
    report_root: str,
) -> None:
    for year in years:
        # Build the expected file path for each (year, approach) combination.
        # The naming convention matches what classify.py writes in aggregation mode.
        paths = {
            approach: (
                f"{report_root}/senegal_{approach}_classification_report"
                f"_{year}_agg_{classification}.csv"
            )
            for approach in approaches
        }

        for approach, path in paths.items():
            # summarize_seed_results reads the CSV and computes mean/std across seeds.
            results = summarize_seed_results(path)
            acc_mean, acc_std = results["accuracy"]
            macro_mean, macro_std = results["macro_f1"]
            wt_mean, wt_std = results["weighted_f1"]

            print(f"[{year}] {approach}")
            print(f"  Accuracy       → mean: {acc_mean:.4f}, std: {acc_std:.4f}")
            print(f"  Macro avg F1   → mean: {macro_mean:.4f}, std: {macro_std:.4f}")
            print(f"  Weighted avg F1 → mean: {wt_mean:.4f}, std: {wt_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize aggregated classification report CSVs."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2021],
        help="Years to process",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["tessera", "raw", "efm", "specmat"],
        help="Approaches to process",
    )
    parser.add_argument(
        "--classification",
        choices=["landcover", "maincrop"],
        default="maincrop",
        help="Classification type",
    )
    parser.add_argument(
        "--report_root",
        default="/maps/mcl66/senegal/classification_reports",
        help="Root folder of classification report CSVs",
    )
    args = parser.parse_args()
    main(
        years=args.years,
        approaches=args.approaches,
        classification=args.classification,
        report_root=args.report_root,
    )
