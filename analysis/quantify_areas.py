"""
analysis/quantify_areas.py — compute per-class area percentages across years and approaches.

Loads tessera, stm, and raw prediction maps for 2018/2019/2021 and outputs a CSV with the
percentage of pixels (excluding masked/background) in each land cover or crop class.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _build_class_base(classification: str) -> pd.DataFrame:
    """Build the initial class base DataFrame for the given classification type."""
    if classification == "landcover":
        base = pd.DataFrame(
            {
                "code": np.array([1, 2, 3, 4, 5, 6]),
                "label": [
                    "Built-up surface",  # 1
                    "Bare soil",         # 2
                    "Water body",        # 3
                    "Wetland",           # 4
                    "Cropland",          # 5
                    "Shrub land",        # 6
                ],
            }
        )
    elif classification == "maincrop":
        base = pd.DataFrame(
            {
                "code": np.array([0, 1, 2, 3, 4, 5]),
                "label": [
                    "Masked",    # 0 — background / non-cropland pixels
                    "Cowpea",    # 1
                    "Fallow",    # 2
                    "Groundnut", # 3
                    "Millet",    # 4
                    "Sorghum",   # 5
                ],
            }
        )
    else:
        raise ValueError(f"Unknown classification: '{classification}'")

    # Classes 6–8 may appear in 2021 maps (additional crops surveyed that year).
    # Appended here so they are present in the merge even if absent in earlier years.
    extra_classes = pd.DataFrame(
        {
            "code": [6, 7, 8],
            "label": ["Tree", "Rice", "Other"],
        }
    )
    # drop_duplicates ensures extra classes don't double-up if already in base.
    return pd.concat([base, extra_classes]).drop_duplicates("code", keep="first")


def main(
    classification: str,
    data_root: str,
    num_agg: int,
    save: bool,
) -> None:
    # Select the prediction subfolder and filename suffix based on task type.
    if classification == "landcover":
        folder = "landcoverclassification"
        classification2 = ""         # no extra suffix in landcover filenames
    else:
        folder = "cropclassification"
        classification2 = "maincrop_"  # crop maps have "maincrop_" before the agg count

    # Initialise the class lookup table; area columns are added iteratively below.
    base = _build_class_base(classification)

    print("Initial class base:")
    print(base)

    for year in [2018, 2019, 2021]:
        # 2021 landcover maps carry the "_remapped" suffix (pasture/nat-veg zeroed out).
        if year == 2021 and classification == "landcover":
            tessera_map = np.load(
                f"{data_root}/{folder}/senegal_tessera_prediction_map_whole_{year}_remapped_{classification2}{num_agg}agg.npy"
            )
            stm_map = np.load(
                f"{data_root}/{folder}/senegal_stm_prediction_map_whole_{year}_remapped_{classification2}{num_agg}agg.npy"
            )
            raw_map = np.load(
                f"{data_root}/{folder}/senegal_raw_prediction_map_whole_{year}_remapped_{classification2}{num_agg}agg.npy"
            )
            print(f"Unique values in maps for {year}:")
            print(" tessera:", np.unique(tessera_map))
            print(" stm    :", np.unique(stm_map))
            print(" raw    :", np.unique(raw_map))

            if classification == "landcover":
                # STM and raw maps are 0-indexed (argmax output); shift to 1-indexed
                # so class codes match the lookup table built above.
                stm_map = stm_map - 1
                raw_map = raw_map - 1
        else:
            # Standard path: no remapping suffix for 2018 and 2019.
            tessera_map = np.load(
                f"{data_root}/{folder}/senegal_tessera_prediction_map_whole_{year}_{classification2}{num_agg}agg.npy"
            )
            stm_map = np.load(
                f"{data_root}/{folder}/senegal_stm_prediction_map_whole_{year}_{classification2}{num_agg}agg.npy"
            )
            raw_map = np.load(
                f"{data_root}/{folder}/senegal_raw_prediction_map_whole_{year}_{classification2}{num_agg}agg.npy"
            )
            print(f"Unique values in maps for {year}:")
            print(" tessera:", np.unique(tessera_map))
            print(" stm    :", np.unique(stm_map))
            print(" raw    :", np.unique(raw_map))

            if classification == "landcover":
                stm_map = stm_map - 1
                raw_map = raw_map - 1

        labels = ["tessera", "stm", "raw"]

        for i, current_map in enumerate([tessera_map, stm_map, raw_map]):
            print(f"Processing {labels[i]} map for year {year}...")

            values, freqs = np.unique(current_map, return_counts=True)
            print(
                f"Count of masked values in {labels[i]} map for {year}: {np.sum(current_map == 0)}"
            )
            # Exclude background (code 0) from the denominator so percentages sum to 100
            # over predicted (non-masked) pixels only.
            masked = np.sum(current_map == 0)
            perc = (freqs / (current_map.size - masked)) * 100

            # Build a per-approach-year column and join into the accumulator DataFrame.
            df_counts = pd.DataFrame({"code": values, f"{labels[i]}_{year}": perc})
            base = pd.merge(base, df_counts, on="code", how="outer")

    print("\nFinal results:")
    print(base)

    if save:
        base.to_csv(
            f"{data_root}/{folder}/senegal_unique_counts.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-class area percentages across years and approaches."
    )
    parser.add_argument(
        "--classification",
        choices=["landcover", "maincrop"],
        default="maincrop",
        help="Classification type",
    )
    parser.add_argument(
        "--data_root",
        default="/maps/mcl66/senegal",
        help="Root data directory",
    )
    parser.add_argument(
        "--num_agg",
        type=int,
        default=15,
        help="Number of aggregation runs used when generating maps",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save results to CSV (use --no-save to disable)",
    )
    args = parser.parse_args()
    main(
        classification=args.classification,
        data_root=args.data_root,
        num_agg=args.num_agg,
        save=args.save,
    )
