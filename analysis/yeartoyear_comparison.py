"""
analysis/yeartoyear_comparison.py — compute year-on-year cropland change between two prediction maps.

Remaps both maps to binary cropland/non-cropland, subtracts them to get a change map
(-1 = crop loss, 0 = no change, 1 = crop gain), and optionally saves as NPY + GeoTIFF.
"""

from __future__ import annotations

import argparse

import numpy as np
import numpy.typing as npt
from utils.geo_utils import convert_npy_to_tiff


def _remap_to_cropland(arr: npt.NDArray) -> npt.NDArray:
    """Remap a landcover array to binary cropland (class 5 → 1, all others → 0)."""
    remapping = {
        1: 0,  # Built-up surface
        2: 0,  # Bare soil
        3: 0,  # Water body
        4: 0,  # Wetland
        5: 1,  # Cropland
        6: 0,  # Shrub land
        7: 0,  # Pasture
    }
    max_key = max(remapping)
    # Lookup table lets us remap the entire array in one vectorised step.
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in remapping.items():
        lut[k] = v
    # clip prevents IndexError if any values exceed max_key.
    return lut[np.clip(arr.astype(np.int64), 0, max_key)]


def main(
    data_source: str,
    year1: int,
    year2: int,
    num_agg: int,
    map_root: str,
    ref_tiff: str,
    save: bool,
) -> None:
    # Load the appropriate prediction maps based on the data source argument.
    if data_source == "same":
        # "same" is a special case: year1 uses tessera output (0-indexed, needs +1),
        # while year2 uses the raw output (already 1-indexed).
        year1_pred = (
            np.load(
                f"{map_root}/senegal_tessera_prediction_map_whole_{year1}_{num_agg}agg.npy"
            )
            + 1  # shift tessera 0-based codes to 1-based
        )
        year2_pred = np.load(
            f"{map_root}/senegal_raw_prediction_map_whole_{year2}_{num_agg}agg.npy"
        )
    else:
        year1_pred = np.load(
            f"{map_root}/senegal_{data_source}_prediction_map_whole_{year1}_{num_agg}agg.npy"
        )
        # 2021 maps have the "_remapped" suffix because pasture/nat-veg were excluded.
        if year2 == 2021:
            year2_pred = np.load(
                f"{map_root}/senegal_{data_source}_prediction_map_whole_{year2}_remapped_{num_agg}agg.npy"
            )
        else:
            year2_pred = np.load(
                f"{map_root}/senegal_{data_source}_prediction_map_whole_{year2}_{num_agg}agg.npy"
            )

    # Reduce both maps to binary cropland presence/absence.
    year1_pred_remapped = _remap_to_cropland(year1_pred)
    year2_pred_remapped = _remap_to_cropland(year2_pred)

    # Pixel-wise subtraction: +1 = new cropland, -1 = lost cropland, 0 = no change.
    change_map = year2_pred_remapped - year1_pred_remapped
    decrease_percentage = np.sum(change_map == -1) / change_map.size * 100
    increase_percentage = np.sum(change_map == 1) / change_map.size * 100
    print(f"Percentage of decrease from {year1} to {year2}: {decrease_percentage:.2f}%")
    print(f"Percentage of increase from {year1} to {year2}: {increase_percentage:.2f}%")

    print(f"min value in change map: {np.min(change_map)}")
    print(f"max value in change map: {np.max(change_map)}")

    change_percentage = np.sum(change_map != 0) / change_map.size * 100
    print(f"Percentage of change from {year1} to {year2}: {change_percentage:.2f}%")

    # Shift values from [-1, 0, 1] → [0, 1, 2] so the map can be saved as uint8
    # and visualised with a straightforward colour scheme (0=loss, 1=stable, 2=gain).
    change_map += 1

    if save:
        npy_out = f"{map_root}/senegal_{data_source}_change_map_{year1}_{year2}.npy"
        np.save(npy_out, change_map)

        tiff_out = f"{map_root}/senegal_{data_source}_change_map_{year1}_{year2}.tiff"
        convert_npy_to_tiff(change_map, ref_tiff, tiff_out, downsample_rate=1)
        print(f"Change map saved to {tiff_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute year-on-year cropland change between two prediction maps."
    )
    parser.add_argument(
        "--data_source",
        choices=["raw", "tessera", "same", "stm"],
        default="raw",
        help="Data source / approach for prediction maps",
    )
    parser.add_argument(
        "--year1",
        type=int,
        default=2019,
        help="First (earlier) year",
    )
    parser.add_argument(
        "--year2",
        type=int,
        default=2021,
        help="Second (later) year",
    )
    parser.add_argument(
        "--num_agg",
        type=int,
        default=15,
        help="Number of aggregation runs used when generating maps",
    )
    parser.add_argument(
        "--map_root",
        default="/maps/mcl66/senegal/landcoverclassification",
        help="Root folder containing prediction map NPY files",
    )
    parser.add_argument(
        "--ref_tiff",
        default=(
            "/maps/mcl66/senegal/representations/"
            "2018_representation_map_10m_utm28n_scales_clipped.tiff"
        ),
        help="Reference GeoTIFF for georeferencing output",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save change map as NPY and GeoTIFF",
    )
    args = parser.parse_args()
    main(
        data_source=args.data_source,
        year1=args.year1,
        year2=args.year2,
        num_agg=args.num_agg,
        map_root=args.map_root,
        ref_tiff=args.ref_tiff,
        save=args.save,
    )
