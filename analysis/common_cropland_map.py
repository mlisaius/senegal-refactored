"""
analysis/common_cropland_map.py — identify pixels classified as cropland across all years.

Loads landcover prediction maps for 2018, 2019, and 2021, remaps to binary cropland/non-cropland,
and saves pixels that are cropland in all three years as the "stable cropland" combo map.
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
        5: 1,  # Cropland — the only class we keep as 1
        6: 0,  # Shrub land
        7: 0,  # Pasture
    }
    max_key = max(remapping)
    # Build a lookup table indexed by class code for O(n) remapping.
    lut = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in remapping.items():
        lut[k] = v
    # clip handles any values above max_key (e.g. class 8 in remapped 2021 maps).
    return lut[np.clip(arr.astype(np.int64), 0, max_key)]


def main(
    approach: str,
    map_root: str,
    ref_tiff: str,
    num_agg: int,
    out_root: str,
) -> None:
    # Load the whole-map prediction arrays for each year from disk.
    map_2018 = np.load(
        f"{map_root}/senegal_{approach}_prediction_map_whole_2018_{num_agg}agg.npy"
    )
    map_2019 = np.load(
        f"{map_root}/senegal_{approach}_prediction_map_whole_2019_{num_agg}agg.npy"
    )
    # 2021 maps carry the "_remapped" suffix because pasture/natural-veg were zeroed out.
    map_2021 = np.load(
        f"{map_root}/senegal_{approach}_prediction_map_whole_2021_remapped_{num_agg}agg.npy"
    )

    print(f"min and max values in 2018 map: {np.min(map_2018)}, {np.max(map_2018)}")
    print(f"min and max values in 2019 map: {np.min(map_2019)}, {np.max(map_2019)}")
    print(f"min and max values in 2021 map: {np.min(map_2021)}, {np.max(map_2021)}")

    # Tessera predictions are 0-indexed (classes 0–6); shift to 1-indexed (1–7)
    # so the remapping table aligns correctly with the standard class codes.
    # STM maps for 2018 and 2019 also need this +1 shift.
    if approach == "tessera":
        map_2021 = map_2021 + 1
        map_2019 = map_2019 + 1
        map_2018 = map_2018 + 1
    elif approach == "stm":
        map_2018 = map_2018 + 1
        map_2019 = map_2019 + 1

    # Convert each year's landcover map to binary cropland (1) / non-cropland (0).
    map_2018_remapped = _remap_to_cropland(map_2018)
    map_2019_remapped = _remap_to_cropland(map_2019)
    map_2021_remapped = _remap_to_cropland(map_2021)

    # "Stable cropland": a pixel is included only if it was classified as cropland
    # in all three years.  Sum == 3 iff all three binary maps agree on cropland.
    sum_maps = map_2018_remapped + map_2019_remapped + map_2021_remapped
    combo = np.where(sum_maps == 3, 1, 0).astype(np.float64)

    print(np.unique(combo, return_counts=True))
    vals, counts = np.unique(combo, return_counts=True)
    class1_pct = counts[vals == 1] / combo.size
    print(f"Total % of landscape that is stable cropland: {class1_pct[0]*100:.2f}%")

    # Save as NPY for downstream use (e.g. maincrop mask in classify.py).
    np.save(f"{out_root}/senegal_{approach}_croplandcombo_map.npy", combo)

    # Also save as GeoTIFF for GIS inspection.
    output_path = f"{out_root}/senegal_{approach}_croplandcombo_map_new.tiff"
    convert_npy_to_tiff(combo, ref_tiff, output_path, downsample_rate=1)
    print(f"Saved {approach} cropland combo map.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify pixels classified as cropland across all years."
    )
    parser.add_argument(
        "--approach",
        choices=["tessera", "stm", "raw", "specmat"],
        default="specmat",
        help="Feature approach used for prediction maps",
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
        "--num_agg",
        type=int,
        default=15,
        help="Number of aggregation runs used when generating maps",
    )
    parser.add_argument(
        "--out_root",
        default="/maps/mcl66/senegal/landcoverclassification",
        help="Output folder for cropland combo map files",
    )
    args = parser.parse_args()
    main(
        approach=args.approach,
        map_root=args.map_root,
        ref_tiff=args.ref_tiff,
        num_agg=args.num_agg,
        out_root=args.out_root,
    )
