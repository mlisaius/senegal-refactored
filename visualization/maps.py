"""
visualization/maps.py — merged from visualize_maps_cropcover.py and visualize_maps_landcover.py.

Provides plot_classification_map() and predefined class palettes for both
landcover and crop-type outputs.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from utils.geo_utils import convert_npy_to_tiff

# ---------------------------------------------------------------------------
# Core plotting function (identical signature in both original files)
# ---------------------------------------------------------------------------


def plot_classification_map(
    data: npt.NDArray,
    class_names: list[str] | None = None,
    class_colors: list[str] | None = None,
    save_path: str = "classification_map.png",
    title: str = "Classification Map",
) -> None:
    """
    Visualize a 2-D classification map with per-class colours.

    Parameters
    ----------
    data : 2-D numpy array of integer class codes
    class_names : list of str, one per unique class in sorted order
    class_colors : list of colour strings, same length as class_names
    save_path : output PNG path
    title : figure title
    """
    # High DPI and large figure size for publication-quality output.
    plt.figure(figsize=(12, 10), dpi=300)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.linewidth": 1.5,
        }
    )

    # Determine which class codes are actually present in the data.
    unique_classes = sorted(np.unique(data))

    if class_colors:
        if len(class_colors) != len(unique_classes):
            raise ValueError(
                f"class_colors length ({len(class_colors)}) must match "
                f"number of unique classes ({len(unique_classes)})"
            )
        # ListedColormap maps each discrete class code to its assigned colour.
        cmap = mcolors.ListedColormap(class_colors)
        # BoundaryNorm ensures each class code maps to exactly one colour band.
        norm = mcolors.BoundaryNorm(unique_classes + [max(unique_classes) + 1], cmap.N)
        plt.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        # Fall back to matplotlib's tab20 palette when no colours are provided.
        cmap = plt.cm.get_cmap("tab20", len(unique_classes))
        plt.imshow(data, cmap=cmap, interpolation="nearest")

    # Build legend patches manually so class names appear regardless of imshow internals.
    legend_patches = []
    for i, cls in enumerate(unique_classes):
        color = class_colors[i] if class_colors else cmap(i)
        label = (
            class_names[i] if (class_names and i < len(class_names)) else f"Class {cls}"
        )
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # Place legend outside the map axes so it doesn't obscure the data.
    plt.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Classes",
        title_fontsize=13,
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    # dpi=600 for the saved file gives extra sharpness for print / poster use.
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()  # release memory — important when processing many maps in a loop
    print(f"Saved classification map PNG to {save_path}")


# ---------------------------------------------------------------------------
# Predefined class palettes
# ---------------------------------------------------------------------------

# Landcover palette: 6 classes, colours chosen for contrast and interpretability.
LANDCOVER_CLASS_NAMES: list[str] = [
    "Built-up surface",  # 1
    "Bare soil",         # 2
    "Water body",        # 3
    "Wetland",           # 4
    "Cropland",          # 5
    "Shrub land",        # 6
]
LANDCOVER_CLASS_COLORS: list[str] = [
    "black",    # Built-up surface
    "#9B4F3C",  # Bare soil
    "#35B6F6",  # Water body
    "pink",     # Wetland
    "green",    # Cropland
    "#0DE914",  # Shrub land
]

# Default crop palette for 2018 / 2019 (5 crop classes + Masked background).
CROP_CLASS_NAMES_DEFAULT: list[str] = [
    "Masked",    # 0
    "Cowpea",    # 1
    "Fallow",    # 2
    "Groundnut", # 3
    "Millet",    # 4
    "Sorghum",   # 5
]
CROP_CLASS_COLORS_DEFAULT: list[str] = [
    "black",    # Masked
    "orange",   # Cowpea
    "#3B230B",  # Fallow
    "pink",     # Groundnut
    "green",    # Millet
    "yellow",   # Sorghum
]

# Extended crop palette for 2021 (3 additional classes: Tree, Rice, Other).
CROP_CLASS_NAMES_2021: list[str] = [
    "Masked",    # 0
    "Cowpea",    # 1
    "Fallow",    # 2
    "Groundnut", # 3
    "Millet",    # 4
    "Sorghum",   # 5
    "Tree",      # 6
    "Rice",      # 7
    "Other",     # 8
]
CROP_CLASS_COLORS_2021: list[str] = [
    "black",    # Masked
    "orange",   # Cowpea
    "#3B230B",  # Fallow
    "pink",     # Groundnut
    "green",    # Millet
    "yellow",   # Sorghum
    "#90EE90",  # Tree
    "purple",   # Rice
    "red",      # Other
]


# ---------------------------------------------------------------------------
# Convenience entry point (mirrors original script usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualise a classification map NPY file"
    )
    parser.add_argument("npy_file", help="Path to .npy prediction map")
    parser.add_argument(
        "--type",
        choices=["landcover", "cropcover"],
        default="landcover",
        help="Map type for default colour scheme",
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument(
        "--ref_tiff",
        default=(
            "/maps/mcl66/senegal/representations/"
            "2018_representation_map_10m_utm28n_scales_clipped.tiff"
        ),
    )
    parser.add_argument("--method", default="specmat")
    args = parser.parse_args()

    data = np.load(args.npy_file)
    print(
        f"Loaded {args.npy_file} with shape {data.shape} and unique classes: {np.unique(data)}"
    )

    # Select the appropriate name/colour palette for the requested map type and year.
    if args.type == "landcover":
        names = LANDCOVER_CLASS_NAMES
        colors = LANDCOVER_CLASS_COLORS
    else:
        # 2021 crop maps have extra classes; use the extended palette for that year.
        names = CROP_CLASS_NAMES_2021 if args.year == 2021 else CROP_CLASS_NAMES_DEFAULT
        colors = (
            CROP_CLASS_COLORS_2021 if args.year == 2021 else CROP_CLASS_COLORS_DEFAULT
        )

    # Derive output paths from the input filename by replacing the extension.
    png_out = args.npy_file.replace(".npy", ".png")
    title = f"Senegal Aggregate {args.method.capitalize()} Classification Map for {args.year}"
    plot_classification_map(
        data, class_names=names, class_colors=colors, save_path=png_out, title=title
    )

    # Also export as GeoTIFF for use in GIS software.
    tiff_out = args.npy_file.replace(".npy", ".tif")
    convert_npy_to_tiff(data, args.ref_tiff, tiff_out, downsample_rate=1)
