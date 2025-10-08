
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from affine import Affine

# -----------------------------
# Plotting function with specific colors
# -----------------------------
def plot_classification_map(data, class_names=None, class_colors=None, save_path="classification_map.png", title="Classification Map"):
    """
    Visualize a classification map with specific class colors.
    - data: 2D numpy array of class codes
    - class_names: list of strings for class labels, order must match sorted(unique_classes)
    - class_colors: list of colors (hex or named) matching each class
    """
    plt.figure(figsize=(12, 10), dpi=300)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.linewidth': 1.5
    })

    unique_classes = sorted(np.unique(data))
    
    if class_colors:
        if len(class_colors) != len(unique_classes):
            raise ValueError("class_colors length must match number of unique classes")
        cmap = mcolors.ListedColormap(class_colors)
        norm = mcolors.BoundaryNorm(unique_classes + [max(unique_classes)+1], cmap.N)
        im = plt.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        cmap = plt.cm.get_cmap("tab20", len(unique_classes))
        im = plt.imshow(data, cmap=cmap, interpolation="nearest")

    # Legend
    legend_patches = []
    for i, cls in enumerate(unique_classes):
        color = class_colors[i] if class_colors else cmap(i)
        label = class_names[i] if class_names and i < len(class_names) else f"Class {cls}"
        legend_patches.append(mpatches.Patch(color=color, label=label))
    
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
               loc="upper left", fontsize=12, frameon=True, fancybox=True,
               shadow=True, title="Classes", title_fontsize=13)
    
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved classification map PNG to {save_path}")


# -----------------------------
# NPY to GeoTIFF
# -----------------------------
def convert_npy_to_tiff(data, ref_tiff_path, output_path, downsample_rate=1):
    """Convert .npy classification map to GeoTIFF using reference raster."""
    if data.dtype == np.int64:
        print("Converting int64 data to uint8...")
        data = data.astype(np.uint8)

    if data.ndim == 2:
        H, W = data.shape
        C = 1
    else:
        H, W, C = data.shape

    if downsample_rate > 1:
        new_H = H // downsample_rate
        new_W = W // downsample_rate
        downsampled_data = np.zeros((new_H, new_W, C), dtype=data.dtype)

        for i in range(new_H):
            for j in range(new_W):
                i_start, i_end = i * downsample_rate, min((i + 1) * downsample_rate, H)
                j_start, j_end = j * downsample_rate, min((j + 1) * downsample_rate, W)
                block = data[i_start:i_end, j_start:j_end, :] if C > 1 else data[i_start:i_end, j_start:j_end]
                downsampled_data[i, j] = np.mean(block, axis=(0, 1)).astype(data.dtype)

        data = downsampled_data
        H, W = new_H, new_W

    # Reference metadata
    with rasterio.open(ref_tiff_path) as ref:
        transform = ref.transform
        crs = ref.crs
        ref_meta = ref.meta.copy()

        if downsample_rate > 1:
            transform = Affine(
                transform.a * downsample_rate, transform.b, transform.c,
                transform.d, transform.e * downsample_rate, transform.f
            )

    # Update metadata
    new_meta = ref_meta.copy()
    new_meta.update({
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": C,
        "dtype": data.dtype,
        "transform": transform,
        "crs": crs
    })

    # Write TIFF
    with rasterio.open(output_path, "w", **new_meta) as dst:
        if C == 1:
            dst.write(data, 1)
        else:
            for i in range(C):
                dst.write(data[:, :, i], i + 1)

    print(f"✅ Saved GeoTIFF to {output_path}")


# -----------------------------
# Define parameters and load data
# -----------------------------
year = 2021
method = "specmat" #"efm" #"stm" # or "stm" or "raw"


if year == 2021:
    #npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_{method}_prediction_map_whole_{year}_remapped_15agg.npy"
    npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_{method}_prediction_map_whole_{year}_landcover_remapped_15agg.npy"
    npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_specmat_prediction_map_whole_{year}_remapped_15agg.npy"
else: 
    #npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_{method}_prediction_map_whole_{year}_15agg.npy"
    npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_{method}_prediction_map_whole_{year}_landcover_15agg.npy"
    npy_file = f"/maps/mcl66/senegal/landcoverclassification/senegal_specmat_prediction_map_whole_{year}_15agg.npy"
    
data = np.load(npy_file)
print(f"Loaded {npy_file} with shape {data.shape} and unique classes: {np.unique(data)}")

# Define class names and colors
class_names = [
        "Built-up surface", # 1,
        "Bare soil", # 2,
        "Water body", # 3,
        "Wetland", # 4,
        "Cropland", # 5,
        "Shrub land", # 6,
    ]

class_colors = [
        "black",      # built up - black
        "#9B4F3C",       # bare soil
        "#35B6F6",    # water body
        "pink",    # wetland
        "green",      # cropland
        "#0DE914",     # shrubland
    ]

# Save PNG
png_out = npy_file.replace(".npy", ".png")
print(f"Saving PNG to {png_out}...")
title = f"Senegal Aggregate {method.capitalize()} Classification Map for {year}"
plot_classification_map(data, class_names=class_names, class_colors=class_colors, save_path=png_out, title=title)

# Optional: save GeoTIFF (requires reference raster)
ref_tiff_path = "/maps/mcl66/senegal/representations_deprecated/2018_representation_map_10m_utm28n_scales_clipped.tiff"
tiff_out = npy_file.replace(".npy", ".tif")
convert_npy_to_tiff(data, ref_tiff_path, tiff_out, downsample_rate=1)
