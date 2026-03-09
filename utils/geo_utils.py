from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine


def convert_npy_to_tiff(
    data: npt.NDArray,
    ref_tiff_path: str,
    output_path: str,
    downsample_rate: int = 1,
) -> None:
    """Convert a numpy array to a georeferenced GeoTIFF using a reference raster."""
    if not os.path.exists(ref_tiff_path):
        raise FileNotFoundError(f"Reference TIFF not found: {ref_tiff_path}")

    if data.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    # int64 is not a valid GeoTIFF dtype; downcast to uint8 for file compatibility.
    if data.dtype == np.int64:
        data = data.astype(np.uint8)

    # Normalise to (H, W, C) shape — C=1 for single-band (classification) maps.
    if data.ndim == 2:
        H, W = data.shape
        C = 1
    else:
        H, W, C = data.shape

    # Optional spatial downsampling: average each (downsample_rate × downsample_rate)
    # pixel block into a single output pixel.  Useful for previewing large rasters.
    if downsample_rate > 1:
        new_H = H // downsample_rate
        new_W = W // downsample_rate
        # Pre-allocate the downsampled output array with the same dtype.
        if C == 1:
            downsampled = np.zeros((new_H, new_W), dtype=data.dtype)
        else:
            downsampled = np.zeros((new_H, new_W, C), dtype=data.dtype)

        for i in range(new_H):
            for j in range(new_W):
                # Compute the pixel block boundaries, clamping to array edges.
                i_start = i * downsample_rate
                i_end = min((i + 1) * downsample_rate, H)
                j_start = j * downsample_rate
                j_end = min((j + 1) * downsample_rate, W)
                if C == 1:
                    block = data[i_start:i_end, j_start:j_end]
                    downsampled[i, j] = np.mean(block)
                else:
                    block = data[i_start:i_end, j_start:j_end, :]
                    # Average each band independently across the spatial block.
                    downsampled[i, j, :] = np.mean(block, axis=(0, 1))

        data = downsampled
        H, W = new_H, new_W

    # Read CRS and transform from the reference raster — these define the spatial
    # extent and projection of our output file.
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        transform = ref.transform
        crs = ref.crs

        # When downsampling, the pixel size grows by downsample_rate in both x and y.
        # Update the Affine transform so geolocation stays correct.
        if downsample_rate > 1:
            transform = Affine(
                transform.a * downsample_rate,  # pixel width
                transform.b,                    # row rotation (usually 0)
                transform.c,                    # x origin (upper-left corner)
                transform.d,                    # column rotation (usually 0)
                transform.e * downsample_rate,  # pixel height (negative for north-up)
                transform.f,                    # y origin (upper-left corner)
            )

    # Build output metadata by overriding only the fields that differ from the reference.
    new_meta = ref_meta.copy()
    new_meta.update(
        {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": C,         # number of bands
            "dtype": data.dtype,
            "transform": transform,
            "crs": crs,
        }
    )

    # Write each band individually; rasterio band indices are 1-based.
    with rasterio.open(output_path, "w", **new_meta) as dst:
        if C == 1:
            dst.write(data, 1)  # single-band write
        else:
            for i in range(C):
                dst.write(data[:, :, i], i + 1)

    print(f"Saved GeoTIFF to {output_path}")
    # Report the effective ground resolution so the caller can sanity-check.
    print(f"Resolution: {10 * downsample_rate} m")


def clip_raster_to_bbox(
    raster_path: str,
    shapefile_path: str,
    out_tiff: str,
    out_npy: str,
) -> None:
    """Clip a raster to a shapefile bounding box and save as TIFF and NPY."""
    # Load the bounding polygon from the shapefile.
    bbox = gpd.read_file(shapefile_path)

    with rasterio.open(raster_path) as src:
        # Reproject bbox to the raster's CRS if they differ, so masking works correctly.
        if bbox.crs != src.crs:
            bbox = bbox.to_crs(src.crs)

        # Mask (clip) the raster to the shapefile geometry; nodata=0 fills outside pixels.
        clipped_array, clipped_transform = rasterio_mask(
            src, bbox.geometry, crop=True, nodata=0
        )
        # Update the metadata to reflect the new spatial extent.
        clipped_meta = src.meta.copy()
        clipped_meta.update(
            {
                "height": clipped_array.shape[1],
                "width": clipped_array.shape[2],
                "transform": clipped_transform,
            }
        )

    # Save as TIFF (georeferenced) for GIS tools.
    with rasterio.open(out_tiff, "w", **clipped_meta) as dest:
        dest.write(clipped_array)

    # Save as NPY (raw array) for fast numpy loading in subsequent pipeline steps.
    np.save(out_npy, clipped_array)
    print(f"Saved clipped TIF: {out_tiff}")
    print(f"Saved clipped NPY: {out_npy} with shape {clipped_array.shape}")


def get_chunk_grid(H: int, W: int, chunk_size: int) -> list[tuple[int, int, int, int]]:
    """Return list of (h0, h1, w0, w1) tuples covering an H×W grid."""
    # Each tuple defines a spatial tile: rows [h0, h1) × cols [w0, w1).
    # min(..., H/W) handles the last tile in each row/column, which may be smaller
    # than chunk_size if H or W is not divisible by chunk_size.
    return [
        (h, min(h + chunk_size, H), w, min(w + chunk_size, W))
        for h in range(0, H, chunk_size)
        for w in range(0, W, chunk_size)
    ]


def save_prediction(pred_map: npt.NDArray, out_prefix: str, cfg: dict) -> None:
    """Save a prediction map as both .npy and .tiff."""
    # .npy for fast reloading by downstream Python scripts.
    npy_path = out_prefix + ".npy"
    tiff_path = out_prefix + ".tiff"
    np.save(npy_path, pred_map)
    # Use the reference TIFF from config to georeference the output correctly.
    ref_tiff = cfg["paths"]["ref_tiff"]
    convert_npy_to_tiff(pred_map, ref_tiff, tiff_path)
