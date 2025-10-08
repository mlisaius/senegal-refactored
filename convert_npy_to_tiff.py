import os
import numpy as np
import rasterio
from rasterio.transform import Affine

year = 2020  # Define the year of interest

def convert_npy_to_tiff(npy_path, ref_tiff_path, out_dir, downsample_rate=1):
    # Load npy data
    data = np.load(npy_path)

    if data.ndim == 2:
        H, W = data.shape
        C = 1
    elif data.ndim == 3:
        H, W, C = data.shape
    else:
        raise ValueError(f"Unexpected array shape {data.shape}, expected (H,W) or (H,W,C)")

    # Downsampling if requested
    if downsample_rate > 1:
        new_H = H // downsample_rate
        new_W = W // downsample_rate

        if C == 1:
            downsampled_data = np.zeros((new_H, new_W), dtype=data.dtype)
        else:
            downsampled_data = np.zeros((new_H, new_W, C), dtype=data.dtype)

        for i in range(new_H):
            for j in range(new_W):
                i_start, i_end = i * downsample_rate, min((i + 1) * downsample_rate, H)
                j_start, j_end = j * downsample_rate, min((j + 1) * downsample_rate, W)

                if C == 1:
                    block = data[i_start:i_end, j_start:j_end]
                    downsampled_data[i, j] = np.mean(block)
                else:
                    block = data[i_start:i_end, j_start:j_end, :]
                    downsampled_data[i, j, :] = np.mean(block, axis=(0, 1))

        data = downsampled_data
        H, W = new_H, new_W

    # Read reference TIFF metadata
    with rasterio.open(ref_tiff_path) as ref:
        ref_meta = ref.meta.copy()
        transform = ref.transform
        crs = ref.crs

        if downsample_rate > 1:
            transform = Affine(
                transform.a * downsample_rate, transform.b, transform.c,
                transform.d, transform.e * downsample_rate, transform.f
            )

    # Update metadata
    new_meta = ref_meta.copy()
    new_meta.update({
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': C,
        'dtype': data.dtype,
        'transform': transform
    })

    # Output path
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    out_path = os.path.join(out_dir, f"{base_name}.tif")

    # Write output GeoTIFF
    with rasterio.open(out_path, 'w', **new_meta) as dst:
        if C == 1:
            dst.write(data, 1)  # Write single band
        else:
            for i in range(C):
                dst.write(data[:, :, i], i + 1)  # Write each band
                #print(f"Band {i + 1} writing complete")

    print(f"Output file saved as: {out_path}")
    print(f"Resolution: Original 10m, after downsampling {10 * downsample_rate}m")

if __name__ == "__main__":
    rep_path = f"/maps/mcl66/senegal/representations/{year}_representation_map_10m_utm28n_128bands.npy"
    scale_path = f"/maps/mcl66/senegal/representations/{year}_representation_map_10m_utm28n_scales.npy"

    ref_tiff_path = f"/maps/mcl66/senegal/representations_deprecated/2018_representation_map_10m_utm28n_128bands.tiff"
    out_dir = "/maps/mcl66/senegal/representations"
    downsample_rate = 1

    convert_npy_to_tiff(rep_path, ref_tiff_path, out_dir, downsample_rate)
    convert_npy_to_tiff(scale_path, ref_tiff_path, out_dir, downsample_rate)
