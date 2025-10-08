import numpy as np
import rasterio

approach = "specmat"  # Options: "tessera", "stm", "raw"

map_2018 = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_{approach}_prediction_map_whole_2018_15agg.npy")
map_2019 = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_{approach}_prediction_map_whole_2019_15agg.npy")
map_2021 = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_{approach}_prediction_map_whole_2021_remapped_15agg.npy")

print(f"min and max values in 2018 map: {np.min(map_2018)}, {np.max(map_2018)}")
print(f"min and max values in 2019 map: {np.min(map_2019)}, {np.max(map_2019)}")
print(f"min and max values in 2021 map: {np.min(map_2021)}, {np.max(map_2021)}")

if approach == "tessera":
    map_2021 = map_2021 + 1  # Adjust for remapping in 2021 
    map_2019 = map_2019 + 1  # Adjust for remapping in 2019
    map_2018 = map_2018 + 1  # Adjust for remapping in 2018
elif approach == "stm":
    map_2018 = map_2018 + 1  # Adjust for remapping in 2018
    map_2019 = map_2019 + 1  # Adjust for remapping in 2019
    #map_2021 = map_2021 - 1

remapping = {
    1: 0,  # Built-up surface
    2: 0,  # Bare soil
    3: 0,  # Water body
    4: 0,  # Wetland
    5: 1,  # Cropland
    6: 0,  # Shrub land
    7: 0   # Pasture
}

map_2018_remapped = np.vectorize(lambda x: remapping.get(x, 0))(map_2018)
map_2019_remapped = np.vectorize(lambda x: remapping.get(x, 0))(map_2019)
map_2021_remapped = np.vectorize(lambda x: remapping.get(x, 0))(map_2021)

# sum maps across years
sum_maps = map_2018_remapped + map_2019_remapped + map_2021_remapped

# 1 where all three maps are cropland, else 0
combo = np.where(sum_maps == 3, 1, 0).astype(np.float64)

def convert_npy_to_tiff(npy, ref_tiff_path, output_path, downsample_rate=1):
            # Load npy data, assuming shape (H, W) or (H, W, C)
            data = npy
            
            if data.dtype == np.int64:
                # Convert int64 to uint8 if necessary
                print("Converting int64 data to uint8...")
                data = data.astype(np.uint8)    
                
            if data.ndim == 2:
                H, W = data.shape
                C = 1
            else:
                H, W, C = data.shape

            # Downsample if needed
            if downsample_rate > 1:
                new_H = H // downsample_rate
                new_W = W // downsample_rate
                downsampled_data = np.zeros((new_H, new_W, C), dtype=data.dtype)

                for i in range(new_H):
                    for j in range(new_W):
                        i_start = i * downsample_rate
                        i_end = min((i + 1) * downsample_rate, H)
                        j_start = j * downsample_rate
                        j_end = min((j + 1) * downsample_rate, W)
                        block = data[i_start:i_end, j_start:j_end, :] if C > 1 else data[i_start:i_end, j_start:j_end]
                        downsampled_data[i, j] = np.mean(block, axis=(0, 1)).astype(data.dtype)

                data = downsampled_data
                H, W = new_H, new_W

            # Reference geospatial info from a valid GeoTIFF
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
                'driver': 'GTiff',
                'height': H,
                'width': W,
                'count': C,
                'dtype': data.dtype,
                'transform': transform,
                'crs': crs
            })

            # Write TIFF
            with rasterio.open(output_path, 'w', **new_meta) as dst:
                if C == 1:
                    dst.write(data, 1)
                else:
                    for i in range(C):
                        dst.write(data[:, :, i], i + 1)

            print(f"✅ Saved GeoTIFF to {output_path}")
            print(f"Resolution: {10 * downsample_rate} m")


print(np.unique(combo, return_counts=True))
vals, counts = np.unique(combo, return_counts=True)
class1_pct = counts[vals == 1] / combo.size
print(f"Total % of landscape that is stable cropland: {class1_pct[0]*100:.2f}%")

np.save(f"/maps/mcl66/senegal/landcoverclassification/senegal_{approach}_croplandcombo_map.npy", combo)
# Convert the prediction map to TIFF format
output_path = f"/maps/mcl66/senegal/landcoverclassification/senegal_{approach}_croplandcombo_map_new.tiff"
ref_tiff_path = f"/maps/mcl66/senegal/representations_deprecated/2018_representation_map_10m_utm28n_scales_clipped.tiff"
convert_npy_to_tiff(combo, ref_tiff_path, output_path, downsample_rate=1)
print(f"Saved {approach} cropland combo map.")