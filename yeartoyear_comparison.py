import numpy as np
import rasterio
from rasterio.transform import Affine   


year1 = 2019
year2 = 2021

#model = 'randomforest'
#year1_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year1}_{model}_1.npy')
#year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_{model}_1.npy')
#year1_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year1}_{model}_1.npy')
#year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_{model}_1.npy')

#model = "mlp"
#year1_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year1}_{model}_9.npy')
#year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_{model}_3.npy')
#year1_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year1}_{model}_3.npy')
#year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_{model}_1.npy')

model = "agg"
year1_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year1}_15agg.npy')
if year2 == 2021:
    year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_remapped_15agg.npy')
else:
    year2_pred = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year2}_15agg.npy')



# remap all labels to 0 for non crop and 1 for crop
remapping = {
    1: 0,  # Built-up surface
    2: 0,  # Bare soil
    3: 0,  # Water body
    4: 0,  # Wetland
    5: 1,  # Cropland
    6: 0,  # Shrub land
    7: 0   # Pasture
}


year1_pred_remapped = np.vectorize(lambda x: remapping.get(x, 0))(year1_pred)
year2_pred_remapped = np.vectorize(lambda x: remapping.get(x, 0))(year2_pred)


change_map = year2_pred_remapped - year1_pred_remapped
decrease_percentage = np.sum(change_map == -1) / change_map.size * 100
increase_percentage = np.sum(change_map == 1) / change_map.size * 100
print(f"Percentage of decrease from {year1} to {year2}: {decrease_percentage:.2f}%")
print(f"Percentage of increase from {year1} to {year2}: {increase_percentage:.2f}%")

#change_map = np.abs(change_map)
print(f"min value in change map: {np.min(change_map)}")
print(f"max value in change map: {np.max(change_map)}")

# calculate the percentage of change
change_percentage = np.sum(change_map != 0) / change_map.size * 100
print(f"Percentage of change from {year1} to {year2}: {change_percentage:.2f}%")

change_map += 1

# Save the change map
#np.save(f'/home/mcl66/code/senegal_code/landcoverclassification/senegal_tessera_change_map_{year1}_{year2}_mlp.npy', change_map)

def convert_npy_to_tiff(npy, ref_tiff_path, output_path, downsample_rate=1):
            # Load npy data, assuming shape (H, W) or (H, W, C)
            data = npy
            
            if data.dtype == np.int64:
                # Convert int64 to int8 if necessary
                print("Converting int64 data to int8...")
                data = data.astype(np.int8)    
                
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
                'dtype': 'int8',
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
                        
# Convert and save the change map as a TIFF
output_tiff_path = f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_change_map_{year1}_{year2}_{model}.tiff'
ref_tiff_path = f"/maps/mcl66/senegal/representations/2018_representation_map_10m_utm28n_scales_clipped.tiff"
convert_npy_to_tiff(change_map, ref_tiff_path, output_tiff_path, downsample_rate=1)
print(f"Change map saved to {output_tiff_path}")