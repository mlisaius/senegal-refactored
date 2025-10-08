import numpy as np
import pandas as pd

year = 2021
labels = "maincrop_code"  # or "landcover_code"

# Load the raster
field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy"
field_id_raster = np.load(field_id_file_path)

# Load the CSV
df = pd.read_csv("/maps/mcl66/senegal/supporting/senegal_fields.csv")
df = df[df["Year"] == year]

# Keep NaNs as-is (don’t fill with 0)
df["Id"] = df["Id"].astype("Int64")  # pandas nullable int
df[labels] = df[labels].astype("Int64")

# Create mapping dictionary, dropping rows with NaNs in either Id or label
df_valid = df.dropna(subset=["Id", labels])
id_to_cropcode = {int(k): int(v) for k, v in zip(df_valid["Id"], df_valid[labels])}

# Define remapping function
def remap_field_id(val):
    if np.isnan(val):   # keep NaN
        return np.nan
    val = int(val)
    if val == 0:        # background/no field
        return 0
    return id_to_cropcode.get(val, 0)

# Apply
vectorized_remap = np.vectorize(remap_field_id, otypes=[float])  # use float to allow NaN
maincrop_raster = vectorized_remap(field_id_raster)

print(f"Unique values in maincrop raster: {np.unique(maincrop_raster[~np.isnan(maincrop_raster)])}")

# Save raster (with NaNs)
if labels == "maincrop_code":
    output_raster_path = field_id_file_path.replace(".npy", "_remapped_crop_labels.npy")
elif labels == "landcover_code":
    output_raster_path = field_id_file_path.replace(".npy", "_remapped_landcover_labels.npy")

np.save(output_raster_path, maincrop_raster)
print(np.nanmean(maincrop_raster), np.nanstd(maincrop_raster),
      np.nanmin(maincrop_raster), np.nanmax(maincrop_raster))
print(f"Remapped raster saved to {output_raster_path}")
