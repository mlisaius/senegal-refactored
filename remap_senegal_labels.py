import numpy as np
import pandas as pd

year = 2021
labels = "maincrop_code"  # or "landcover_code"

# Load the raster
field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy"
field_id_raster = np.load(field_id_file_path)

# Load the CSV and ensure IDs and codes are integers
df = pd.read_csv("/maps/mcl66/senegal/supporting/senegal_fields.csv")
df = df[df["Year"] == year]
df["Id"] = df["Id"].fillna(0).astype(int)
df[labels] = df[labels].fillna(0).astype(int)
#df["landcover_code"] = df["landcover_code"].fillna(0).astype(int)


# Create a dictionary from field ID to maincrop code
id_to_cropcode = {int(k): int(v) for k, v in zip(df["Id"], df[labels])}

# Define a safe remapping function
def remap_field_id(val):
    try:
        val = int(val)
        if val == 0:
            return 0
        return id_to_cropcode.get(val, 0)
    except:
        return 0  # Catch anything unexpected

# Vectorize and apply
vectorized_remap = np.vectorize(remap_field_id)
maincrop_raster = vectorized_remap(field_id_raster)

print(f"Unique values in maincrop raster: {np.unique(maincrop_raster)}")

# Save the remapped raster
if labels == "maincrop_code":
    output_raster_path = field_id_file_path.replace(".npy", "_remapped_crop_labels.npy")
elif labels == "landcover_code":
    output_raster_path = field_id_file_path.replace(".npy", "_remapped_landcover_labels.npy")
    
np.save(output_raster_path, maincrop_raster)
print(np.mean(maincrop_raster), np.std(maincrop_raster), np.min(maincrop_raster), np.max(maincrop_raster))
print(f"Remapped raster saved to {output_raster_path}")
