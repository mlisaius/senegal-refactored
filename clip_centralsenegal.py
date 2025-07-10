import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

# --- File paths ---
year = 2018  # Define the year of interest
#input_raster_path = f"/maps/mcl66/senegal/representations/{year}_representation_map_10m_utm28n_128bands.tiff"
input_raster_path = f"/maps/mcl66/senegal/representations/{year}_representation_map_10m_utm28n_scales.tif"
bbox_shapefile = "/maps/mcl66/senegal/supporting/centralsenegal_jcam_bbox.shp"

output_raster_path = input_raster_path.replace(".tif", "_clipped.tiff")
output_npy_path = input_raster_path.replace(".tif", "_clipped.npy")

# --- Load bounding box shapefile ---
bbox = gpd.read_file(bbox_shapefile)

# --- Reproject bbox to raster CRS if needed ---
with rasterio.open(input_raster_path) as src:
    raster_crs = src.crs
    if bbox.crs != raster_crs:
        bbox = bbox.to_crs(raster_crs)
    
    #print("src crs", src.crs)
    #print("bbox crs", bbox.crs)

    # --- Clip raster with bbox geometry ---
    clipped_array, clipped_transform = mask(src, bbox.geometry, crop=True, nodata=0)
    clipped_meta = src.meta.copy()
    clipped_meta.update({
        "height": clipped_array.shape[1],
        "width": clipped_array.shape[2],
        "transform": clipped_transform
    })

# --- Save clipped raster as GeoTIFF ---
with rasterio.open(output_raster_path, "w", **clipped_meta) as dest:
    dest.write(clipped_array)

# --- Save clipped raster as .npy ---
np.save(output_npy_path, clipped_array)
print(f"Saved {output_npy_path} with shape {clipped_array.shape}")


print("Clipped raster saved as:")
print(" → TIF:", output_raster_path)
print(" → NPY:", output_npy_path)
