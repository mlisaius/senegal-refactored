import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# define the year of interest
year = 2021

# use if unsure about how the scaling works
sanitycheck = False  # Set to True to visualize RGB representation

# Load your arrays
rep_file_path = f"/maps/mcl66/senegal/alphaearth/efm_{year}_10m_clipped.npy"

labels = np.load(f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy").squeeze()  # shape (w, h)


# def load_and_dequantize_representation(representation_file_path, scales_file_path):
#     """
#     Load and dequantize int8 representations back to float32.
    
#     Args:
#         representation_file_path: Path to the int8 representation file (H,W,C)
#         scales_file_path: Path to the float32 scales file (H,W)
    
#     Returns:
#         representation_f32: float32 ndarray of shape (H,W,C)
#     """
#     # Load the files
#     representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
#     scales = np.squeeze(np.load(scales_file_path))  # (H, W), dtype=float32
    
#     # Convert int8 to float32 for computation
#     representation_f32 = representation_int8.astype(np.float32)
    
#     # Expand scales to match representation shape
#     # scales shape: (H, W) -> (H, W, 1)
#     #scales_expanded = scales[..., np.newaxis]
#     # scales shape: (H, W) -> (1, H, W)
#     scales_expanded = scales[np.newaxis, :, :]

    
#     # Dequantize by multiplying with scales
#     representation_f32 = representation_f32 * scales_expanded
    
#     return representation_f32

#values = load_and_dequantize_representation(rep_file_path, scales_file_path)
values = np.load(rep_file_path)[:,:3863,:].transpose(2,0,1)

print(f"Loaded values with shape {values.shape} and labels with shape {labels.shape}")

if sanitycheck == True:
    rgb = values[:3]  # shape now (3, H, W)

    # Transpose to (H, W, 3) for visualization
    rgb_img = np.transpose(rgb, (1, 2, 0))

    # Normalize to 0–1 for saving (optional, adjust depending on range)
    rgb_img = rgb_img - np.min(rgb_img)
    rgb_img = rgb_img / (np.max(rgb_img) + 1e-8)  # avoid divide-by-zero

    # Save to file
    plt.imsave("test.png", rgb_img)


# Check shapes
print(values.shape, labels.shape)
assert values.shape[1:] == labels.shape, "Mismatch in spatial dimensions"

# Flatten spatial dims
w, h = labels.shape
num_pixels = w * h

# Flatten labels (w,h) -> (num_pixels,)
labels_flat = labels.flatten()

# Flatten values (128,w,h) -> (128, num_pixels)
values_flat = values.reshape(64, num_pixels)

# Transpose values_flat -> (num_pixels, 128)
values_flat = values_flat.T

# Select pixels where label != 0
mask = labels_flat != 0
labels_sel = labels_flat[mask]
values_sel = values_flat[mask]

# Create DataFrame
df = pd.DataFrame(values_sel, columns=[f"value_{i+1}" for i in range(64)])
df.insert(0, "label", labels_sel)

# Export to CSV
df.to_csv(f"/home/mcl66/code/senegal_code/pixelcsvs/{year}_efm_with_labels.csv", index=False)

print(f"Saved {len(df)} rows to  {year}_efm_with_labels.csv")
