import numpy as np
import pandas as pd

classification = "maincrop"
SAVE = True

if classification == "landcover":
    folder = "landcoverclassification"
    classification2 = ""
    base = pd.DataFrame({
        "code": np.array([1, 2, 3, 4, 5, 6]),
        "label": [
            "Built-up surface",  # 1
            "Bare soil",         # 2
            "Water body",        # 3
            "Wetland",           # 4
            "Cropland",          # 5
            "Shrub land"         # 6
        ]
    })

elif classification == "maincrop":
    folder = "cropclassification"
    classification2 = "maincrop_"
    base = pd.DataFrame({
        "code": np.array([0, 1, 2, 3, 4, 5]),
        "label": [
            "Masked",    # 0
            "Cowpea",    # 1
            "Fallow",    # 2
            "Groundnut", # 3
            "Millet",    # 4
            "Sorghum"    # 5
        ]
    })

# Optional: define extra classes you know may appear in 2021
extra_classes = pd.DataFrame({
    "code": [6, 7, 8],
    "label": ["Tree", "Rice", "Other"]
})

# Merge them in advance (so labels show up even if absent in some years)
base = pd.concat([base, extra_classes]).drop_duplicates("code", keep="first")

print("Initial class base:")
print(base)

for year in [2018, 2019, 2021]:
    if year == 2021 and classification == "landcover":
        tessera_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_tessera_prediction_map_whole_{year}_remapped_{classification2}15agg.npy"
        )
        stm_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_stm_prediction_map_whole_{year}_remapped_{classification2}15agg.npy"
        )
        raw_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_raw_prediction_map_whole_{year}_remapped_{classification2}15agg.npy"
        )
        print(f"Unique values in maps for {year}:")
        print(" tessera:", np.unique(tessera_map))
        print(" stm    :", np.unique(stm_map))
        print(" raw    :", np.unique(raw_map))

        if classification == "landcover":
            stm_map = stm_map - 1
            raw_map = raw_map - 1
    else:
        tessera_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_tessera_prediction_map_whole_{year}_{classification2}15agg.npy"
        )
        stm_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_stm_prediction_map_whole_{year}_{classification2}15agg.npy"
        )
        raw_map = np.load(
            f"/maps/mcl66/senegal/{folder}/senegal_raw_prediction_map_whole_{year}_{classification2}15agg.npy"
        )
        print(f"Unique values in maps for {year}:")
        print(" tessera:", np.unique(tessera_map))
        print(" stm    :", np.unique(stm_map))
        print(" raw    :", np.unique(raw_map))

        if classification == "landcover":
            stm_map = stm_map - 1
            raw_map = raw_map - 1

    labels = ["tessera", "stm", "raw"]
    

    for i, current_map in enumerate([tessera_map, stm_map, raw_map]):
        print(f"Processing {labels[i]} map for year {year}...")

        # Compute frequencies
        values, freqs = np.unique(current_map, return_counts=True)
        print(f"Count of masked values in {labels[0]} map for {year}: {np.sum(current_map == 0)}")
        masked = np.sum(current_map == 0)
        perc = (freqs / (current_map.size - masked)) * 100  # Percentage excluding masked values

        # Turn into DataFrame
        df_counts = pd.DataFrame({"code": values, f"{labels[i]}_{year}": perc})

        # Outer merge ensures all codes align and new ones are added
        base = pd.merge(base, df_counts, on="code", how="outer")

print("\nFinal results:")
print(base)

# Save to CSV
if SAVE:
    base.to_csv(
        f"/maps/mcl66/senegal/{folder}/senegal_unique_counts.csv",
        index=False
    )
