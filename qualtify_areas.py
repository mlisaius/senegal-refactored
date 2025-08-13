import numpy as np
import pandas as pd


#year = 2018
classification = "landcover"

if classification == "landcover":
    base = pd.DataFrame({"code": np.array([1,2,3,4,5,6]), "label":["Built-up surface", # 1,
                                            "Bare soil", # 2,
                                            "Water body", # 3,
                                            "Wetland", # 4,
                                            "Cropland", # 5,
                                            "Shrub land"]})
                 
print(base)

for year in [2018, 2019]: #, 2021]:
    tessera_map = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year}_15agg.npy")
    stm_map = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_stm_prediction_map_whole_{year}_15agg.npy") -1 
    raw_map = np.load(f"/maps/mcl66/senegal/landcoverclassification/senegal_raw_prediction_map_whole_{year}_15agg.npy") -1

    labels = ["tessera", "stm", "raw"]

    for i, map in enumerate([tessera_map, stm_map, raw_map]):
        # Convert to DataFrame
        #df = pd.DataFrame(map)
        
        
        counts = (np.array(np.unique(map, return_counts=True))).T
        counts = counts.astype(np.float64)
        print(counts[:,1], map.size)

        perc = counts[:, 1]/(map.size)
        #print(counts, counts.shape)
        base[f"{labels[i]}_{year}"] = perc

print(base)
    # Count unique values
    #unique_counts = df.apply(pd.Series.value_counts).fillna(0).astype(int)
    #print(f"Unique counts for {year}:\n{unique_counts}")
    # Save to CSV
base.to_csv(f"/maps/mcl66/senegal/landcoverclassification/senegal_unique_counts.csv")
    
    #print(f"Unique counts for {year} saved to CSV.")
