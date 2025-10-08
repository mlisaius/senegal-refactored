import csv
import glob
import os
import statistics as stats

year = 2018
classification = "landcover" # landcover or maincrop

csv_folder = '/maps/mcl66/senegal/classification_reports_singlemodel/'

# Path to your CSV file
for year in [2018]:
    for filename in glob.glob(os.path.join(csv_folder, f'*{year}*landcover.csv')):
    #for filename in [tessera, stm, raw, efm]:
        # Storage for values
        accuracy_vals = []
        macro_vals = []
        weighted_vals = []

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # skip header row

            for row in reader:
                label = row[2]          # column index 2
                value = float(row[5])   # column index 4

                if label == "accuracy":
                    accuracy_vals.append(value)
                elif label == "macro avg":
                    macro_vals.append(value)
                elif label == "weighted avg":
                    weighted_vals.append(value)
                else: continue

        def summarize(values):
            if not values:
                return None, None
            return stats.mean(values), stats.stdev(values) if len(values) > 1 else 0.0

        acc_mean, acc_std = summarize(accuracy_vals)
        macro_mean, macro_std = summarize(macro_vals)
        weighted_mean, weighted_std = summarize(weighted_vals)

        print(f"File: {filename}")
        print(f"Accuracy    → mean: {acc_mean:.4f}, std: {acc_std:.4f}")
        print(f"Macro avg F1   → mean: {macro_mean:.4f}, std: {macro_std:.4f}")
        print(f"Weighted avg F1 → mean: {weighted_mean:.4f}, std: {weighted_std:.4f}")