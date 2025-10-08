# Core libraries
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
import logging
from collections import Counter
from contextlib import contextmanager

# Data handling
import numpy as np
import pandas as pd
import h5py
from imblearn.over_sampling import SMOTE

# Parallel processing
from joblib import Parallel, delayed

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sklearn - modeling and evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Other models
from xgboost import XGBClassifier

# Stats
from scipy import stats

# mapping
import rasterio
from rasterio.transform import Affine

year = 2018
new_year = 2021  # Year for prediction
mixedyear = f"{year}_{new_year}"

TRAINING_RATIO = 0.7
#MODEL = "RandomForest"  # Options: "LogisticRegression", "RandomForest", or "MLP", "XGBOOST", "SVM"
CLASSIFICATION = "landcover"  # Options: "landcover", "maincrop"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
SAVE = "no" # Save model prediction map, "yes" or "no"
SAMPLING = "bypercentage_count"  # "bypercentage", "bypercentage_count", "bycount"  #sampling strategy 
WHOLEMAP = False  # If True, process both the labels and the whole map at once, otherwise process on labels
REPORT = True # If True, save classification report and confusion matrix to CSV
AUGMENT = False  # If True, apply SMOTE to balance training data
REMAP2021 = True  # If True, remap labels for 2021 to reduce pasture and natural vegetation to 0
RUNS = 3 # Number of runs for ensemble methods

# MLP hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.005
NUM_EPOCHS = 200
PATIENCE = 50  # Early stopping parameter


# Log settings
log_file = "btfm_feature_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)
logging.info("Program started.")
logging.info(f"Using device: {DEVICE}")

# Configuration parameters
njobs = 12
chunk_size = 1000
#bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_ndvi_by_doy_clipped.npy"
bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_ndvi_monthly_clipped.npy"
field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy"
updated_fielddata_path = "/maps/mcl66/senegal/supporting/senegal_fields.csv"


#Class names for visualization
if CLASSIFICATION == "landcover":
    n_classes = 6
    classcode = "landcover_code"
    label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_landcover_labels.npy"
    class_names = [
        "Built-up surface", # 1,
        "Bare soil", # 2,
        "Water body", # 3,
        "Wetland", # 4,
        "Cropland", # 5,
        "Shrub land", # 6,
        "Pasture", # 7,
        "Natural vegetation" # 8
    ]
    outfolder = "landcoverclassification"
    
elif CLASSIFICATION == "maincrop":
    outfolder = "cropclassification"
    agg_pred_map_mask = np.load('/maps/mcl66/senegal/landcoverclassification/senegal_tessera_croplandcombo_map.npy')
    
    classcode = "maincrop_code"
    label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_crop_labels.npy"
    class_names = [
        "Cowpea", # 1,
        "Fallow", # 2,
        "Groundnut", # 3,
        "Millet", # 4,
        "Sorghum", # 5
        "Tree", #6
        "Rice", #7
        "Other" #8
    ]
    n_classes = 5
    if year == 2021:
        n_classes = 8

    

def safe_vstack(arrays, empty_shape=None):
    filtered = [arr for arr in arrays if arr.size > 0]
    if filtered:
        return np.vstack(filtered)
    else:
        if empty_shape is not None:
            return np.empty(empty_shape)
        else:
            return np.empty((0,))

def safe_hstack(arrays, empty_shape=None):
    filtered = [arr for arr in arrays if arr.size > 0]
    if filtered:
        return np.hstack(filtered)
    else:
        if empty_shape is not None:
            return np.empty(empty_shape, dtype=int)
        else:
            return np.empty((0,), dtype=int)


VAL_TEST_SPLIT_RATIO = 1/2  # Validation to test set split ratio

#seeds = [1]
#seeds = [1, 2, 3, 4, 5]  # List of seeds for reproducibility
seeds = list(range(1, RUNS+1))

# initialize aggregated matrices
#n_classes = 6 if CLASSIFICATION == "landcover" else 8  # Adjust based on classification type
#print(f"Number of classes for matrix: {n_classes}")
#aggregate_cm = np.zeros((n_classes, n_classes), dtype=int)
#normalized_cms = []
trained_models = []

for seed in seeds: 
    if REMAP2021:
            if (year == 2021) & (CLASSIFICATION == "landcover"):
                REDUCE_LABELS = True  # For 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
                remapped_labels = "_remapped"
            else:
                REDUCE_LABELS = False
                remapped_labels = ""
    # ----------------- Data loading and preprocessing -----------------
    logging.info(f"Training ratio: {TRAINING_RATIO}")
    logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
    #logging.info(f"Selected model: {MODEL}")
    logging.info("Loading labels and field IDs...")

    labels = (np.load(label_file_path).astype(np.int64)).squeeze()
    
    field_ids = np.load(field_id_file_path).squeeze()
    
    if REDUCE_LABELS:
        # for 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
        logging.info("Reducing labels...")
        # remap labels to 0 for pasture (7) and natural vegetation (8)
        mask = np.isin(labels, [7, 8])
        labels[mask] = 0  # Set pasture and natural vegetation to 0
        field_ids[mask] = 0  # Set corresponding field IDs to 0
                
    #if MODEL == "XGBOOST":
    labels -= 1
    
    H, W = labels.shape
    logging.info(f"Data dimensions: {H}x{W}")


    # Select valid classes
    logging.info("Identifying valid classes...")
    class_counts = Counter(labels.ravel())
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}  
    #if MODEL == "XGBOOST":
    valid_classes.discard(-1)
    #else:
    #    valid_classes.discard(0)
    logging.info(f"Valid classes: {sorted(valid_classes)}")

    # ----------------- Train/validation/test set split -----------------
    logging.info("Splitting data into train/val/test sets...")
    fielddata_df = pd.read_csv(updated_fielddata_path)
    fielddata_df = fielddata_df[fielddata_df['Year'] == year]
    print(f"Total fields in fielddata: {len(fielddata_df)}")

    
    # Shuffle the DataFrame in-place before sampling
    fielddata_df = fielddata_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    area_summary = fielddata_df.groupby(classcode)['Area_ha'].sum().reset_index()
    count_summary = fielddata_df.groupby(classcode).size().reset_index(name='count')

    #area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)

    if SAMPLING == "bypercentage":
        print("Sampling by percentage of area...with VAL_TEST_SPLIT_RATIO:", VAL_TEST_SPLIT_RATIO)
        train_fids = []
        val_fids = []
        test_fids = []

        for _, row in area_summary.iterrows():
            sn_code = row[classcode]
            total_area = row['Area_ha']
            target_train_area = total_area * TRAINING_RATIO

            # Get and shuffle all fields for this class
            rows_sncode = fielddata_df[fielddata_df[classcode] == sn_code]
            rows_sncode = rows_sncode.sort_values(by='Area_ha').reset_index(drop=True)  # sorted by area (smallest to largest)

            # --- TRAINING SELECTION BY AREA ---
            selected_fids = []
            selected_area_sum = 0.0
            for _, r2 in rows_sncode.iterrows():
                if selected_area_sum < target_train_area:
                    selected_fids.append(int(r2['Id']))
                    selected_area_sum += r2['Area_ha']
                else:
                    break

            train_fids.extend(selected_fids)

            # --- REMAINING FIELDS FOR VAL/TEST ---
            remaining_rows = rows_sncode[~rows_sncode['Id'].isin(selected_fids)].copy()
            remaining_rows = remaining_rows.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle remaining

            val_count = max(1, int(len(remaining_rows) * VAL_TEST_SPLIT_RATIO))
            val_rows = remaining_rows.iloc[:val_count]
            test_rows = remaining_rows.iloc[val_count:]

            val_fids.extend(val_rows['Id'].astype(int).tolist())
            test_fids.extend(test_rows['Id'].astype(int).tolist())

        # Remove potential duplicates just in case
        train_fids = np.array(list(set(train_fids)))
        val_fids = np.array(list(set(val_fids)))
        test_fids = np.array(list(set(test_fids)))

        logging.info(f"Number of train FIDs: {len(train_fids)}")
        logging.info(f"Number of val FIDs: {len(val_fids)}")
        logging.info(f"Number of test FIDs: {len(test_fids)}")

    
    if SAMPLING == "bypercentage_count":
        fewshot = ""
        # Collect training set field IDs
        train_fids = []
        val_fids = []
        test_fids = []

        for _, row in count_summary.iterrows():
            sn_code = row[classcode]
            total_count = row['count']
            target_train_count = int(total_count * TRAINING_RATIO)

            # Filter rows for this class and shuffle them
            rows_sncode = fielddata_df[fielddata_df[classcode] == sn_code]
            rows_sncode = rows_sncode.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

            # Split into train, val, test
            selected_train = rows_sncode.iloc[:target_train_count]
            remaining = rows_sncode.iloc[target_train_count:]

            # Compute validation count
            val_count = max(1, int(len(remaining) * VAL_TEST_SPLIT_RATIO))

            selected_val = remaining.iloc[:val_count]
            selected_test = remaining.iloc[val_count:]

            # Append field IDs to each set
            train_fids.extend(selected_train['Id'].astype(int).tolist())
            val_fids.extend(selected_val['Id'].astype(int).tolist())
            test_fids.extend(selected_test['Id'].astype(int).tolist())

        # Convert to numpy arrays
        train_fids = np.array(list(set(train_fids)))
        val_fids = np.array(list(set(val_fids)))
        test_fids = np.array(list(set(test_fids)))

        logging.info(f"Number of train FIDs: {len(train_fids)}")
        logging.info(f"Number of val FIDs: {len(val_fids)}")
        logging.info(f"Number of test FIDs: {len(test_fids)}")

    
    if SAMPLING == "bycount":
        # Initialize field ID lists
        fewshot = "_"
        train_fids = []
        val_fids = []
        test_fids = []
        #snar_codes = fielddata_df['landcover_code'].unique()
        snar_codes = fielddata_df[classcode].unique()
        print(f"Unique landcover codes: {snar_codes}")

        for sn_code in snar_codes:
            rows_sncode = fielddata_df[fielddata_df[classcode] == sn_code]
            fids = rows_sncode['Id'].unique()
            fids = np.array(fids)
            np.random.shuffle(fids)

            NUM_FIELDS = 5
            if len(fids) >= NUM_FIELDS*2:
                # Enough fields: take 10 for train, 10 for val, rest for test
                test_fids.extend(fids[:NUM_FIELDS])
                val_fids.extend(fids[NUM_FIELDS:NUM_FIELDS*2])
                train_fids.extend(fids[NUM_FIELDS*2:])
            else:
                total = len(fids)
                n_val = min(5, total // 2)
                n_test = min(5, total - n_val)
                n_train = max(0, total - n_val - n_test)

                val_fids.extend(fids[:n_val])
                test_fids.extend(fids[n_val:n_val+n_test])
                train_fids.extend(fids[n_val+n_test:])

        train_fids = np.array(train_fids)
        val_fids = np.array(val_fids)
        test_fids = np.array(test_fids)
        
        for name, fids in [("Train", train_fids), ("Val", val_fids), ("Test", test_fids)]:
            subset = fielddata_df[fielddata_df["Id"].isin(fids)]
            print(f"\n{name} set class distribution:")
            print(subset[classcode].value_counts())

    #print("Train field IDs:", train_fids)
    print("Val field IDs:", val_fids)
    logging.info(f"Train fields: {len(train_fids)}, Val fields: {len(val_fids)}, Test fields: {len(test_fids)}")

    # ----------------- Create training/validation/testing split map -----------------
    logging.info("Creating train/val/test split map for visualization...")
    # Now we need to create the train/test/val mask using field_ids
    # Vectorized mask creation
    train_test_mask = np.zeros_like(field_ids, dtype=np.int8)
    train_test_mask[np.isin(field_ids, train_fids)] = 1
    train_test_mask[np.isin(field_ids, val_fids)] = 2
    train_test_mask[np.isin(field_ids, test_fids)] = 3

    #if MODEL != "XGBOOST":
    #    valid_label_mask = (labels > 0)
    #elif MODEL == "XGBOOST":
    valid_label_mask = (labels >= 0)

    # Report class distribution by pixel after split
    y = labels  # Don't prefilter

    for split_id, name in [(1, "Train"), (2, "Val"), (3, "Test")]:
        mask = (train_test_mask == split_id)
        y_split = y[mask]
        print(f"{name} pixel class distribution:")
        print(pd.Series(y_split).value_counts())


    # ----------------- Chunk processing -----------------
    def process_chunk(h_start, h_end, w_start, w_end, file_path):
        logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
        
        # Load the chunk (new shape: H, W, bands)
        print("Loading chunk from:", file_path)
        #print("tile chunk shape:", np.load(file_path).shape)
        #/maps/mcl66/senegal/d-pixel/merged_clipped/2021_merged_ndvi_monthly_clipped.npy
        tile_chunk = (np.load(file_path).transpose(1,2,0))[h_start:h_end, w_start:w_end, :]  # shape: (h, w, b)

        # Reshape to (h * w, b) for ML model
        h, w, b = tile_chunk.shape
        s2_band_chunk = tile_chunk.reshape(-1, b)

        # Load labels and field_ids
        y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
        fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()

        # Filter valid pixels
        valid_mask = np.isin(y_chunk, list(valid_classes))
        X_chunk = s2_band_chunk[valid_mask]
        y_chunk = y_chunk[valid_mask]
        fieldid_chunk = fieldid_chunk[valid_mask]

        # Train/val/test split
        train_mask = np.isin(fieldid_chunk, train_fids)
        val_mask = np.isin(fieldid_chunk, val_fids)
        test_mask = np.isin(fieldid_chunk, test_fids)

        return (X_chunk[train_mask], y_chunk[train_mask],
                X_chunk[val_mask], y_chunk[val_mask],
                X_chunk[test_mask], y_chunk[test_mask])

    # Parallel processing
    chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
            for h in range(0, H, chunk_size)
            for w in range(0, W, chunk_size)]
    logging.info(f"Total chunks: {len(chunks)}")

    results = Parallel(n_jobs=njobs)(
        delayed(process_chunk)(h_start, h_end, w_start, w_end, bands_file_path)
        for h_start, h_end, w_start, w_end in chunks
    )

    feature_dim = None
    
    for res in results:
        if res[0].size > 0:
            feature_dim = res[0].shape[1]  # number of columns = feature dim
            break

    if feature_dim is None:
        raise ValueError("No training data found to infer feature dimension!")
    logging.info(f"Feature dimension: {feature_dim}")

        
    # feature_dim inferred previously
    X_train = safe_vstack([res[0] for res in results], empty_shape=(0, feature_dim))
    y_train = safe_hstack([res[1] for res in results], empty_shape=(0,))

    X_val = safe_vstack([res[2] for res in results], empty_shape=(0, feature_dim))
    y_val = safe_hstack([res[3] for res in results], empty_shape=(0,))

    X_test = safe_vstack([res[4] for res in results], empty_shape=(0, feature_dim))
    y_test = safe_hstack([res[5] for res in results], empty_shape=(0,))


    logging.info(f"Unique y training: {np.unique(y_train)}")
    logging.info(f"Unique y val: {np.unique(y_val)}")
    logging.info(f"Unique y test: {np.unique(y_test)}")

    logging.info(f"Data split summary:")
    logging.info(f"  Train set: {X_train.shape[0]} samples")
    logging.info(f"  Validation set: {X_val.shape[0]} samples")
    logging.info(f"  Test set: {X_test.shape[0]} samples")

    # Print data shapes
    logging.info(f"X_train shape: {X_train.shape}")
    input_size = X_train.shape[1]
    logging.info(f"Input feature dimension: {input_size}")
    
    
    def generate_class_spectra(ndvi_blocks, n_clusters=60):
        """
        Cluster NDVI blocks into class spectra using K-Means.
        
        Parameters:
            ndvi_blocks (np.ndarray): NDVI data, shape (timesteps, H, W)
            n_clusters (int): Number of clusters
        
        Returns:
            class_spectra (np.ndarray): (n_clusters, timesteps)
            label_map (np.ndarray): (H, W) cluster labels for each pixel
        """
        timesteps, H, W = ndvi_blocks.shape
        
        # Flatten spatial dims
        #X = ndvi_blocks.reshape(timesteps, H*W).T  # (num_pixels, timesteps)
        X = ndvi_blocks.reshape(timesteps, H*W).T  # (num_pixels, timesteps)

        # Remove pixels that are all-NaN or too sparse
        valid_mask = ~np.isnan(X).all(axis=1)
        X = X[valid_mask]

        # Optionally interpolate remaining NaNs linearly per pixel
        from scipy.interpolate import interp1d
        def interp_nan(curve):
            x = np.arange(len(curve))
            m = ~np.isnan(curve)
            if m.sum() < len(curve) // 2:   # skip if too many NaNs
                return None
            f = interp1d(x[m], curve[m], kind="linear", fill_value="extrapolate")
            return f(x)

        X_interp = np.array([interp_nan(c) for c in X if interp_nan(c) is not None])
        X = X_interp.astype(np.float32)
        logging.info(f"Valid NDVI curves after cleaning: {len(X)} ({len(X)/len(valid_mask):.1%} of total)")

        # KMeans
        #kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=1, random_state=42)

        labels = kmeans.fit_predict(X)
        
        # Compute class spectra
        class_spectra = np.zeros((n_clusters, timesteps))
        for i in range(n_clusters):
            class_spectra[i] = X[labels == i].mean(axis=0)
        
        # Reshape labels back to image
        label_map = labels.reshape(H, W)
        
        return class_spectra, label_map


    def generate_ideal_spectra(X_train, y_train):
        """
        Generate ideal spectra from labeled NDVI training data.
        
        Parameters:
            X_train (np.ndarray): NDVI data for training samples, shape (n_samples, timesteps)
            y_train (np.ndarray): Labels for training samples, shape (n_samples,)
        
        Returns:
            ideal_spectra (dict): {class_label: mean NDVI curve}
        """
        ideal_spectra = {}
        unique_labels = np.unique(y_train)
        
        for label in unique_labels:
            mask = (y_train == label)
            ideal_spectra[label] = X_train[mask].mean(axis=0)
        
        return ideal_spectra


    def generate_class_spectra(ndvi_path, n_clusters=60):
        """
        Cluster NDVI blocks into class spectra using K-Means.
        
        Parameters:
            ndvi_blocks (np.ndarray): NDVI data, shape (timesteps, H, W)
            n_clusters (int): Number of clusters
        
        Returns:
            class_spectra (np.ndarray): (n_clusters, timesteps)
            label_map (np.ndarray): (H, W) cluster labels for each pixel
        """
        ndvi_blocks = np.load(ndvi_path)
        
        timesteps, H, W = ndvi_blocks.shape
        
        # Flatten spatial dims
        X = ndvi_blocks.reshape(timesteps, H*W).T  # (num_pixels, timesteps)
        X = np.nan_to_num(X, nan=0.0)
        
        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Compute class spectra
        class_spectra = np.zeros((n_clusters, timesteps))
        for i in range(n_clusters):
            class_spectra[i] = X[labels == i].mean(axis=0)
        
        # Reshape labels back to image
        label_map = labels.reshape(H, W)
        
        return class_spectra, label_map


    def generate_ideal_spectra(X_train, y_train):
        """
        Generate ideal spectra from labeled NDVI training data.
        
        Parameters:
            X_train (np.ndarray): NDVI data for training samples, shape (n_samples, timesteps)
            y_train (np.ndarray): Labels for training samples, shape (n_samples,)
        
        Returns:
            ideal_spectra (dict): {class_label: mean NDVI curve}
        """
        ideal_spectra = {}
        unique_labels = np.unique(y_train)
        
        for label in unique_labels:
            mask = (y_train == label)
            ideal_spectra[label] = X_train[mask].mean(axis=0)
        
        return ideal_spectra


    def match_spectra(class_spectra, ideal_spectra, threshold=0.8):
        """
        Match class spectra to ideal spectra using R² similarity.
        
        Parameters:
            class_spectra (np.ndarray): (n_clusters, timesteps)
            ideal_spectra (dict): {class_label: NDVI curve}
            threshold (float): R² cutoff for assigning a label
        
        Returns:
            matches (dict): {cluster_idx: matched_class_label or None}
            scores (dict): {cluster_idx: (best_label, best_r2)}
        """
        matches = {}
        scores = {}
        
        for i, class_curve in enumerate(class_spectra):
            best_label = None
            best_r2 = -np.inf
            for label, ideal_curve in ideal_spectra.items():
                class_curve = np.nan_to_num(class_curve, nan=np.nanmean(class_curve))
                ideal_curve = np.nan_to_num(ideal_curve, nan=np.nanmean(ideal_curve))
                r2 = r2_score(ideal_curve, class_curve)
                if r2 > best_r2:
                    best_r2 = r2
                    best_label = label
            # Only assign if strong enough correlation
            matches[i] = best_label if best_r2 >= threshold else None
            scores[i] = (best_label, best_r2)
        
        return matches, scores

    
    class_spectra, label_map = generate_class_spectra(bands_file_path, n_clusters=60)
    class_spectra = np.nan_to_num(class_spectra, nan=0.0)
    ideal_spectra = generate_ideal_spectra(X_train, y_train)
    ideal_spectra = np.nan_to_num(ideal_spectra, nan=0.0)

    matches, scores = match_spectra(class_spectra, ideal_spectra, threshold=0.8)
    
    
    def classify_with_ideal(X, ideal_spectra):
        """
        Classify each NDVI curve by matching it to the closest ideal spectrum (highest R²).
        
        Parameters:
            X (np.ndarray): Test NDVI data, shape (n_samples, timesteps)
            ideal_spectra (dict): {class_label: mean NDVI curve}
        
        Returns:
            y_pred (np.ndarray): Predicted labels for X
        """
        labels = list(ideal_spectra.keys())
        ideal_matrix = np.stack([ideal_spectra[l] for l in labels])  # (num_classes, timesteps)
        
        # Compute cosine similarity (can replace with R² if preferred)
        sims = cosine_similarity(X, ideal_matrix)  # (n_samples, num_classes)
        
        # Assign best matching label
        best_idx = sims.argmax(axis=1)
        y_pred = np.array([labels[i] for i in best_idx])
        return y_pred


    def evaluate_test_set(X_test, y_test, ideal_spectra):
        """
        Evaluate SMT classification on holdout test set.
        
        Returns metrics.
        """
        X_test = np.nan_to_num(X_test, nan=0.0)
        ideal_spectra = {k: np.nan_to_num(v, nan=0.0) for k, v in ideal_spectra.items()}
        y_pred = classify_with_ideal(X_test, ideal_spectra)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        report = classification_report(y_test, y_pred, digits=3)
        
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        
        return y_pred, acc, cm, report

    
    year = new_year


    #bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_bands_by_doy_clipped.npy"
    bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_ndvi_monthly_clipped.npy"
    mask_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_masks_by_doy_clipped.npy"
    field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy"
    updated_fielddata_path = "/maps/mcl66/senegal/supporting/senegal_fields.csv"

    print(f"Processing year: {year}")

    if REMAP2021:
            if (year == 2021) & (CLASSIFICATION == "landcover"):
                REDUCE_LABELS = True  # For 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
                remapped_labels = "_remapped"
            else:
                REDUCE_LABELS = False
                remapped_labels = ""
                
    if CLASSIFICATION == "landcover":
        classcode = "landcover_code"
        label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_landcover_labels.npy"
        class_names = [
            "Built-up surface", # 1,
            "Bare soil", # 2,
            "Water body", # 3,
            "Wetland", # 4,
            "Cropland", # 5,
            "Shrub land", # 6,
            "Pasture", # 7,
            "Natural vegetation" # 8
        ]
        outfolder = "landcoverclassification"
        
    elif CLASSIFICATION == "maincrop":
        outfolder = "cropclassification"
        agg_pred_map_mask = np.load('/maps/mcl66/senegal/landcoverclassification/senegal_tessera_croplandcombo_map.npy')
        
        classcode = "maincrop_code"
        label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_crop_labels.npy"
        class_names = [
            "Cowpea", # 1,
            "Fallow", # 2,
            "Groundnut", # 3,
            "Millet", # 4,
            "Sorghum", # 5
            "Tree", #6
            "Rice", #7
            "Other" #8
        ]
    # ----------------- Data loading and preprocessing -----------------
    logging.info(f"Training ratio: {TRAINING_RATIO}")
    logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
    #logging.info(f"Selected model: {MODEL}")
    logging.info("Loading labels and field IDs...")

    fielddata_df = pd.read_csv(updated_fielddata_path)
    fielddata_df = fielddata_df[fielddata_df['Year'] == year]
    all_ids = fielddata_df['Id'].astype(int).unique()

    print(f"Loading year 2 labels from {label_file_path}")
    labels = (np.load(label_file_path).astype(np.int64)).squeeze()

    field_ids = np.load(field_id_file_path).squeeze()

    if REDUCE_LABELS:
        # for 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
        logging.info("Reducing labels...")
        # remap labels to 0 for pasture (7) and natural vegetation (8)
        mask = np.isin(labels, [7, 8])
        labels[mask] = 0  # Set pasture and natural vegetation to 0
        field_ids[mask] = 0  # Set corresponding field IDs to 0
                
    #if MODEL == "XGBOOST":
    labels -= 1

    H, W = labels.shape
    logging.info(f"Data dimensions: {H}x{W}")


    # Select valid classes
    logging.info("Identifying valid classes...")
    class_counts = Counter(labels.ravel())
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}  
    #if MODEL == "XGBOOST":
    valid_classes.discard(-1)
    #else:
    #    valid_classes.discard(0)
    logging.info(f"Valid classes: {sorted(valid_classes)}")
    
    
    # treat NaNs as 0
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    y_pred, acc, cm, report = evaluate_test_set(X_test, y_test, ideal_spectra)

    def save_classification_report(y_true, y_pred, filename="classification_report.csv"):
        # Get classification report as a dict and convert to DataFrame (multi-row format)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})

        # Add the seed and cpu_time columns to every row
        # report_df.insert(0, "cpu_time", cpu_time)
        report_df.insert(0, "seed", seed)

        # If file exists, append the new block; else create new file
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, report_df], ignore_index=True)
        else:
            combined_df = report_df

        # Save the full DataFrame back to CSV
        combined_df.to_csv(filename, index=False, float_format='%.4f')

    class_report_filename = f'/maps/mcl66/senegal/classification_reports/senegal_specmat_classification_report_{mixedyear}_agg_{CLASSIFICATION}.csv'

    save_classification_report(y_test, y_pred, class_report_filename)