# Core libraries
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Other models
from xgboost import XGBClassifier

# Stats
from scipy import stats

# mapping
import rasterio
from rasterio.transform import Affine

year = 2018
new_year = 2021
mixedyear = f"{year}_{new_year}"

TRAINING_RATIO = 0.7
MODEL = "RandomForest"  # Options: "LogisticRegression", "RandomForest", or "MLP", "XGBOOST", "SVM"
CLASSIFICATION = "landcover"  # Options: "landcover", "maincrop"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
SAVE = "no" # Save model prediction map, "yes" or "no"
SAMPLING = "bypercentage_count"  # "bypercentage", "bypercentage_count", "bycount"  #sampling strategy 
WHOLEMAP = False  # If True, process both the labels and the whole map at once, otherwise process on labels
REPORT = True # If True, save classification report and confusion matrix to CSV
AUGMENT = False  # If True, apply SMOTE to balance training data
NUM_SEEDS = 2
REMAP2021 = True  # If True, remap labels for 2021 to reduce pasture and natural vegetation to 0


# MLP hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.005
NUM_EPOCHS = 200
PATIENCE = 50  # Early stopping parameter


#if MODEL in ["XGBOOST", "MLP"]:
VAL_TEST_SPLIT_RATIO = 1/4  # Validation to test set split ratio
#else:
#    VAL_TEST_SPLIT_RATIO = 1/20  # Validation to test set split ratio


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
# bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_bands_by_doy_clipped_dec10.npy"
# mask_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_masks_by_doy_clipped_dec10.npy"

# label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_landcover_labels_dec10.npy"
# field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_dec10.npy"
# updated_fielddata_path = "/maps/mcl66/senegal/supporting/senegal_fields.csv"

# sar_asc_bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_sar_bands_by_doy_clipped_dec10.npy"

bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_bands_by_doy_clipped.npy"
mask_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_masks_by_doy_clipped.npy"

label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_landcover_labels.npy"
field_id_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped.npy"
updated_fielddata_path = "/maps/mcl66/senegal/supporting/senegal_fields.csv"

sar_asc_bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_sar_bands_by_doy_clipped.npy"



# Sentinel normalization constants
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                        2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                       1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

TRAIN_TIME_STEPS = 0

#Class names for visualization
if CLASSIFICATION == "landcover":
    classcode = "landcover_code"
    n_classes = 6
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
    n_classes = 5
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

# initialize aggregated matrices

# aggregate_cm = np.zeros((n_classes, n_classes), dtype=int)
# normalized_cms = []
seeds = list(range(20, 21 + NUM_SEEDS))

trained_models = []

for seed in seeds: 
    
    if REMAP2021:
            if (year == 2021) & (CLASSIFICATION == "landcover"):
                print("Remapping labels for 2021 landcover classification...")
                REDUCE_LABELS = True  # For 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
                remapped_labels = "_remapped"
            else:
                REDUCE_LABELS = False
                remapped_labels = ""
            
    start = time.process_time()
              
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----------------- Data loading and preprocessing -----------------
    logging.info(f"Training ratio: {TRAINING_RATIO}")
    logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
    logging.info(f"Selected model: {MODEL}")
    logging.info("Loading labels and field IDs...")

    labels = (np.load(label_file_path).astype(np.int64)).squeeze()
    field_ids = np.load(field_id_file_path).squeeze()
    

    #if MODEL == "XGBOOST":
    
    if REDUCE_LABELS:
            # for 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
            logging.info("Reducing labels...")
            # remap labels to 0 for pasture (7) and natural vegetation (8)
            mask = np.isin(labels, [7, 8])
            labels[mask] = 0  # Set pasture and natural vegetation to 0
            field_ids[mask] = 0  # Set corresponding field IDs to 0
    
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
    
    logging.info("Splitting data into train/val/test sets...")
    fielddata_df = pd.read_csv(updated_fielddata_path)
    fielddata_df = fielddata_df[fielddata_df['Year'] == year]

    # Shuffle the DataFrame in-place before sampling
    fielddata_df = fielddata_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    area_summary = fielddata_df.groupby(classcode)['Area_ha'].sum().reset_index()
    count_summary = fielddata_df.groupby(classcode).size().reset_index(name='count')                                     


    ids = fielddata_df["Id"].unique()

    logging.info(f"Unique field IDs: {len(ids)}")

    # ----------------- Train/validation/test set split -----------------


    
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
            rows_sncode = rows_sncode.sample(frac=1).reset_index(drop=True)  # Shuffle

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
        
    

    # ----------------- Chunking -----------------
    def process_chunk(h_start, h_end, w_start, w_end, file_path):
        logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
        
        # Load data for feature extraction (only once per chunk)
        s2_bands = np.load(bands_file_path,mmap_mode = 'r')[:, h_start:h_end, w_start:w_end, :]
        #s2_bands = s2_data[..., :10]  # First 10 bands to normalize
        #s2_vis = s2_data[..., 10:]    # Last 4 bands are vegetation indices (NDVI, GCVI, EVI, LSWI)
        TRAIN_TIME_STEPS = s2_bands.shape[0]
        print(f"Time steps in this chunk: {TRAIN_TIME_STEPS}", s2_bands.shape)
        # Normalize original bands
        s2_bands = (s2_bands - S2_BAND_MEAN) / S2_BAND_STD

        # Recombine normalized bands with VIs
        #s2_data = np.concatenate([s2_bands, s2_vis], axis=-1)
        #s2_mask = (np.load(mask_file_path)[:, h_start:h_end, w_start:w_end]).squeeze(axis=-1)
        s2_mask = np.load(mask_file_path)[:, h_start:h_end, w_start:w_end]
        s2_mask = np.squeeze(s2_mask) 
        s2_mask = s2_mask[..., np.newaxis]

        # Apply the mask (broadcasts automatically)
        s2_bands = s2_bands * s2_mask
        time_steps, h, w, s2_bands_total = s2_bands.shape
        s2_band_chunk = s2_bands.transpose(1, 2, 0, 3).reshape(-1, time_steps * s2_bands_total)  # (h*w, time_steps * bands)
        
        sar_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
        #sar_desc_data = np.load(sar_desc_bands_file_path)[:, h_start:h_end, w_start:w_end]

        # check the data shapes of s2, sar_asc, and sar_desc
        logging.info(f"S2 data shape: {s2_bands.shape}")
        
        # Normalize only the original bands
        sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD

        # Recombine
        #sar_chunk = np.concatenate([sar_asc_bands, sar_asc_rvi], axis=-1)

        time_steps, h, w, bands = sar_chunk.shape
        sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
        # Concatenate s2 and s1
        #print(f"shape of s2_band_chunk {s2_band_chunk.shape} and shape of sar_band_chunk {sar_band_chunk.shape}")
        X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1) # (h*w, time_steps*bands*2)

        
        y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
        fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()
        
        # Filter valid data
        valid_mask = np.isin(y_chunk, list(valid_classes))
        X_chunk, y_chunk, fieldid_chunk = X_chunk[valid_mask], y_chunk[valid_mask], fieldid_chunk[valid_mask]
        
        # Split into train/val/test sets
        train_mask = np.isin(fieldid_chunk, train_fids)
        val_mask = np.isin(fieldid_chunk, val_fids)
        test_mask = np.isin(fieldid_chunk, test_fids)
        
        
        
        return (X_chunk[train_mask], y_chunk[train_mask], 
                X_chunk[val_mask], y_chunk[val_mask],
                X_chunk[test_mask], y_chunk[test_mask])

    # Define chunks for parallel processing
    chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
            for h in range(0, H, chunk_size)
            for w in range(0, W, chunk_size)]
    logging.info(f"Total chunks: {len(chunks)}")

    results = Parallel(n_jobs=njobs)(
        delayed(process_chunk)(h_start, h_end, w_start, w_end, bands_file_path)
        for h_start, h_end, w_start, w_end in chunks
    )


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


    # if MODEL != "XGBOOST":
    #     # Remove class 0 (background) from training data
    #     logging.info("Removing class 0 (background) from training data...")
    #     indices_to_remove = np.where(y_train == 0)[0]
    #     y_train = np.delete(y_train, indices_to_remove, axis=0)
    #     X_train = np.delete(X_train, indices_to_remove, axis=0)

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
    
    for MODEL in ["XGBOOST", "RandomForest", "MLP"]: #xgboost

        print(f"\nTraining model: {MODEL}")   
        # if MODEL in ["XGBOOST", "MLP"]:
        #     VAL_TEST_SPLIT_RATIO = 1/4  # Validation to test set split ratio
        # else:
        #     VAL_TEST_SPLIT_RATIO = 1/20  # Validation to test set split ratio

        
        # ----------------- Define MLP model -----------------
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction

            def forward(self, inputs, targets):
                """
                inputs: raw logits (batch_size, num_classes)
                targets: class indices (batch_size,)
                """
                log_probs = nn.functional.log_softmax(inputs, dim=1)  # log probabilities
                probs = torch.exp(log_probs)                         # probabilities
                targets_one_hot = nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()

                pt = (probs * targets_one_hot).sum(dim=1)            # probs of the true classes
                log_pt = (log_probs * targets_one_hot).sum(dim=1)    # log probs of the true classes

                focal_term = (1 - pt) ** self.gamma
                loss = -self.alpha * focal_term * log_pt

                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss

            
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.4):
                super(MLP, self).__init__()
                
                # First layer with BatchNorm and ReLU
                self.layer1 = nn.Sequential(
                    nn.Linear(input_size, hidden_sizes[0]),
                    nn.BatchNorm1d(hidden_sizes[0]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Second layer with BatchNorm and ReLU
                self.layer2 = nn.Sequential(
                    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                    nn.BatchNorm1d(hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Third layer with BatchNorm and ReLU
                self.layer3 = nn.Sequential(
                    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                    nn.BatchNorm1d(hidden_sizes[2]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
                
                # Output layer
                self.output = nn.Linear(hidden_sizes[2], num_classes)
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.output(x)
                return x

        # Function to train MLP model
        def train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size):
            """Train MLP model and return the trained model"""
            logging.info(f"Starting MLP training with input size: {input_size}")
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)  # Shift labels from 1-18 to 0-17

            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Initialize model
            hidden_sizes = [2048, 1024, 512]  # Three hidden layer sizes
            model = MLP(input_size, hidden_sizes, num_classes).to(DEVICE)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)  # Weighted loss
            #criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')  # Focal loss
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            
            # Train the model
            best_val_loss = float('inf')
            early_stop_counter = 0
            best_model_state = None
            
            logging.info(f"MLP model architecture:\n{model}")
            
            for epoch in range(NUM_EPOCHS):
                # Training phase
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += batch_y.size(0)
                    correct += predicted.eq(batch_y).sum().item()
                
                train_loss = train_loss / len(train_loader)
                train_acc = 100.0 * correct / total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += batch_y.size(0)
                        correct += predicted.eq(batch_y).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100.0 * correct / total
                
                # Update learning rate
                scheduler.step(val_loss)
                
                logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    early_stop_counter = 0
                    logging.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
                else:
                    early_stop_counter += 1
                
                # Early stopping
                if early_stop_counter >= PATIENCE:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Load best model
            model.load_state_dict(best_model_state)
            return model

        # MLP model prediction function
        def mlp_predict(model, X):
            """Make predictions using MLP model"""
            model.eval()
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            
            # Batch predict for large datasets
            batch_size = 1024
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    outputs = model(batch_X)
                    _, preds = torch.max(outputs, 1)
                    predictions.append((preds).cpu().numpy())
            
            return np.concatenate(predictions)
        
        def mlp_predict_proba(model, X):
            """Return class probabilities from the MLP model (softmax output)."""
            model.eval()
            X_tensor = torch.FloatTensor(X).to(DEVICE)

            batch_size = 1024
            probs = []

            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    outputs = model(batch_X)
                    softmaxed = torch.softmax(outputs, dim=1)
                    probs.append(softmaxed.cpu().numpy())

            return np.vstack(probs) 

        # ----------------- Model training -----------------
        logging.info(f"\nTraining {MODEL}...")

        if MODEL == "LogisticRegression":
            model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                C=1e4,
                max_iter=100000,
                n_jobs=njobs,
                random_state=42
            )
            model.fit(X_train, y_train)
            
        elif MODEL == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=njobs,
                random_state=42
            )
            
            if AUGMENT == False:
                # Train Random Forest model
                logging.info("Training Random Forest model...")
                model.fit(X_train, y_train)
            
            elif AUGMENT == True:
                # Apply SMOTE to balance training data
                logging.info("Applying SMOTE to balance training set...")
                sm = SMOTE(random_state=seed)
                X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
                
                logging.info(f"Training samples before SMOTE: {len(y_train)}, after SMOTE: {len(y_train_balanced)}")
                
                # Train MLP model on balanced data
                model.fit(X_train_balanced, y_train_balanced)
            
            
        elif MODEL == "XGBOOST":
                    # Compute balanced sample weights
                    # sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

                    # Define the model
                    model = XGBClassifier(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.1,
                        gamma=0,
                        min_child_weight=1,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        reg_alpha=0,
                        reg_lambda=1,
                        objective='multi:softprob',
                        num_class=7,  # Update this if the number of classes changes
                        n_jobs= 2, #njobs,  # Assumes you've defined `njobs` elsewhere
                        random_state=42
                    )
                    #Fit with sample weights
                    model.fit(X_train, y_train)
                    
        elif MODEL == "MLP":
            # Get number of classes
            num_classes = max(valid_classes) + 1
            logging.info(f"Number of classes: {num_classes}")
            
            
            if AUGMENT == False:
                # Train MLP model
                mlp_model = train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size)
            elif AUGMENT == True:
                # Apply SMOTE to balance training data
                logging.info("Applying SMOTE to balance training set...")
                sm = SMOTE(random_state=seed)
                X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
                
                logging.info(f"Training samples before SMOTE: {len(y_train)}, after SMOTE: {len(y_train_balanced)}")
                
                # Train MLP model on balanced data
                mlp_model = train_mlp(X_train_balanced, y_train_balanced, X_val, y_val, num_classes, input_size)
                
            
            
            
            
            # Create a wrapper class for consistency with other models
            class MLPWrapper:
                    def __init__(self, model):
                        self.model = model
                        
                    def predict(self, X):
                        return mlp_predict(self.model, X)
                    
                    def predict_proba(self, X):
                        return mlp_predict_proba(self.model, X) 
            
            model = MLPWrapper(mlp_model)
            
        else:
            raise ValueError(f"Unknown model type: {MODEL}. Use 'LogisticRegression', 'RandomForest', or 'MLP'")
        
        trained_models.append(model)
        
        end = time.process_time()
        elapsed_time = end - start
        logging.info(f"Training completed in {elapsed_time:.2f} seconds")
            


def ensemble_predict_proba(models, features):
    prob_sum = np.zeros((features.shape[0], n_classes))
    for model in models:
        prob_sum += model.predict_proba(features)
    return prob_sum / len(models)


year = new_year


bands_file_path = f"/maps/mcl66/senegal/d-pixel/merged_clipped/{year}_merged_bands_by_doy_clipped.npy"
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
    
def process_chunk(h_start, h_end, w_start, w_end):
    logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")

    # ----- Sentinel-2 -----
    s2_bands = np.load(bands_file_path, mmap_mode='r')[:, h_start:h_end, w_start:w_end, :]
    s2_bands = (s2_bands - S2_BAND_MEAN) / S2_BAND_STD

    # Apply mask (ensure this mask is not leaking labels!)
    #s2_mask = (np.load(mask_file_path)[:, h_start:h_end, w_start:w_end]).squeeze(axis=-1)
    s2_mask = np.load(mask_file_path)[:, h_start:h_end, w_start:w_end]
    s2_mask = np.squeeze(s2_mask) 
    s2_mask = s2_mask[..., np.newaxis]
    s2_bands = s2_bands * s2_mask

    time_steps, h, w, s2_bands_total = s2_bands.shape
    s2_band_chunk = s2_bands.transpose(1, 2, 0, 3).reshape(-1, time_steps * s2_bands_total)
    
    expected_time_steps = TRAIN_TIME_STEPS  # <-- set this to the value from training
    print(f"Expected time steps: {expected_time_steps}, actual time steps: {time_steps}")
    #if time_steps < expected_time_steps:
    #    missing_steps = expected_time_steps - time_steps
    print("s2_band_chunk shape is:", s2_band_chunk.shape)
    pad_shape = (s2_band_chunk.shape[0], 5)
    print("pad_shape is:", pad_shape)
    #s2_band_chunk = np.concatenate([s2_band_chunk, pad_shape], axis=1)
    #if time_steps > expected_time_steps:
    #    s2_band_chunk = s2_band_chunk[:, :772]#:expected_time_steps * s2_bands_total]

    # ----- Sentinel-1 -----
    sar_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD

    time_steps, h, w, bands = sar_chunk.shape
    sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands)

    # ----- Features -----
    print("sar_band_chunk shape is:", sar_band_chunk.shape)
    print("s2_band_chunk shape is:", s2_band_chunk.shape)
    dummy = np.zeros((s2_band_chunk.shape[0], 10), dtype=s2_band_chunk.dtype)
    s2_band_chunk = np.concatenate((s2_band_chunk, dummy), axis=1)
    print("After padding, s2_band_chunk shape is:", s2_band_chunk.shape)
    X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1)
    print("X_chunk shape is:", X_chunk.shape)

    # ----- Labels + Field IDs -----
    y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
    fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()

    # Filter: only valid classes
    valid_mask = np.isin(y_chunk, list(valid_classes))
    X_chunk, y_chunk, fieldid_chunk = (
        X_chunk[valid_mask],
        y_chunk[valid_mask],
        fieldid_chunk[valid_mask],
    )

    # Filter: only fields from this year
    all_mask = np.isin(fieldid_chunk, all_ids)
    X_chunk, y_chunk = X_chunk[all_mask], y_chunk[all_mask]

    return (X_chunk, y_chunk)


# Define chunks for parallel processing
chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
          for h in range(0, H, chunk_size)
          for w in range(0, W, chunk_size)]
logging.info(f"Total chunks: {len(chunks)}")

results = Parallel(n_jobs=njobs)(
    delayed(process_chunk)(h_start, h_end, w_start, w_end)
    for h_start, h_end, w_start, w_end in chunks
)


def safe_vstack(arrays, empty_shape=None):
    filtered = [arr for arr in arrays if arr.size > 0]
    return np.vstack(filtered) if filtered else np.empty(empty_shape or (0,))


def safe_hstack(arrays, empty_shape=None):
    filtered = [arr for arr in arrays if arr.size > 0]
    return np.hstack(filtered) if filtered else np.empty(empty_shape or (0,), dtype=int)


# Infer feature dimension
feature_dim = None  
for res in results:
    if res[0].size > 0:
        feature_dim = res[0].shape[1]
        break

if feature_dim is None:
    raise ValueError("No data found to infer feature dimension!")
logging.info(f"Feature dimension: {feature_dim}")



# Final dataset (single group)
X_all = safe_vstack([res[0] for res in results], empty_shape=(0, feature_dim))
y_all = safe_hstack([res[1] for res in results], empty_shape=(0,))
print("x_all shape", X_all.shape) 

y_probs = ensemble_predict_proba(trained_models, X_all)  # shape: (N, C)
y_pred = np.argmax(y_probs, axis=1)


print(f"Unique y_pred: {np.unique(y_pred)}")
logging.info("Classification Report (Test Set):\n" + classification_report(y_all, y_pred, digits=4))    

end = time.process_time()
cpu_time = end - start

def save_classification_report(y_true, y_pred, filename="classification_report.csv"):
            # Get classification report as a dict and convert to DataFrame (multi-row format)
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})

            # Add the seed and cpu_time columns to every row
            report_df.insert(0, "cpu_time", cpu_time)
            report_df.insert(0, "seed", seed)

            # If file exists, append the new block; else create new file
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, report_df], ignore_index=True)
            else:
                combined_df = report_df

            # Save the full DataFrame back to CSV
            combined_df.to_csv(filename, index=False, float_format='%.4f')

class_report_filename = f'/maps/mcl66/senegal/classification_reports/senegal_raw_classification_report_{mixedyear}_agg_{CLASSIFICATION}.csv'

save_classification_report(y_all, y_pred, class_report_filename)
