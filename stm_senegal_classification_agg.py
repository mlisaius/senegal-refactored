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

year = 2021

TRAINING_RATIO = 0.7
MODEL = "RandomForest"  # Options: "LogisticRegression", "RandomForest", or "MLP", "XGBOOST", "SVM"
CLASSIFICATION = "maincrop"  # Options: "landcover", "maincrop"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
SAVE = "yes" # Save model prediction map, "yes" or "no"
SAMPLING = "bypercentage_count"  # "bypercentage", "bypercentage_count", "bycount"  #sampling strategy 
WHOLEMAP = True  # If True, process both the labels and the whole map at once, otherwise process on labels
REPORT = True # If True, save classification report and confusion matrix to CSV
AUGMENT = False  # If True, apply SMOTE to balance training data
NUM_SEEDS = 2
REMAP2021 = True  # If True, remap labels for 2021 to reduce pasture and natural vegetation to 0


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
bands_file_0 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group0_chunk.npy", mmap_mode = 'r')
bands_file_1 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group1_chunk.npy", mmap_mode = 'r')
bands_file_2 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group2_chunk.npy", mmap_mode = 'r')
bands_file_3 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group3_chunk.npy", mmap_mode = 'r')
bands_file_4 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group4_chunk.npy", mmap_mode = 'r')
bands_file_5 = np.load(f"/maps/mcl66/senegal/d-pixel/stms/stms_{year}_group5_chunk.npy", mmap_mode = 'r')


#label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_landcover_labels.npy"
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



#Class names for visualization
if CLASSIFICATION == "landcover":
    outfolder = "landcoverclassification"
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
elif CLASSIFICATION == "maincrop":
    outfolder = "cropclassification"
    # remap all labels to 0 for non crop and 1 for crop
    remapping = {
        0: 0,  # Built-up surface
        1: 0,  # Bare soil
        2: 0,  # Water body
        3: 0,  # Wetland
        4: 1,  # Cropland
        5: 0,  # Shrub land
        #6: 0   # Pasture
    }

    agg_pred_map = np.load(f'/maps/mcl66/senegal/landcoverclassification/senegal_tessera_prediction_map_whole_{year}_15agg.npy')
    agg_pred_map_mask = np.vectorize(lambda x: remapping.get(x, 0))(agg_pred_map)   
    
    classcode = "maincrop_code"
    label_file_path = f"/maps/mcl66/senegal/label_rasters/raster_{year}_clipped_remapped_crop_labels.npy"
    class_names = [
        "Maize", # 1,
        "Rice", # 2,
        "Sorghum", # 3,
        "Millet", # 4,
        "Groundnut", # 5,
        "Sesame", # 6,
        "Cotton" # 7
    ]

if REMAP2021:
    print("REMAP2021 is True, reducing labels for 2021.")
    if year == 2021:
        REDUCE_LABELS = True  # For 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
        remapped_labels = "_remapped"
        print("year is 2021, reducing labels to 0 for pasture and natural vegetation.")
    else:
        REDUCE_LABELS = False
        remapped_labels = ""

trained_models = []


#seeds = [1]
#seeds = [1, 2, 3, 4, 5]  # List of seeds for reproducibility
seeds = list(range(1, 1 + NUM_SEEDS))


for MODEL in ["RandomForest", "MLP"]: #xgboost
    for seed in seeds:       
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        if MODEL in ["XGBOOST", "MLP"]:
            VAL_TEST_SPLIT_RATIO = 1/4  # Validation to test set split ratio
        else:
            VAL_TEST_SPLIT_RATIO = 1/20  # Validation to test set split ratio

        
            
        start = time.process_time()
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
            y_train_tensor = torch.LongTensor(y_train - 1)  # Shift labels from 1-18 to 0-17

            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val - 1)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Initialize model
            hidden_sizes = [2048, 1024, 512]  # Three hidden layer sizes
            model = MLP(input_size, hidden_sizes, num_classes).to(DEVICE)
            
            # Define loss function and optimizer
            #criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)  # Weighted loss
            criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')  # Focal loss
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
                    predictions.append((preds+1).cpu().numpy())
            
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


        # ----------------- Data loading and preprocessing -----------------
        logging.info(f"Training ratio: {TRAINING_RATIO}")
        logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
        logging.info(f"Selected model: {MODEL}")
        logging.info("Loading labels and field IDs...")

        labels = (np.load(label_file_path).astype(np.int64)).squeeze()
        print(label_file_path)
        
        field_ids = np.load(field_id_file_path).squeeze()
        
        if REDUCE_LABELS:
            # for 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
            logging.info("Reducing labels...")
            # remap labels to 0 for pasture (7) and natural vegetation (8)
            mask = np.isin(labels, [7, 8])
            labels[mask] = 0  # Set pasture and natural vegetation to 0
            field_ids[mask] = 0  # Set corresponding field IDs to 0
                    
        if MODEL == "XGBOOST":
            labels -= 1
        H, W = labels.shape
        logging.info(f"Data dimensions: {H}x{W}")


        # Select valid classes
        logging.info("Identifying valid classes...")
        class_counts = Counter(labels.ravel())
        valid_classes = {cls for cls, count in class_counts.items() if count >= 2}  
        if MODEL == "XGBOOST":
            valid_classes.discard(-1)
        else:
            valid_classes.discard(0)
        logging.info(f"Valid classes: {sorted(valid_classes)}")

        # ----------------- Train/validation/test set split -----------------
        # ----------------- Train/validation/test set split -----------------
        logging.info("Splitting data into train/val/test sets...")
        fielddata_df = pd.read_csv(updated_fielddata_path)
        fielddata_df = fielddata_df[fielddata_df['Year'] == year]
        
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
                remaining_rows = remaining_rows.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle remaining

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
                rows_sncode = rows_sncode.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

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

        if MODEL != "XGBOOST":
            valid_label_mask = (labels > 0)
        elif MODEL == "XGBOOST":
            valid_label_mask = (labels >= 0)

        # Report class distribution by pixel after split
        y = labels  # Don't prefilter

        for split_id, name in [(1, "Train"), (2, "Val"), (3, "Test")]:
            mask = (train_test_mask == split_id)
            y_split = y[mask]
            print(f"{name} pixel class distribution:")
            print(pd.Series(y_split).value_counts())


        # ----------------- Chunking -----------------
        def process_chunk(h_start, h_end, w_start, w_end):
            logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
            
            # Load data for feature extraction (only once per chunk)
            # Load data for feature extraction (only once per chunk)
            bands_0 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_1 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_2 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_3 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_4 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_5 = bands_file_0[h_start:h_end, w_start:w_end, :]
            
            
            bands_all = np.concatenate([bands_0, bands_1, bands_2, bands_3, bands_4, bands_5], axis=2)

                    
            h, w, s2_bands_total = bands_all.shape
            s2_band_chunk = bands_all.reshape(-1, s2_bands_total)  # (h*w, bands)
            
            sar_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
            
            sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD

            time_steps, h, w, bands = sar_chunk.shape
            sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
            # Concatenate s2 and s1
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
            delayed(process_chunk)(h_start, h_end, w_start, w_end)
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

        print(f"Unique y training: {np.unique(y_train)}")
        print(f"Unique y val: {np.unique(y_val)}")
        print(f"Unique y test: {np.unique(y_test)}")

        if MODEL != "XGBOOST":
            # Remove class 0 (background) from training data
            logging.info("Removing class 0 (background) from training data...")
            indices_to_remove = np.where(y_train == 0)[0]
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            X_train = np.delete(X_train, indices_to_remove, axis=0)

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
            num_classes = max(valid_classes) 
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


# ----------------- Classification Map Generation -----------------
logging.info("\nGenerating classification maps...")

# Generate a color palette (using tab20 and extending if needed)
def get_color_palette(n_classes):
    """Generate a color palette for classification maps."""
    # Start with tab20 which has 20 distinct colors
    base_cmap = plt.cm.get_cmap('tab20')
    colors = [base_cmap(i) for i in range(20)]
    
    # If we need more colors, add from other colormaps
    if n_classes > 20:
        extra_cmap = plt.cm.get_cmap('tab20b')
        colors.extend([extra_cmap(i) for i in range(n_classes - 20)])
    
    # Return only the number of colors we need
    return colors[:n_classes]

# Setup for visualization
# Add 1 to max class for background (0)
max_class = max(valid_classes) 
n_classes = len(valid_classes)
logging.info(f"Creating color mapping for {n_classes} classes (1-{max_class})")

# Generate color palette
colors = get_color_palette(max_class + 1)  # +1 for background class (0)
# Set background (0) to white
colors[0] = (1, 1, 1, 1)  # White for background

# Create colormap
cmap = ListedColormap(colors)

class_colors = [
    '#000000',     # background (black, or can be transparent if you prefer)
    'grey',        # Built-up surface
    'purple', # Bare soil
    'saddlebrown',      # Water body
    'blue',       # Wetland
    'darkgreen',      # Cropland
    'purple',  # Shrub land
    'green',   # Pasture
    'yellow'  # Natural vegetation
]

# Create colormap
cmap = ListedColormap(class_colors)

# Define a plotting function for consistent formatting
def plot_classification_map(data, title, cmap, class_names, save_path, figsize=(12, 10)):
    """Create a nicely formatted classification map without colorbar."""
    plt.figure(figsize=figsize, dpi=300)
    
    # Set up the plot with publication quality
    plt.rcParams.update({
        'font.family': 'sans-serif',  # Use a generic font family available everywhere
        'font.size': 12,
        'axes.linewidth': 1.5
    })
    
    # Plot the data
    im = plt.imshow(data, cmap=cmap, interpolation='nearest')
            
    # Create a legend with class names
    if class_names:
        # Get the number of unique classes in the data
        unique_classes = sorted(np.unique(data))
        # Filter out 0 if it's background
        if 0 in unique_classes and len(unique_classes) > 1:
            unique_classes = sorted([c for c in unique_classes if c > 0])
        
        print(unique_classes)
        # Create legend patches for each class
        legend_patches = []
        for cls in unique_classes:
            if cls == 0:
                continue  # Skip background
            if cls <= len(class_names):
                # Use class color from colormap
                color = cmap(cls / max(unique_classes))
                label = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
                legend_patches.append(mpatches.Patch(color=color, label=label))
        
        # Add legend outside the plot with larger text and make it more prominent
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), 
                loc='upper left', fontsize=14, frameon=True, fancybox=True, 
                shadow=True, title="Classes", title_fontsize=15)
    
    # Add title and style adjustments
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved classification map to {save_path}")



# ----------------- Getting accuracy metrics -----------------
logging.info("Generating prediction map - this may take some time...")
labels = (np.load(label_file_path).astype(np.int64)).squeeze()

if REDUCE_LABELS:
    # for 2021, reduce labels to 0 for pasture (7) and natural vegetation (8)
    logging.info("Reducing labels...")
    # remap labels to 0 for pasture (7) and natural vegetation (8)
    mask = np.isin(labels, [7, 8])
    labels[mask] = 0  # Set pasture and natural vegetation to 0
    field_ids[mask] = 0  # Set corresponding field IDs to 0

class_counts = Counter(labels.ravel())
valid_classes = {cls for cls, count in class_counts.items() if count >= 2} 
valid_classes.discard(0) 
print(f"Valid classes: {sorted(valid_classes)} for training")

logging.info("Splitting data into train/val/test sets...")
fielddata_df = pd.read_csv(updated_fielddata_path)
fielddata_df = fielddata_df[fielddata_df['Year'] == year]

# Shuffle the DataFrame in-place before sampling
fielddata_df = fielddata_df.sample(frac=1, random_state=seed).reset_index(drop=True)

area_summary = fielddata_df.groupby(classcode)['Area_ha'].sum().reset_index()
count_summary = fielddata_df.groupby(classcode).size().reset_index(name='count')                                     


ids = fielddata_df["Id"].unique()
logging.info(f"Unique field IDs: {len(ids)}")


def process_chunk(h_start, h_end, w_start, w_end):
    logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
    
    # Load data for feature extraction (only once per chunk)
    # Load data for feature extraction (only once per chunk)
    bands_0 = bands_file_0[h_start:h_end, w_start:w_end, :]
    bands_1 = bands_file_0[h_start:h_end, w_start:w_end, :]
    bands_2 = bands_file_0[h_start:h_end, w_start:w_end, :]
    bands_3 = bands_file_0[h_start:h_end, w_start:w_end, :]
    bands_4 = bands_file_0[h_start:h_end, w_start:w_end, :]
    bands_5 = bands_file_0[h_start:h_end, w_start:w_end, :]
    
    
    bands_all = np.concatenate([bands_0, bands_1, bands_2, bands_3, bands_4, bands_5], axis=2)

            
    h, w, s2_bands_total = bands_all.shape
    s2_band_chunk = bands_all.reshape(-1, s2_bands_total)  # (h*w, bands)
    
    sar_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
    
    sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD

    time_steps, h, w, bands = sar_chunk.shape
    sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
    # Concatenate s2 and s1
    X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1)  # (h*w, time_steps*bands*2)

    
    y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
    fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()
    
    # Filter valid data
    valid_mask = np.isin(y_chunk, list(valid_classes))
    X_chunk, y_chunk, fieldid_chunk = X_chunk[valid_mask], y_chunk[valid_mask], fieldid_chunk[valid_mask]
    
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


X_all = safe_vstack([res[0] for res in results], empty_shape=(0, feature_dim))
y_all = safe_hstack([res[1] for res in results], empty_shape=(0,))

print(f"Unique y all: {np.unique(y_all)}")
print("Final dataset shape:", X_all.shape, y_all.shape)

y_probs = ensemble_predict_proba(trained_models, X_all)  # shape: (N, C)
y_pred = np.argmax(y_probs, axis=1)
y_pred += 1  # Adjust for 1-based indexing if needed
print(f"Unique y_pred: {np.unique(y_pred)}")
logging.info("Classification Report (Test Set):\n" + classification_report(y_all, y_pred, digits=4))    

def save_classification_report(y_true, y_pred, filename="classification_report.csv"):
            # Get classification report as a dict and convert to DataFrame (multi-row format)
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "label"})

            # Add the seed and cpu_time columns to every row
            report_df.insert(0, "cpu_time", 0)
            report_df.insert(0, "seed", 0)

            # If file exists, append the new block; else create new file
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, report_df], ignore_index=True)
            else:
                combined_df = report_df

            # Save the full DataFrame back to CSV
            combined_df.to_csv(filename, index=False, float_format='%.4f')

class_report_filename = f'/maps/mcl66/senegal/classification_reports/senegal_stm_classification_report_{year}_agg_landcover.csv'

save_classification_report(y_all, y_pred, class_report_filename)


if SAVE == "yes":
        # Create prediction map
        pred_map_whole = np.zeros_like(labels)

        # Optimized batch prediction for a whole chunk
        def batch_predict_chunk(h_start, h_end, w_start, w_end):
            """Process and predict a chunk of the image more efficiently."""
            # Create mask for valid classes in this chunk
            chunk_labels = labels[h_start:h_end, w_start:w_end]
            chunk_fieldids = field_ids[h_start:h_end, w_start:w_end]
            
            # Create empty prediction array for this chunk
            chunk_pred = np.zeros_like(chunk_labels)
            
            
            predict_mask = np.ones_like(chunk_labels, dtype=bool)

            # If there are no pixels to predict, return early
            if not np.any(predict_mask):
                return h_start, h_end, w_start, w_end, chunk_pred
            
            # Get coordinates of pixels that need prediction
            h_indices, w_indices = np.where(predict_mask)
            
            # Load data for feature extraction (only once per chunk)
            # Load data for feature extraction (only once per chunk)
            bands_0 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_1 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_2 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_3 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_4 = bands_file_0[h_start:h_end, w_start:w_end, :]
            bands_5 = bands_file_0[h_start:h_end, w_start:w_end, :]
            
            bands_all = np.concatenate([bands_0, bands_1, bands_2, bands_3, bands_4, bands_5], axis=2)
            
            
            sar_data = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
            
            # Normalize only the original bands
            sar_data = (sar_data - S1_BAND_MEAN) / S1_BAND_STD
            
            # Batch size for processing within chunk
            batch_size = 1000
            for i in range(0, len(h_indices), batch_size):
                batch_h = h_indices[i:i+batch_size]
                batch_w = w_indices[i:i+batch_size]
                
                # Extract features for this batch of pixels
                batch_features = []
                for j in range(len(batch_h)):
                    h_idx, w_idx = batch_h[j], batch_w[j]
                    
                    # S2 feature extraction
                    s2_pixel = bands_all[h_idx, w_idx, :]
                    #s2_norm = (s2_pixel - S2_BAND_MEAN) / S2_BAND_STD
                    s2_features = s2_pixel.reshape(-1)
                    
                    # S1 feature extraction
                    sar_pixel = sar_data[:, h_idx, w_idx]
                    sar_features = sar_pixel.reshape(-1)
                    
                    # Combine features
                    #print(f"shape of s2_features {s2_features.shape} and shape of sar_features {sar_features.shape}")

                    features = np.concatenate((s2_features, sar_features))

                    batch_features.append(features)
                
                # Convert to numpy array
                batch_features = np.array(batch_features)
                
                # Batch prediction
                batch_preds = model.predict(batch_features)
                
                # Place predictions into chunk
                for j in range(len(batch_h)):
                    h_idx, w_idx = batch_h[j], batch_w[j]
                    chunk_pred[h_idx, w_idx] = batch_preds[j]
            
            return h_start, h_end, w_start, w_end, chunk_pred

        # Define chunks for parallel processing of prediction map
        pred_chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
                    for h in range(0, H, chunk_size)
                    for w in range(0, W, chunk_size)]

        # Process prediction map in parallel
        logging.info("Processing prediction map in parallel (optimized)...")

        pred_results_whole = Parallel(n_jobs=njobs)(
            delayed(batch_predict_chunk)(h_start, h_end, w_start, w_end)
            for h_start, h_end, w_start, w_end in pred_chunks
        )

            # Combine prediction results
        for h_start, h_end, w_start, w_end, chunk_pred in pred_results_whole:
            pred_map_whole[h_start:h_end, w_start:w_end] = chunk_pred

        # 2. Model Prediction Map
        logging.info("Saving model prediction classification map...")

        def convert_npy_to_tiff(npy, ref_tiff_path, output_path, downsample_rate=1):
            # Load npy data, assuming shape (H, W) or (H, W, C)
            data = npy
            
            if data.dtype == np.int64:
                # Convert int64 to uint8 if necessary
                print("Converting int64 data to uint8...")
                data = data.astype(np.uint8)    
                
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
                'dtype': data.dtype,
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

            print(f"✅ Saved GeoTIFF to {output_path}")
            print(f"Resolution: {10 * downsample_rate} m")


        if SAVE == "yes":
            print("Saving whole map prediction classification map...")
            plot_classification_map(
                pred_map_whole, 
                f"{MODEL.lower()} Classification Predictions", 
                cmap, 
                class_names, 
                f"/maps/mcl66/senegal/{outfolder}/senegal_stm_prediction_map_whole_{year}{remapped_labels}_15agg.png"
            )
            # Save the prediction map as a numpy file
            np.save(f"/maps/mcl66/senegal/{outfolder}/senegal_stm_prediction_map_whole_{year}{remapped_labels}_15agg.npy", pred_map_whole)
            
            # Convert the prediction map to TIFF format
            output_path = f"/maps/mcl66/senegal/{outfolder}/senegal_stm_prediction_map_whole_{year}{remapped_labels}_15agg.tiff"
            ref_tiff_path = f"/maps/mcl66/senegal/representations/2018_representation_map_10m_utm28n_scales_clipped.tiff"
            convert_npy_to_tiff(pred_map_whole, ref_tiff_path, output_path, downsample_rate=1)


        # Generate a composite map that shows the differences between prediction and ground truth
        logging.info("Generating prediction difference map...")








