"""
models/neural.py — PyTorch MLP model and sklearn-compatible wrapper for Senegal classification.

Provides:
  - FocalLoss: class-imbalance-aware loss function
  - MLP: 3-layer fully-connected network with BatchNorm and Dropout
  - MLPWrapper: sklearn-compatible predict / predict_proba interface
  - train_mlp: full training loop with early stopping and LR scheduling
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Select GPU if available, otherwise fall back to CPU.  All tensors and the
# model are moved to this device so the training loop works unchanged either way.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in multi-class classification.

    Focal loss down-weights easy (well-classified) examples so the model
    concentrates gradient signal on hard / minority-class samples.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha: float = 1, gamma: float = 2, reduction: str = "mean"
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha   # overall scaling factor
        self.gamma = gamma   # focusing parameter: higher = more focus on hard examples
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute log-probabilities once and reuse for both pt and log_pt.
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # One-hot encode targets so we can select the probability of the true class.
        targets_one_hot = nn.functional.one_hot(
            targets, num_classes=inputs.size(1)
        ).float()

        # pt = probability assigned to the correct class for each sample.
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)

        # (1 - pt)^gamma is the focusing term: near 0 for easy samples (pt ≈ 1),
        # near 1 for hard samples (pt ≈ 0).
        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # per-sample losses, shape (batch,)


class MLP(nn.Module):
    """3-layer MLP with BatchNorm, ReLU, and Dropout for land cover classification."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],  # expected to have exactly 3 elements
        num_classes: int,
        dropout_rate: float = 0.4,
    ) -> None:
        super(MLP, self).__init__()

        # Each hidden layer follows the pattern: Linear → BatchNorm → ReLU → Dropout.
        # BatchNorm stabilises training on high-dimensional satellite feature vectors.
        # Dropout provides regularisation to reduce overfitting.
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Final linear layer maps to raw class logits (no softmax — handled by loss).
        self.output = nn.Linear(hidden_sizes[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x


class MLPWrapper:
    """Sklearn-compatible wrapper exposing predict and predict_proba for a trained MLP."""

    def __init__(self, model: MLP, batch_size: int = 1024) -> None:
        self.model = model
        # Store batch_size on the instance so it can be configured from config.yaml
        # and reused consistently across predict and predict_proba.
        self.batch_size = batch_size

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        # Switch to eval mode: disables BatchNorm running-stat updates and Dropout.
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        predictions = []
        with torch.no_grad():  # no gradient tracking needed at inference time
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                outputs = self.model(batch_X)
                # argmax gives the 0-based predicted class; add 1 to match 1-based labels.
                _, preds = torch.max(outputs, 1)
                predictions.append((preds + 1).cpu().numpy())
        return np.concatenate(predictions)

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        # Same batched inference pattern as predict, but return softmax probabilities
        # instead of hard class labels.  Used by ensemble_predict_proba in classifiers.py.
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i : i + self.batch_size]
                outputs = self.model(batch_X)
                probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        return np.vstack(probs)  # (n_samples, n_classes)


def train_mlp(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_val: npt.NDArray,
    y_val: npt.NDArray,
    num_classes: int,
    input_size: int,
    cfg: dict,
) -> MLPWrapper:
    """Train an MLP with early stopping and LR scheduling; return an MLPWrapper."""
    # Unpack all MLP hyperparameters from the config in one place for readability.
    batch_size = cfg["mlp"]["batch_size"]
    learning_rate = cfg["mlp"]["learning_rate"]
    num_epochs = cfg["mlp"]["num_epochs"]
    patience = cfg["mlp"]["patience"]         # early stopping patience in epochs
    hidden_sizes = cfg["mlp"]["hidden_sizes"]
    dropout_rate = cfg["mlp"]["dropout_rate"]
    focal_alpha = cfg["mlp"]["focal_loss"]["alpha"]
    focal_gamma = cfg["mlp"]["focal_loss"]["gamma"]

    logging.info(f"Starting MLP training with input size: {input_size}")

    # Convert numpy arrays to PyTorch tensors.
    # Labels are shifted by -1 to be 0-based (PyTorch CrossEntropy convention).
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train - 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val - 1)

    # shuffle=True for training ensures different mini-batch compositions each epoch.
    # shuffle=False for validation ensures reproducible evaluation order.
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False
    )

    model = MLP(input_size, hidden_sizes, num_classes, dropout_rate).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Halve the learning rate when val_loss stalls for 3 epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    early_stop_counter = 0  # counts consecutive epochs without val_loss improvement
    best_model_state = None  # stores the weights of the best checkpoint

    logging.info(f"MLP model architecture:\n{model}")

    for epoch in range(num_epochs):
        # --- Training pass ---
        model.train()
        train_loss = 0.0
        correct = total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()   # compute gradients
            optimizer.step()  # update weights

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # Normalise accumulated loss by the number of batches.
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # --- Validation pass ---
        model.eval()
        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():  # no gradient computation needed for evaluation
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        # Pass val_loss to the scheduler; it will reduce LR if no improvement.
        scheduler.step(val_loss)
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save model weights whenever a new best val_loss is found.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Deep-copy the state dict so further training doesn't overwrite it.
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0
            logging.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1

        # Stop early if val_loss hasn't improved for `patience` consecutive epochs.
        if early_stop_counter >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is None:
        raise RuntimeError(
            "MLP training produced no valid model state (training may have failed immediately)."
        )

    # Restore the best checkpoint before wrapping — ensures inference uses the best weights.
    model.load_state_dict(best_model_state)
    # Pass batch_size through to MLPWrapper so inference uses the same chunk size as training.
    return MLPWrapper(model, batch_size=batch_size)
