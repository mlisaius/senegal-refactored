from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from imblearn.over_sampling import SMOTE
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .neural import train_mlp


def train_random_forest(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    njobs: int,
    seed: int,
) -> RandomForestClassifier:
    """Train a RandomForest classifier with 100 estimators."""
    model = RandomForestClassifier(
        n_estimators=100,    # number of trees — balances accuracy vs. training time
        max_depth=None,      # grow trees fully; pruning is handled by min_samples_*
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=njobs,        # parallelise tree building across CPU cores
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    njobs: int,
    seed: int,
) -> LogisticRegression:
    """Train a multinomial LogisticRegression with high regularization (C=1e4)."""
    model = LogisticRegression(
        multi_class="multinomial",  # true multi-class (softmax), not one-vs-rest
        solver="lbfgs",             # memory-efficient quasi-Newton solver
        C=1e4,                      # very weak L2 regularization (high C = less penalty)
        max_iter=100000,            # allow enough iterations for convergence on many classes
        n_jobs=njobs,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    num_classes: int,
    njobs: int,
    seed: int,
) -> XGBClassifier:
    """Train an XGBClassifier with 400 estimators using multi:softprob objective."""
    model = XGBClassifier(
        n_estimators=400,           # more trees than RF because each tree is shallow
        max_depth=3,                # shallow trees reduce overfitting on high-dim features
        learning_rate=0.1,
        gamma=0,                    # no minimum loss-reduction requirement for splits
        min_child_weight=1,
        subsample=0.8,              # row subsampling per tree for variance reduction
        colsample_bytree=1.0,       # use all features (no column subsampling)
        reg_alpha=0,                # no L1 regularization
        reg_lambda=1,               # L2 regularization weight
        objective="multi:softprob", # outputs class probabilities (needed for predict_proba)
        num_class=num_classes,
        n_jobs=2,                   # XGBoost internal parallelism (separate from joblib)
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train: npt.NDArray, y_train: npt.NDArray, seed: int) -> SVC:
    """Train an SVC with probability estimates enabled."""
    # probability=True enables Platt scaling so predict_proba is available for ensemble.
    model = SVC(probability=True, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_model(
    model_name: str,
    X_train: npt.NDArray,
    y_train: npt.NDArray,
    X_val: npt.NDArray,
    y_val: npt.NDArray,
    cfg: dict,
) -> ClassifierMixin:
    """Dispatch to the appropriate trainer and optionally apply SMOTE augmentation."""
    njobs = cfg["njobs"]
    # _current_seed is injected by the training loop; fall back to 42 if absent.
    seed = cfg.get("_current_seed", 42)
    augment = cfg["training"]["augment"]

    if augment:
        # SMOTE generates synthetic minority-class samples so each class is equally
        # represented during training.  Applied before model selection so all models
        # benefit from the same balanced dataset.
        logging.info("Applying SMOTE to balance training set...")
        sm = SMOTE(random_state=seed)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logging.info(f"Training samples after SMOTE: {len(y_train)}")

    # Route to the appropriate training function based on the model name string.
    if model_name == "RandomForest":
        return train_random_forest(X_train, y_train, njobs, seed)
    elif model_name == "LogisticRegression":
        return train_logistic_regression(X_train, y_train, njobs, seed)
    elif model_name == "XGBOOST":
        # XGBoost needs to know the number of output classes at construction time.
        num_classes = len(set(y_train))
        return train_xgboost(X_train, y_train, num_classes, njobs, seed)
    elif model_name == "SVM":
        return train_svm(X_train, y_train, seed)
    elif model_name == "MLP":
        # MLP also needs num_classes and input_size for layer construction.
        num_classes = int(max(y_train))
        input_size = X_train.shape[1]
        return train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size, cfg)
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            "Choose from: RandomForest, LogisticRegression, XGBOOST, SVM, MLP."
        )


def ensemble_predict_proba(models: list, X: npt.NDArray) -> npt.NDArray:
    """Average predicted probabilities across all models."""
    prob_sum = None
    for model in models:
        p = model.predict_proba(X)
        if prob_sum is None:
            # Initialise accumulator with the first model's output shape.
            prob_sum = p.copy()
        else:
            # Accumulate probability vectors element-wise.
            prob_sum += p
    # Divide by the number of models to get the mean probability vector per pixel.
    return prob_sum / len(models)
