"""
Random Forest baseline for EUR/USD forecasting.

Flattens the sequence into a single feature vector and trains separate
RF models for direction (classification) and returns/volatility (regression).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict


def train_rf_fold(
    train_X: np.ndarray, train_y: Dict,
    val_X: np.ndarray, val_y: Dict,
    test_X: np.ndarray, test_y: Dict,
    fold_id: int = 0,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
) -> Dict:
    """
    Train Random Forest on a single fold.

    Flattens (N, T, F) → (N, T*F) feature vectors.
    Returns same format as TGT trainer fold results.
    """
    # Flatten sequences
    tr_X = train_X.reshape(len(train_X), -1)
    va_X = val_X.reshape(len(val_X), -1)
    te_X = test_X.reshape(len(test_X), -1)

    # Direction classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1, class_weight="balanced",
    )
    clf.fit(tr_X, train_y["direction"])

    # Return regressor
    reg_ret = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1,
    )
    reg_ret.fit(tr_X, train_y["return"])

    # Volatility regressor
    reg_vol = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1,
    )
    reg_vol.fit(tr_X, train_y["volatility"])

    # Predict on test
    dir_proba = clf.predict_proba(te_X)
    # Ensure all 3 classes are represented in output
    if dir_proba.shape[1] < 3:
        full_proba = np.zeros((len(te_X), 3))
        for i, c in enumerate(clf.classes_):
            full_proba[:, c] = dir_proba[:, i]
        dir_proba = full_proba

    ret_pred = reg_ret.predict(te_X).astype(np.float32)
    vol_pred = np.maximum(reg_vol.predict(te_X), 0).astype(np.float32)

    # Convert probabilities to logit-like scores for compatibility
    dir_logits = np.log(np.maximum(dir_proba, 1e-8)).astype(np.float32)

    dir_acc = (dir_logits.argmax(1) == test_y["direction"]).mean()

    # Val accuracy for reporting
    va_pred = clf.predict(va_X)
    val_acc = (va_pred == val_y["direction"]).mean()

    print(f"      📊 RF Fold {fold_id} | val_acc={val_acc:.3f} | test_acc={dir_acc:.3f}")

    return {
        "fold_id": fold_id,
        "best_epoch": 0,
        "best_val_loss": 1.0 - val_acc,
        "test_losses": {"total": 1.0 - dir_acc},
        "test_direction_accuracy": float(dir_acc),
        "test_predictions": {
            "direction": dir_logits,
            "return": ret_pred,
            "volatility": vol_pred,
        },
        "test_targets": {
            "direction": test_y["direction"],
            "return": test_y["return"],
            "volatility": test_y["volatility"],
        },
    }
