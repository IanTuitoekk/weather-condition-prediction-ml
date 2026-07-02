"""
Random Forest Model

"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Configuration:
RANDOM_STATE = 42
CV_FOLDS = 3

# Project paths:
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports" / "RandomForest"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"


def normalize_target_labels(labels: pd.Series) -> pd.Series:
    """Standardize spacing and capitalization in weather labels."""
    return (
        labels.astype("string")
        .str.strip()
        .str.lower()
        .str.title()
    )


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and validate Member 3's processed train/test datasets."""
    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required processed files:\n" + "\n".join(missing_files)
        )

    print("Loading processed datasets...")

    X_train = pd.read_csv(X_TRAIN_PATH, dtype=np.float32)
    X_test = pd.read_csv(X_TEST_PATH, dtype=np.float32)
    y_train_df = pd.read_csv(Y_TRAIN_PATH)
    y_test_df = pd.read_csv(Y_TEST_PATH)

    if y_train_df.shape[1] != 1 or y_test_df.shape[1] != 1:
        raise ValueError("Each target CSV must contain exactly one column.")

    if len(X_train) != len(y_train_df):
        raise ValueError("X_train and y_train contain different row counts.")

    if len(X_test) != len(y_test_df):
        raise ValueError("X_test and y_test contain different row counts.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test do not have identical columns.")

    # Replace any unexpected non-finite values safely.
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = normalize_target_labels(y_train_df.iloc[:, 0])
    y_test = normalize_target_labels(y_test_df.iloc[:, 0])

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Missing target labels were found.")

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Training classes after normalization: {y_train.nunique()}")
    print(f"Testing classes after normalization:  {y_test.nunique()}")

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_datasets()

    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Perform hyperparameter tuning
    print("\nPerforming hyperparameter tuning using RandomizedSearchCV...")
    
    # Restrict to pre-pruned hyperparameter search space to prevent oversized models
    param_dist = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15, 20],        # Limit max_depth to reduce tree size
        "min_samples_split": [10, 20, 50], # Higher split threshold to limit complexity
        "min_samples_leaf": [4, 8, 16],    # Higher leaf size limit to reduce leaf nodes
    }

    rf_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # RandomizedSearchCV to keep runtime reasonable (8 iterations)
    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=8,
        cv=cv,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=1,  # Windows-safe: single process for outer folds, parallel trees
        verbose=1
    )

    start_time = time.time()
    search.fit(X_train, y_train_encoded)
    tuning_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds.")

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_

    print("\nBest parameters:")
    print(best_params)
    print(f"Best CV F1 (weighted) score: {best_cv_score:.4f}")

    # Evaluate final model on test dataset
    print("\nEvaluating final model on test dataset...")
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average="macro", zero_division=0
    )

    print("\nResults")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Save classification report
    report = classification_report(
        y_test_encoded,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "random_forest_classification_report.csv"
    )

    # Confusion matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test_encoded, y_pred, normalize="true")
    size = max(14, len(encoder.classes_) * 0.36)
    fig, ax = plt.subplots(figsize=(size, size))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    display.plot(ax=ax, xticks_rotation=90, include_values=False)
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(
        REPORTS_DIR / "random_forest_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Feature Importance analysis
    print("Generating feature importance plot...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top 15 features
    top_n = 15
    top_indices = indices[:top_n]
    top_features = [X_train.columns[i] for i in top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_importances[::-1], align="center")
    plt.yticks(range(top_n), [top_features[i] for i in range(top_n)][::-1])
    plt.xlabel("Relative Importance")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "random_forest_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save predictions
    predictions = pd.DataFrame({
        "actual": y_test,
        "predicted": encoder.inverse_transform(y_pred),
    })
    predictions.to_csv(
        REPORTS_DIR / "random_forest_predictions.csv",
        index=False,
    )

    # Save model and encoder with compression to reduce file size on disk
    joblib.dump(best_model, MODELS_DIR / "random_forest_model.pkl", compress=3)
    joblib.dump(encoder, MODELS_DIR / "random_forest_label_encoder.pkl", compress=3)

    # Save metrics
    metrics = {
        "model": "Random Forest",
        "best_parameters": best_params,
        "best_cv_score": float(best_cv_score),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "tuning_time_seconds": float(tuning_time)
    }
    with open(REPORTS_DIR / "random_forest_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Markdown report
    report_text = f"""# Member 6 - Random Forest Report

## Best Parameters

{best_params}

## Cross Validation Score (F1 Weighted)

{best_cv_score:.4f}

## Test Performance

- Accuracy: {accuracy:.4f}
- Weighted Precision: {weighted_precision:.4f}
- Weighted Recall: {weighted_recall:.4f}
- Weighted F1: {weighted_f1:.4f}
- Macro Precision: {macro_precision:.4f}
- Macro Recall: {macro_recall:.4f}
- Macro F1: {macro_f1:.4f}

## Deliverables

- Tuned Random Forest model
- Feature importance plot
- Confusion matrix
- Classification report
- Performance metrics
"""
    (REPORTS_DIR / "member6_random_forest_report.md").write_text(report_text)
    print("\nRandom Forest workflow completed successfully.")


if __name__ == "__main__":
    main()
