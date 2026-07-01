"""
Logistic Regression Model

Loads Member 3's processed datasets, trains and evaluates a multiclass
Logistic Regression model, performs stratified cross-validation, creates
evaluation plots, and saves all outputs required for the project.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize


# Configuration:

RANDOM_STATE = 42
CV_FOLDS = 3
MAX_ITERATIONS = 200

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Project paths:

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# Data loading and validation:

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

    # float32 reduces memory use while remaining adequate for this model
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

# Cross-validation:

def perform_cross_validation(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[np.ndarray, list[str]]:
    """
    Run stratified cross-validation.

    Classes with fewer observations than the number of folds are excluded
    from cross-validation only. The final model is still fitted on the full
    training set.
    """
    class_counts = pd.Series(y_train_encoded).value_counts().sort_index()
    valid_class_ids = class_counts[class_counts >= CV_FOLDS].index.to_numpy()
    rare_class_ids = class_counts[class_counts < CV_FOLDS].index.to_numpy()

    cv_mask = np.isin(y_train_encoded, valid_class_ids)
    X_cv = X_train.loc[cv_mask]
    y_cv = y_train_encoded[cv_mask]

    excluded_classes = [
        str(label_encoder.classes_[class_id]) for class_id in rare_class_ids
    ]

    print("\nCross-validation information")
    print("-" * 40)
    print(f"Folds: {CV_FOLDS}")
    print(f"Rows used: {len(y_cv)}")
    print(f"Classes used: {len(np.unique(y_cv))}")

    if excluded_classes:
        print("Classes excluded from CV because they had too few examples:")
        for class_name in excluded_classes:
            print(f"  - {class_name}")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # n_jobs=1 is more reliable on Windows and avoids duplicating the
    # large training matrix across multiple worker processes.
    scores = cross_val_score(
        model,
        X_cv,
        y_cv,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1,
    )

    return scores, excluded_classes

# Visualizations:

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
) -> Path:
    """Save a normalized multiclass confusion matrix."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    size = max(14, len(class_names) * 0.36)
    fig, ax = plt.subplots(figsize=(size, size))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format=None,
        include_values=False,
        colorbar=True,
    )

    ax.set_title("Normalized Confusion Matrix — Logistic Regression", pad=18)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()

    output_path = REPORTS_DIR / "logistic_regression_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> tuple[Path, float]:
    """Save a micro-average one-vs-rest ROC curve."""
    y_binary = label_binarize(
        y_true,
        classes=np.arange(number_of_classes),
    )

    fpr, tpr, _ = roc_curve(y_binary.ravel(), probabilities.ravel())
    micro_auc = roc_auc_score(
        y_binary,
        probabilities,
        average="micro",
        multi_class="ovr",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"Micro-average AUC = {micro_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curve — Logistic Regression")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = REPORTS_DIR / "logistic_regression_roc_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path, float(micro_auc)


def save_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> tuple[Path, float]:
    """Save a micro-average multiclass Precision-Recall curve."""
    y_binary = label_binarize(
        y_true,
        classes=np.arange(number_of_classes),
    )

    precision, recall, _ = precision_recall_curve(
        y_binary.ravel(),
        probabilities.ravel(),
    )
    micro_ap = average_precision_score(
        y_binary,
        probabilities,
        average="micro",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"Micro-average AP = {micro_ap:.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curve — Logistic Regression")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = REPORTS_DIR / "logistic_regression_precision_recall_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path, float(micro_ap)

# Report creation:

def save_markdown_report(
    metrics: dict,
    excluded_cv_classes: list[str],
) -> Path:
    """Create the required Member 4 Markdown report."""
    excluded_text = (
        ", ".join(excluded_cv_classes)
        if excluded_cv_classes
        else "None"
    )

    report = f"""# Member 4 – Logistic Regression Report

## Objective

Train and evaluate a multiclass Logistic Regression model for predicting
weather conditions using the processed features produced by Member 3.

## Dataset

- Training rows: {metrics['training_rows']:,}
- Testing rows: {metrics['testing_rows']:,}
- Number of processed features: {metrics['number_of_features']}
- Number of target classes: {metrics['number_of_classes']}

## Model Configuration

- Algorithm: Logistic Regression
- Solver: {metrics['solver']}
- Maximum iterations: {metrics['maximum_iterations']}
- Class weighting: {metrics['class_weight']}
- Random state: {RANDOM_STATE}

The target labels were standardized for capitalization and surrounding
whitespace before training. Balanced class weights were used because the
weather-condition classes were highly imbalanced.

## Cross-Validation

A {CV_FOLDS}-fold Stratified K-Fold cross-validation procedure was used.

- Fold weighted F1 scores: {metrics['cross_validation_f1_scores']}
- Mean weighted F1 score: {metrics['mean_cross_validation_f1']:.4f}
- Standard deviation: {metrics['cross_validation_f1_std']:.4f}
- Classes excluded from cross-validation: {excluded_text}

## Test-Set Performance

- Accuracy: {metrics['accuracy']:.4f}
- Weighted precision: {metrics['weighted_precision']:.4f}
- Weighted recall: {metrics['weighted_recall']:.4f}
- Weighted F1-score: {metrics['weighted_f1_score']:.4f}
- Macro precision: {metrics['macro_precision']:.4f}
- Macro recall: {metrics['macro_recall']:.4f}
- Macro F1-score: {metrics['macro_f1_score']:.4f}
- Micro ROC-AUC: {metrics['micro_roc_auc']:.4f}
- Micro average precision: {metrics['micro_average_precision']:.4f}

## Generated Deliverables

- Trained Logistic Regression model
- Label encoder
- Classification report
- Normalized confusion matrix
- Multiclass ROC curve
- Multiclass Precision-Recall curve
- JSON metrics file
- Test-set predictions

## Conclusion

The model provides a linear multiclass baseline for comparison with the
Decision Tree, Random Forest, boosting, and ensemble models developed by
the other team members. Weighted and macro metrics are both reported
because the target classes are imbalanced.
"""

    output_path = REPORTS_DIR / "member4_logistic_report.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path

# Main workflow:

def main() -> None:
    X_train, X_test, y_train, y_test = load_datasets()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Remove test rows whose labels were never present in training.
    known_test_mask = y_test.isin(label_encoder.classes_)
    if not known_test_mask.all():
        unknown_classes = sorted(y_test.loc[~known_test_mask].unique())
        print("\nWarning: removing unseen test classes:")
        for class_name in unknown_classes:
            print(f"  - {class_name}")

        X_test = X_test.loc[known_test_mask].reset_index(drop=True)
        y_test = y_test.loc[known_test_mask].reset_index(drop=True)

    y_test_encoded = label_encoder.transform(y_test)
    number_of_classes = len(label_encoder.classes_)

    print(f"\nFinal number of target classes: {number_of_classes}")

    model = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=MAX_ITERATIONS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Cross-validation:
    print("\nPerforming cross-validation...")
    cv_start = time.perf_counter()

    cv_scores, excluded_cv_classes = perform_cross_validation(
        model,
        X_train,
        y_train_encoded,
        label_encoder,
    )

    cv_time = time.perf_counter() - cv_start

    print(f"CV weighted F1 scores: {np.round(cv_scores, 4)}")
    print(f"Mean CV weighted F1: {cv_scores.mean():.4f}")
    print(f"CV standard deviation: {cv_scores.std():.4f}")
    print(f"Cross-validation time: {cv_time:.2f} seconds")

    # Final training:
    print("\nTraining final Logistic Regression model...")
    train_start = time.perf_counter()
    model.fit(X_train, y_train_encoded)
    training_time = time.perf_counter() - train_start

    print(f"Training completed in {training_time:.2f} seconds.")
    print(f"Iterations used: {int(np.max(model.n_iter_))}")

    if np.max(model.n_iter_) >= MAX_ITERATIONS:
        print(
            "Warning: the model reached the iteration limit. "
            "The results were still saved."
        )

    # Predictions:
    print("\nGenerating test predictions...")
    prediction_start = time.perf_counter()
    y_pred_encoded = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    prediction_time = time.perf_counter() - prediction_start

    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    # Metrics:
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test_encoded,
            y_pred_encoded,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test_encoded,
            y_pred_encoded,
            average="macro",
            zero_division=0,
        )
    )

    roc_path, micro_roc_auc = save_roc_curve(
        y_test_encoded,
        probabilities,
        number_of_classes,
    )

    pr_path, micro_average_precision = save_precision_recall_curve(
        y_test_encoded,
        probabilities,
        number_of_classes,
    )

    confusion_path = save_confusion_matrix(
        y_test_encoded,
        y_pred_encoded,
        label_encoder.classes_,
    )

    # Macro ROC-AUC is calculated only across classes represented in y_test.
    present_classes = np.unique(y_test_encoded)
    macro_roc_auc = None
    if len(present_classes) > 1:
        try:
            y_test_present_binary = label_binarize(
                y_test_encoded,
                classes=np.arange(number_of_classes),
            )[:, present_classes]
            probabilities_present = probabilities[:, present_classes]

            # Normalize the selected probabilities for a valid comparison.
            row_sums = probabilities_present.sum(axis=1, keepdims=True)
            probabilities_present = np.divide(
                probabilities_present,
                row_sums,
                out=np.zeros_like(probabilities_present),
                where=row_sums != 0,
            )

            macro_roc_auc = float(
                roc_auc_score(
                    y_test_present_binary,
                    probabilities_present,
                    average="macro",
                    multi_class="ovr",
                )
            )
        except ValueError:
            macro_roc_auc = None

    # Classification report:
    report_dict = classification_report(
        y_test_encoded,
        y_pred_encoded,
        labels=np.arange(number_of_classes),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    classification_report_path = (
        REPORTS_DIR / "logistic_regression_classification_report.csv"
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        classification_report_path,
        index=True,
    )

    # Predictions file:
    predictions_path = REPORTS_DIR / "logistic_regression_predictions.csv"
    pd.DataFrame(
        {
            "actual_condition": y_test.to_numpy(),
            "predicted_condition": y_pred_labels,
            "correct_prediction": y_test.to_numpy() == y_pred_labels,
        }
    ).to_csv(predictions_path, index=False)

    # ---------------- Save model and encoder ----------------
    model_path = MODELS_DIR / "logistic_regression_model.pkl"
    encoder_path = MODELS_DIR / "logistic_regression_label_encoder.pkl"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    # ---------------- Save metrics ----------------
    metrics = {
        "model": "Logistic Regression",
        "solver": "saga",
        "penalty": "l2",
        "class_weight": "balanced",
        "maximum_iterations": MAX_ITERATIONS,
        "iterations_used": int(np.max(model.n_iter_)),
        "training_rows": int(X_train.shape[0]),
        "testing_rows": int(X_test.shape[0]),
        "number_of_features": int(X_train.shape[1]),
        "number_of_classes": int(number_of_classes),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1_score": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "micro_roc_auc": float(micro_roc_auc),
        "macro_roc_auc": macro_roc_auc,
        "micro_average_precision": float(micro_average_precision),
        "cross_validation_f1_scores": [
            float(score) for score in cv_scores
        ],
        "mean_cross_validation_f1": float(cv_scores.mean()),
        "cross_validation_f1_std": float(cv_scores.std()),
        "cross_validation_time_seconds": float(cv_time),
        "training_time_seconds": float(training_time),
        "prediction_time_seconds": float(prediction_time),
        "cv_classes_excluded": excluded_cv_classes,
    }

    metrics_path = REPORTS_DIR / "logistic_regression_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    markdown_report_path = save_markdown_report(
        metrics,
        excluded_cv_classes,
    )

    # Console summary:
    print("\n" + "=" * 58)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 58)
    print(f"Accuracy:                 {accuracy:.4f}")
    print(f"Weighted precision:       {weighted_precision:.4f}")
    print(f"Weighted recall:          {weighted_recall:.4f}")
    print(f"Weighted F1-score:        {weighted_f1:.4f}")
    print(f"Macro precision:          {macro_precision:.4f}")
    print(f"Macro recall:             {macro_recall:.4f}")
    print(f"Macro F1-score:           {macro_f1:.4f}")
    print(f"Micro ROC-AUC:            {micro_roc_auc:.4f}")
    if macro_roc_auc is not None:
        print(f"Macro ROC-AUC:            {macro_roc_auc:.4f}")
    print(f"Micro average precision:  {micro_average_precision:.4f}")
    print(f"Mean CV weighted F1:      {cv_scores.mean():.4f}")

    print("\nSaved outputs:")
    print(f"Model:                    {model_path}")
    print(f"Label encoder:            {encoder_path}")
    print(f"Metrics:                  {metrics_path}")
    print(f"Classification report:    {classification_report_path}")
    print(f"Predictions:              {predictions_path}")
    print(f"Confusion matrix:         {confusion_path}")
    print(f"ROC curve:                {roc_path}")
    print(f"Precision-Recall curve:   {pr_path}")
    print(f"Member report:            {markdown_report_path}")
    print("\nWorkflow completed successfully.")


if __name__ == "__main__":
    main()
