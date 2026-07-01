import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# Configuration


RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports" / "DecisionTree"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Load datasets

print("Loading datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).iloc[:, 0]

print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")


# Encode labels

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)


# Hyperparameter tuning


print("\nPerforming hyperparameter tuning...")

parameter_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

decision_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=parameter_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
)

grid_search.fit(
    X_train,
    y_train_encoded
)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"Best CV score: "
    f"{grid_search.best_score_:.4f}"
)

# Train final model


print("\nTraining final model...")

best_model.fit(
    X_train,
    y_train_encoded
)


# Predictions


print("Generating predictions...")

y_pred = best_model.predict(X_test)


# Metrics

accuracy = accuracy_score(
    y_test_encoded,
    y_pred
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

print("\nResults")
print("=" * 40)
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

# Classification report

report = classification_report(
    y_test_encoded,
    y_pred,
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    REPORTS_DIR /
    "decision_tree_classification_report.csv"
)


# Confusion matrix


cm = confusion_matrix(
    y_test_encoded,
    y_pred,
    normalize="true",
)

fig, ax = plt.subplots(
    figsize=(12, 12)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_,
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    include_values=False,
)

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.savefig(
    REPORTS_DIR /
    "decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Tree visualization

print("Generating tree visualization...")

plt.figure(
    figsize=(30, 15)
)

plot_tree(
    best_model,
    filled=True,
    max_depth=3,
    fontsize=5,
    class_names=encoder.classes_,
)

plt.title(
    "Decision Tree Visualization"
)

plt.savefig(
    REPORTS_DIR /
    "decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Save predictions


predictions = pd.DataFrame(
    {
        "actual": y_test,
        "predicted":
            encoder.inverse_transform(y_pred),
    }
)

predictions.to_csv(
    REPORTS_DIR /
    "decision_tree_predictions.csv",
    index=False,
)


# Save model


joblib.dump(
    best_model,
    MODELS_DIR /
    "decision_tree_model.pkl",
)

joblib.dump(
    encoder,
    MODELS_DIR /
    "decision_tree_label_encoder.pkl",
)


# Save metrics

metrics = {
    "model": "Decision Tree",
    "best_parameters":
        grid_search.best_params_,
    "best_cv_score":
        float(grid_search.best_score_),
    "accuracy":
        float(accuracy),
    "weighted_precision":
        float(weighted_precision),
    "weighted_recall":
        float(weighted_recall),
    "weighted_f1":
        float(weighted_f1),
    "macro_precision":
        float(macro_precision),
    "macro_recall":
        float(macro_recall),
    "macro_f1":
        float(macro_f1),
}

with open(
    REPORTS_DIR /
    "decision_tree_metrics.json",
    "w",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
    )

# Markdown report


report_text = f"""
# Member 5 - Decision Tree Report

## Best Parameters

{grid_search.best_params_}

## Cross Validation Score

{grid_search.best_score_:.4f}

## Test Performance

- Accuracy: {accuracy:.4f}
- Weighted Precision: {weighted_precision:.4f}
- Weighted Recall: {weighted_recall:.4f}
- Weighted F1: {weighted_f1:.4f}
- Macro Precision: {macro_precision:.4f}
- Macro Recall: {macro_recall:.4f}
- Macro F1: {macro_f1:.4f}

## Deliverables

- Tuned Decision Tree model
- Tree visualization
- Confusion matrix
- Classification report
- Performance metrics
"""

(
    REPORTS_DIR /
    "member5_decision_tree_report.md"
).write_text(report_text)

print("\nDecision Tree workflow completed successfully.")