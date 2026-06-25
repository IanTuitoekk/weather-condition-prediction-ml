# Weather Condition-prediction-ml

# Team Responsibilities, Branches and Deliverables

Each team member will work on a dedicated feature branch and submit their work through Pull Requests (PRs) for review before merging into the `main` branch.

---

## Member 1 – Data Cleaning & Quality Assessment

### Branch

```bash
feature/data-cleaning
```

### Files to Create

```text
src/data_cleaning.py
reports/member1_data_cleaning_report.md
```

### Tasks

* Load and inspect dataset.
* Handle missing values.
* Remove duplicate records.
* Detect and handle outliers.
* Verify data consistency and data types.
* Save cleaned dataset to `data/processed/`.

### Deliverables

* Cleaned dataset.
* Data quality report.
* Missing-value visualizations.
* Outlier analysis.

---

## Member 2 – Exploratory Data Analysis (EDA)

### Branch

```bash
feature/eda
```

### Files to Create

```text
notebooks/eda.ipynb
reports/member2_eda_report.md
```

### Tasks

* Generate descriptive statistics.
* Analyze feature distributions.
* Analyze weather condition frequencies.
* Generate correlation matrix.
* Identify patterns and trends.

### Deliverables

* EDA notebook.
* Correlation heatmap.
* Distribution plots.
* Summary of insights.

---

## Member 3 – Feature Engineering & Preprocessing

### Branch

```bash
feature/preprocessing
```

### Files to Create

```text
src/preprocessing.py
reports/member3_preprocessing_report.md
```

### Tasks

* Encode categorical variables.
* Scale numerical features.
* Feature selection.
* Train-test split.
* Save preprocessing pipeline.

### Deliverables

* Processed datasets.
* Preprocessing pipeline.
* Feature engineering documentation.

---

## Member 4 – Logistic Regression Model

### Branch

```bash
feature/logistic-regression
```

### Files to Create

```text
src/logistic_regression.py
reports/member4_logistic_report.md
```

### Tasks

* Train Logistic Regression model.
* Perform cross-validation.
* Generate predictions.
* Evaluate model performance.

### Deliverables

* Logistic Regression model.
* Confusion Matrix.
* ROC Curve.
* Precision-Recall Curve.
* Performance metrics.

---

## Member 5 – Decision Tree Model

### Branch

```bash
feature/decision-tree
```

### Files to Create

```text
src/decision_tree.py
reports/member5_decision_tree_report.md
```

### Tasks

* Train Decision Tree model.
* Hyperparameter tuning.
* Visualize the decision tree.
* Evaluate model performance.

### Deliverables

* Tuned Decision Tree model.
* Tree visualization.
* Confusion Matrix.
* Performance metrics.

---

## Member 6 – Random Forest Model

### Branch

```bash
feature/random-forest
```

### Files to Create

```text
src/random_forest.py
reports/member6_random_forest_report.md
```

### Tasks

* Train Random Forest model.
* Hyperparameter tuning.
* Analyze feature importance.
* Evaluate model performance.

### Deliverables

* Tuned Random Forest model.
* Feature importance plots.
* Confusion Matrix.
* Performance metrics.

---

## Member 7 – XGBoost / Gradient Boosting & Model Comparison

### Branch

```bash
feature/xgboost-comparison
```

### Files to Create

```text
src/xgboost_model.py
src/model_comparison.py
reports/member7_xgboost_report.md
```

### Tasks

* Train XGBoost or Gradient Boosting model.
* Hyperparameter tuning.
* Early stopping.
* Evaluate performance.
* Compare all trained models.

### Deliverables

* XGBoost model.
* Feature importance visualization.
* Model comparison table.
* Model ranking charts.

---

## Member 8 – Ensemble Model & Deployment

### Branch

```bash
feature/ensemble-deployment
```

### Files to Create

```text
src/ensemble_model.py
app/app.py
reports/member8_deployment_report.md
```

### Tasks

* Build Voting Classifier or Stacking Ensemble.
* Use best models from Members 4–7.
* Save final trained model.
* Develop Streamlit dashboard.
* Build prediction interface.
* Demonstrate final system.

### Deliverables

* Ensemble model.
* Serialized trained model.
* Streamlit application.
* Deployment screenshots.

---

# Branch Creation

After cloning the repository:

```bash
git checkout -b feature/data-cleaning
```

Replace the branch name with your assigned feature branch.

Ensure that you are in the main branch before checking out
---

# Pull Request Workflow

1. Pull the latest changes from `main`.
2. Create your assigned feature branch.
3. Complete your assigned tasks.
4. Commit your changes.
5. Push your branch to GitHub.
6. Open a Pull Request to `main`.
7. Repository owner reviews and merges approved changes.

Example:

```text
feature/logistic-regression
            │
            ▼
      Pull Request
            │
            ▼
           main
```

---

# Repository Rules

* Do not commit directly to `main`.
* Use only your assigned feature branch.
* Submit all work through Pull Requests.
* Pull the latest changes from `main` before starting work.
* Ensure code runs successfully before opening a Pull Request.
* Include meaningful commit messages.

---


