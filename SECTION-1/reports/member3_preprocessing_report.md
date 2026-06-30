# Member 3 – Feature Engineering & Preprocessing Report

## 1. Executive Summary

This stage focused on preparing the cleaned weather dataset for machine learning model development. The preprocessing workflow transformed raw data into a structured, machine-learning-ready format by performing feature engineering, categorical encoding, numerical scaling, train-test splitting, and pipeline generation.

The process also integrated insights obtained from Member 2's Exploratory Data Analysis (EDA) to improve data quality and reduce redundancy.

---

## 2. Dataset Information

Dataset Used:

GlobalWeatherRepository_cleaned.csv

Original Dataset Shape:

- Rows: 120,627
- Columns: 41

Processed Dataset Shape:

- Rows: 120,625
- Columns: 36

Rows Removed:

- 2 records removed due to extremely rare target classes with insufficient samples for stratified splitting.

Target Variable:

`condition_text`

---

## 3. Feature Engineering Decisions Based on EDA Findings

### A. Multicollinearity Removal

EDA identified a near-perfect correlation between:

- `temperature_celsius`
- `feels_like_celsius`

Action Taken:

Removed:

- `feels_like_celsius`

Reason:

Removing highly correlated features reduces multicollinearity and prevents redundant information from affecting model performance.

---

### B. Duplicate Metric Removal

EDA identified duplicate measurements represented in different units.

Removed variables:

- `temperature_fahrenheit`
- `wind_mph`
- `pressure_in`

Reason:

These variables contain duplicate information already represented by:

- temperature_celsius
- wind_kph
- pressure_mb

Removing them reduces dimensionality without losing information.

---

### C. Air Quality Feature Transformation

Feature:

`air_quality_PM2.5`

Observation from EDA:

The PM2.5 distribution displayed strong right skewness with significant outliers.

Action Taken:

Applied logarithmic transformation:

log(1 + x)

Reason:

The transformation compresses extreme values and stabilizes feature variance.

---

### D. Rare Target Class Handling

EDA revealed severe class imbalance among weather categories.

Rare classes removed:

- Heavy freezing drizzle
- Moderate or heavy freezing rain

Reason:

Stratified train-test splitting requires a minimum of two samples for each target category.

---

## 4. Preprocessing Steps

### Categorical Variable Encoding

Method used:

One-Hot Encoding

Reason:

Machine learning algorithms require numerical inputs, and One-Hot Encoding prevents artificial ordering among categories.

---

### Numerical Feature Scaling

Method used:

StandardScaler

Formula:

z = (x − μ) / σ

Reason:

Scaling ensures features contribute equally during model training.

---

### Train-Test Split

Configuration:

- Training set: 80%
- Testing set: 20%
- Random State: 42
- Stratified split enabled

Reason:

Stratification preserves class distributions across datasets.

---

## 5. Preprocessing Pipeline

A preprocessing pipeline was created using:

- ColumnTransformer
- StandardScaler
- OneHotEncoder
- Pipeline

Purpose:

- Maintain reproducibility
- Prevent data leakage
- Simplify future model deployment

---

## 6. Deliverables Generated

Processed datasets:

- data/processed/X_train.csv
- data/processed/X_test.csv
- data/processed/y_train.csv
- data/processed/y_test.csv

Pipeline:

- models/preprocessing_pipeline.pkl

Code:

- src/preprocessing.py

Documentation:

- reports/member3_preprocessing_report.md

---

## 7. Conclusion

The preprocessing stage successfully transformed the cleaned weather dataset into a machine-learning-ready format. Feature engineering decisions were guided by EDA findings to reduce redundancy, address imbalance issues, and improve overall model readiness.