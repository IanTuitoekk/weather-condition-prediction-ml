# Member 4 – Logistic Regression Report

## Objective

Train and evaluate a multiclass Logistic Regression model for predicting
weather conditions using the processed features produced by Member 3.

## Dataset

- Training rows: 96,500
- Testing rows: 24,125
- Number of processed features: 210
- Number of target classes: 46

## Model Configuration

- Algorithm: Logistic Regression
- Solver: saga
- Maximum iterations: 200
- Class weighting: balanced
- Random state: 42

The target labels were standardized for capitalization and surrounding
whitespace before training. Balanced class weights were used because the
weather-condition classes were highly imbalanced.

## Cross-Validation

A 3-fold Stratified K-Fold cross-validation procedure was used.

- Fold weighted F1 scores: [0.4537185116981069, 0.4444262938339399, 0.4473304851999985]
- Mean weighted F1 score: 0.4485
- Standard deviation: 0.0039
- Classes excluded from cross-validation: Moderate Or Heavy Snow In Area With Thunder, Patchy Snow Nearby

## Test-Set Performance

- Accuracy: 0.4136
- Weighted precision: 0.6538
- Weighted recall: 0.4136
- Weighted F1-score: 0.4401
- Macro precision: 0.1388
- Macro recall: 0.2902
- Macro F1-score: 0.1475
- Micro ROC-AUC: 0.9513
- Micro average precision: 0.4447

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
