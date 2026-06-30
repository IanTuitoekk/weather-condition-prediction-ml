import pandas as pd
import numpy as np
import joblib
from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================
# Load cleaned dataset
# ==========================

df = pd.read_csv(
    "data/GlobalWeatherRepository_cleaned.csv"
)

print("Original shape:", df.shape)

# ==========================
# Feature Selection
# Based on Member 2 EDA
# ==========================

columns_to_drop = [

    # remove multicollinearity
    'feels_like_celsius',

    # remove duplicate metrics
    'temperature_fahrenheit',
    'wind_mph',
    'pressure_in',

]

existing = [
    col for col in columns_to_drop
    if col in df.columns
]

df.drop(
    columns=existing,
    inplace=True
)

# ==========================
# Handle skewed PM2.5
# ==========================

if 'air_quality_PM2.5' in df.columns:
    df['air_quality_PM2.5'] = np.log1p(
        df['air_quality_PM2.5']
    )

# ==========================
# Define target
# ==========================

target = 'condition_text'

X = df.drop(columns=[target])
y = df[target]

# ==========================
# Remove extremely rare classes
# ==========================

class_counts = y.value_counts()

valid_classes = class_counts[
    class_counts >= 2
].index

mask = y.isin(valid_classes)

X = X[mask]
y = y[mask]

print("Filtered shape:", X.shape)

# ==========================
# Split feature types
# ==========================

numerical_features = X.select_dtypes(
    include=['int64','float64']
).columns

categorical_features = X.select_dtypes(
    include=['object','string']
).columns

# ==========================
# Pipelines
# ==========================

numeric_pipeline=Pipeline(
    steps=[
        ('scaler',StandardScaler())
    ]
)

categorical_pipeline=Pipeline(
    steps=[
        (
            'encoder',
            OneHotEncoder(
                handle_unknown='ignore',
                max_categories=20
            )
        )
    ]
)

preprocessor=ColumnTransformer(
    transformers=[
        (
            'num',
            numeric_pipeline,
            numerical_features
        ),
        (
            'cat',
            categorical_pipeline,
            categorical_features
        )
    ]
)

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Transform
# ==========================

X_train_processed=preprocessor.fit_transform(
    X_train
)

X_test_processed=preprocessor.transform(
    X_test
)

# ==========================
# Save outputs
# ==========================
X_train_df = pd.DataFrame.sparse.from_spmatrix(
    X_train_processed
)

X_test_df = pd.DataFrame.sparse.from_spmatrix(
    X_test_processed
)

X_train_df.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test_df.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

joblib.dump(
    preprocessor,
    "models/preprocessing_pipeline.pkl"
)

print("Preprocessing complete")
print("Files saved successfully")