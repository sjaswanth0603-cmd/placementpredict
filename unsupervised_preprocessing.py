"""
Unsupervised Learning Data Preprocessing Module.

Handles loading of dataset (defaulting to train.csv), selection of numeric
features, missing value handling via median imputation, and feature scaling
using StandardScaler. Ensures placement target and identifiers are isolated
from training data.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILENAME = "train.csv"
FALLBACK_DATA_FILENAME = "placement_predict_50k Dataset (3)(in).csv"

# Output directories
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Canonical numeric feature columns for placement unsupervised learning
NUMERIC_FEATURE_COLUMNS = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular",
]

TARGET_COLUMN = "PlacementStatus"
ID_COLUMN = "StudentID"


def resolve_dataset_path(custom_path: Optional[str] = None) -> str:
    """Resolve the dataset path, prioritizing custom_path, then train.csv, then fallback."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    primary_path = os.path.join(BASE_DIR, DEFAULT_DATA_FILENAME)
    if os.path.exists(primary_path):
        return primary_path

    fallback_path = os.path.join(BASE_DIR, FALLBACK_DATA_FILENAME)
    if os.path.exists(fallback_path):
        return fallback_path

    # Check parent directory
    nested_path = os.path.join(BASE_DIR, "placement", FALLBACK_DATA_FILENAME)
    if os.path.exists(nested_path):
        return nested_path

    raise FileNotFoundError(
        f"Placement dataset not found. Checked: '{primary_path}', '{fallback_path}', '{nested_path}'."
    )


def load_raw_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset into a DataFrame."""
    resolved_path = resolve_dataset_path(path)
    df = pd.read_csv(resolved_path)
    return df


def prepare_unsupervised_data(
    filepath: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """Clean, impute, and scale numeric features for unsupervised learning.

    Args:
        filepath: Optional path to dataset CSV.
        sample_size: Optional integer to take a representative random sample.
        random_state: Random seed for sampling reproducibility.

    Returns:
        X_scaled: Standardized numpy array of shape (N, 20).
        df_imputed: Imputed numeric features DataFrame.
        df_meta: Metadata DataFrame containing StudentID and PlacementStatus.
        scaler: Fitted StandardScaler instance.
        feature_names: List of numeric feature names.
    """
    raw_df = load_raw_dataset(filepath)

    if sample_size and sample_size < len(raw_df):
        raw_df = raw_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Validate target column presence for evaluation/visualization
    if TARGET_COLUMN not in raw_df.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' not found in dataset.")

    # Metadata extraction (target and ID preserved, NEVER used to train clustering)
    meta_cols = [col for col in [ID_COLUMN, TARGET_COLUMN] if col in raw_df.columns]
    df_meta = raw_df[meta_cols].copy()

    # Determine numeric feature columns
    available_numeric = [
        col for col in NUMERIC_FEATURE_COLUMNS if col in raw_df.columns
    ]
    if len(available_numeric) < len(NUMERIC_FEATURE_COLUMNS):
        # Auto-detect any other numeric columns if canonical list differs
        fallback_numeric = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        available_numeric = [
            c for c in fallback_numeric if c not in [ID_COLUMN, TARGET_COLUMN, "IsAnomaly", "Salary Package"]
        ]

    df_numeric = raw_df[available_numeric].copy()

    # Median Imputation
    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(imputed_values, columns=available_numeric, index=raw_df.index)

    # Standardization (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)

    return X_scaled, df_imputed, df_meta, scaler, available_numeric


def ensure_output_directories():
    """Ensure that all required output folders exist."""
    for directory in [PLOTS_DIR, DATA_DIR, REPORTS_DIR]:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    ensure_output_directories()
    X_scaled, df_imputed, df_meta, scaler, features = prepare_unsupervised_data()
    print("========== PREPROCESSING FOR UNSUPERVISED LEARNING ==========")
    print(f"Dataset successfully loaded and preprocessed.")
    print(f"Total instances: {X_scaled.shape[0]}")
    print(f"Number of numeric features: {X_scaled.shape[1]}")
    print(f"Features list: {features}")
    print(f"PlacementStatus distribution:\n{df_meta[TARGET_COLUMN].value_counts().to_dict()}")
    print("Missing values after median imputation: 0")
    print(f"Feature mean (approx): {np.mean(X_scaled):.4f}, std (approx): {np.std(X_scaled):.4f}")
