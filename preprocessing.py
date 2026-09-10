import numpy as np
import pandas as pd
from load_data import load_data


def run_preprocessing() -> dict:
    """Clean and preprocess the placement dataset.

    Performs:
      1. Missing value imputation (median for numeric, mode for categorical).
      2. Duplicate removal.
      3. Removal of identifier columns (StudentID).
      4. Separation of features and target (PlacementStatus).
      5. One-hot encoding of categorical variables.
    """
    data = load_data()
    original_rows, original_columns = data.shape

    # Check and handle missing values
    missing_before = int(data.isnull().sum().sum())
    numeric_columns = data.select_dtypes(include=np.number).columns
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns

    for column in numeric_columns:
        if data[column].isnull().any():
            data[column] = data[column].fillna(data[column].median())

    for column in categorical_columns:
        if data[column].isnull().any():
            mode_val = data[column].mode()
            if not mode_val.empty:
                data[column] = data[column].fillna(mode_val[0])

    missing_after = int(data.isnull().sum().sum())

    # Remove duplicates
    duplicates_before = int(data.duplicated().sum())
    data = data.drop_duplicates()
    duplicates_after = int(data.duplicated().sum())

    # Remove unnecessary ID column
    removed_columns = []
    if "StudentID" in data.columns:
        data = data.drop(columns=["StudentID"])
        removed_columns.append("StudentID")

    # Separate target and features
    target_column = "PlacementStatus"
    if target_column in data.columns:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data.copy()
        y = pd.Series(dtype=float)

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert bools to integers
    bool_cols = X_encoded.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

    final_rows, final_columns = X_encoded.shape
    preview = X_encoded.head(10).to_dict("records")

    return {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": final_rows,
        "final_columns": final_columns,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "removed_columns": removed_columns,
        "target_column": target_column,
        "feature_count": final_columns,
        "target_count": len(y),
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "columns": list(X_encoded.columns),
        "preview": preview,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print("========== PREPROCESSING SUMMARY ==========")
    print(f"Original shape: ({result['original_rows']}, {result['original_columns']})")
    print(f"Processed shape: ({result['final_rows']}, {result['final_columns']})")
    print(f"Missing handled: {result['missing_before']} -> {result['missing_after']}")
    print(f"Duplicates removed: {result['duplicates_before'] - result['duplicates_after']}")