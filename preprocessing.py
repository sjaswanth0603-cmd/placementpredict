import pandas as pd
import numpy as np

from load_data import load_data


# ============================================================
# PREPROCESSING
# ============================================================

def run_preprocessing():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    data = load_data()

    original_rows = data.shape[0]
    original_columns = data.shape[1]


    # --------------------------------------------------------
    # 2. CHECK MISSING VALUES
    # --------------------------------------------------------

    missing_before = int(data.isnull().sum().sum())


    # --------------------------------------------------------
    # 3. HANDLE MISSING VALUES
    # --------------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns


    # Numeric columns -> median

    for column in numeric_columns:

        if data[column].isnull().any():

            data[column] = data[column].fillna(
                data[column].median()
            )


    # Categorical columns -> mode

    for column in categorical_columns:

        if data[column].isnull().any():

            mode_value = data[column].mode()

            if not mode_value.empty:

                data[column] = data[column].fillna(
                    mode_value[0]
                )


    missing_after = int(data.isnull().sum().sum())


    # --------------------------------------------------------
    # 4. REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicates_before = int(data.duplicated().sum())

    data = data.drop_duplicates()

    duplicates_after = int(data.duplicated().sum())


    # --------------------------------------------------------
    # 5. REMOVE UNNECESSARY ID COLUMN
    # --------------------------------------------------------

    removed_columns = []

    if "StudentID" in data.columns:

        data = data.drop(columns=["StudentID"])

        removed_columns.append("StudentID")


    # --------------------------------------------------------
    # 6. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    target_column = "PlacementStatus"

    if target_column in data.columns:

        X = data.drop(columns=[target_column])

        y = data[target_column]

    else:

        X = data.copy()

        y = pd.Series(dtype=float)


    # --------------------------------------------------------
    # 7. ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()


    X_encoded = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True
    )


    # --------------------------------------------------------
    # 8. CONVERT BOOLEAN COLUMNS TO INTEGER
    # --------------------------------------------------------

    bool_columns = X_encoded.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_columns) > 0:

        X_encoded[bool_columns] = X_encoded[
            bool_columns
        ].astype(int)


    # --------------------------------------------------------
    # 9. FINAL INFORMATION
    # --------------------------------------------------------

    final_rows = X_encoded.shape[0]
    final_columns = X_encoded.shape[1]


    # --------------------------------------------------------
    # 10. PREVIEW
    # --------------------------------------------------------

    preview = X_encoded.head(10).to_dict("records")


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

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

        "feature_count": X_encoded.shape[1],

        "target_count": len(y),

        "categorical_features": categorical_features,

        "numeric_features": numeric_features,

        "columns": list(X_encoded.columns),

        "preview": preview
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = run_preprocessing()

    print("\n========== PREPROCESSING ==========")

    print(
        "Original rows:",
        result["original_rows"]
    )

    print(
        "Original columns:",
        result["original_columns"]
    )

    print(
        "Final rows:",
        result["final_rows"]
    )

    print(
        "Final columns:",
        result["final_columns"]
    )

    print(
        "Missing before:",
        result["missing_before"]
    )

    print(
        "Missing after:",
        result["missing_after"]
    )

    print(
        "Duplicates:",
        result["duplicates_before"]
    )

    print(
        "Removed columns:",
        result["removed_columns"]
    )