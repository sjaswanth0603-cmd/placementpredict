import os
import pandas as pd


DATA_PATH = r"C:\Users\S.JASWANTH NAIDU\PycharmProjects\PythonProject5\placement_predict_50k Dataset (3)(in).csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    return df


# ============================================================
# DATA SUMMARY
# ============================================================

def get_data_summary() -> dict:

    df = load_data()

    summary = {

        "n_rows": df.shape[0],

        "n_cols": df.shape[1],

        "columns": list(df.columns),

        "dtypes": {
            col: str(df[col].dtype)
            for col in df.columns
        },

        "missing_counts": {
            col: int(df[col].isnull().sum())
            for col in df.columns
        },

        "preview": df.head(10).to_dict("records")
    }

    return summary


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(get_data_summary())