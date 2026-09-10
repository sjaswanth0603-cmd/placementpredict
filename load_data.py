import os
import pandas as pd

DEFAULT_DATA_FILENAME = "placement_predict_50k Dataset (3)(in).csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALLBACK_PATH = r"C:\Users\S.JASWANTH NAIDU\PycharmProjects\PythonProject5\placement_predict_50k Dataset (3)(in).csv"


def find_data_path() -> str:
    """Resolve the dataset file path reliably across different environments."""
    local_path = os.path.join(BASE_DIR, DEFAULT_DATA_FILENAME)
    if os.path.exists(local_path):
        return local_path
    if os.path.exists(FALLBACK_PATH):
        return FALLBACK_PATH
    raise FileNotFoundError(
        f"Dataset not found at '{local_path}' or '{FALLBACK_PATH}'."
    )


def load_data(path: str = None) -> pd.DataFrame:
    """Load placement dataset into a pandas DataFrame."""
    resolved_path = path or find_data_path()
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Dataset not found:\n{resolved_path}")
    return pd.read_csv(resolved_path)


def get_data_summary() -> dict:
    """Generate basic dataset summary including shape, types, missing counts, and preview."""
    df = load_data()
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict("records"),
    }


if __name__ == "__main__":
    summary = get_data_summary()
    print(f"Loaded {summary['n_rows']} rows and {summary['n_cols']} columns successfully.")