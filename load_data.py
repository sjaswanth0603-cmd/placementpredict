import os
import pandas as pd

DATA_PATH = r"C:\Users\S.JASWANTH NAIDU\PycharmProjects\placement\placement_predict_50k Dataset (3)(in).csv"
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def get_data_summary():
    df = load_data(DATA_PATH)

    summary = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {
            col: int(df[col].isna().sum()) for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records")
    }

    return summary


if __name__ == "__main__":
    print(get_data_summary())