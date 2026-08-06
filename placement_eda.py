import os

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data
matplotlib.use("Agg")

#DATA_PATH = r"E:\2-1\ML\placement.csv"


CHARTS_DIR=os.path.join(os.path.dirname(__file__),"static","charts")
def _chart_path(filename:str)-> str:
    os.makedirs(CHARTS_DIR,exist_ok=True)
    return os.path.join(CHARTS_DIR,filename)
def _save(filename:str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")
def run_eda()-> dict:
    data=load_data()
    charts =[]
    missing=data.isnull().sum()
    missing_pct =(missing/len(data))*100
    missing_df=pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df=missing_df[missing_df["missing_count"] >0].sort_values(
        "missing_count",ascending=False
    )

    if not missing_df.empty:
      plt.figure(figsize=(10, 5))
      sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
      plt.xticks(rotation=45, ha="right")
      plt.ylabel("Missing %")
      plt.title("Missing Values by column")
      _save("missing_values.png")
      charts.append("missing_values.png")

    duplicate_count=int(data.duplicated().sum())

    target_counts =data["PlacementStatus"].value_counts().to_dict()
    plt.figure()
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("Placement Status (0 =Not Placed. 1=Placed")
    plt.ylabel("Count")
    plt.title("Count of Placement Status")
    _save("target_distribution.png")
    charts.append("target_distribution.png")

    hist_cols = [
        "CGPA", "AttendancePercent", "AptitudeTestScore",
        "SoftSkillsRating"
    ]
    for col in hist_cols:
        if col in data.columns:
            plt.figure()
            sns.histplot(data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            fname = f"hist_{col.lower()}.png"
            _save(fname)
            charts.append(fname)

    missing_dict = {
        col: int(cnt)
        for col, cnt in missing.items()
        if cnt > 0
    }

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicate_count,
        "missing": missing_dict,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "charts": charts,
    }


if __name__ == "__main__":
    results = run_eda()
    print(results)