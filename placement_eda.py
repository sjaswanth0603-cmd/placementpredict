import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for web application

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from load_data import load_data

# Set styling
sns.set_theme(style="whitegrid")

# Charts directory configuration
CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)

def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)

def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight")
    plt.close("all")

def run_eda() -> dict:
    print("\n========== EDA STARTED ==========")

    # 1. LOAD DATA
    data = load_data()
    print("=" * 80)
    print("1. Data loaded")
    print("=" * 80)
    print("shape:", data.shape)
    print("\nFirst 5 rows of data:\n", data.head())

    charts = []

    # 2. BASIC INFO / STRUCTURE
    print("\n" + "=" * 80)
    print("2. BASIC INFO")
    print("=" * 80)
    data.info()
    print("\nColumns dtypes:\n", data.dtypes)
    print("\nDescribe (numeric):\n", data.describe())
    try:
        print("\nDescribe (categorical):\n", data.describe(include="object"))
    except ValueError:
        print("\nDescribe (categorical): No categorical columns to describe.")

    # 3. MISSING VALUES
    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)
    print(missing_df)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Percentage of missing values")
        plt.title("Missing values by column")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # 4. DUPLICATES
    print("\n" + "=" * 80)
    print("4. Duplicate Rows")
    print("=" * 80)
    duplicate_count = int(data.duplicated().sum())
    print("Duplicate rows:", duplicate_count)

    # 5. TARGET VARIABLE - PLACEMENT STATUS
    print("\n" + "=" * 80)
    print("5. TARGET VARIABLE- PLACEMENTSTATUS")
    print("=" * 80)
    target_counts = {}
    if "PlacementStatus" in data.columns:
        target_counts = data["PlacementStatus"].value_counts().to_dict()
        print(data["PlacementStatus"].value_counts())

        plt.figure(dpi=125)
        sns.countplot(x="PlacementStatus", data=data)
        plt.xlabel("PlacementStatus(0 = Not placed, 1 = Placed)")
        plt.ylabel("count")
        plt.title("Placement Status Distribution")
        _save("target_distribution.png")
        charts.append("target_distribution.png")

    # 6. NUMERIC FEATURE DISTRIBUTION
    print("\n" + "=" * 80)
    print("6. NUMERIC FEATURE DISTRIBUTIION")
    print("=" * 80)
    hist_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "Salary Package"]
    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:
        plt.figure(figsize=(14, 10))
        data[hist_cols].hist(figsize=(14, 10), bins=20)
        _save("numeric_distribution.png")
        charts.append("numeric_distribution.png")

    # Mean Line example for CGPA
    if "CGPA" in data.columns:
        plt.figure(dpi=125)
        sns.histplot(data["CGPA"], kde=True)
        plt.axvline(x=np.mean(data["CGPA"]), color="green", linestyle="--", label="Mean")
        plt.legend()
        plt.title("CGPA Distribution with mean")
        _save("cgpa_distribution_mean.png")
        charts.append("cgpa_distribution_mean.png")

    # 7. OUTLIER DETECTION (BOXPLOTS)
    print("\n" + "=" * 80)
    print("7. OUTLIER DETECTION (BOXPLOTS)")
    print("=" * 80)
    box_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "Salary Package"]
    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=data[col], color="skyblue")
        plt.title(f"Boxplot of {col}", fontsize=18)
        col_safe = col.replace(" ", "_").lower()
        fname = f"boxplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # 8. CORRELATION ANALYSIS (Multivariate)
    print("\n" + "=" * 80)
    print("8. CORRELERATION ANALYSIS")
    print("=" * 80)
    corr = data.select_dtypes(include=[np.number]).corr()
    print(np.round(corr, 2))

    plt.figure(figsize=(16, 12), dpi=100)
    sns.heatmap(np.round(corr, 2), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    _save("correlation_heatmap.png")
    charts.append("correlation_heatmap.png")

    # 9. SCATTER OR REGRESSION PLOTS (BI-VARIATE)
    print("\n" + "=" * 80)
    print("9. RELATIONSHIP PLOTS (BI-VARIATE)")
    print("=" * 80)

    if "CGPA" in data.columns and "Salary Package" in data.columns:
        plt.figure(figsize=(12, 6), dpi=100)
        sns.regplot(x=data["CGPA"], y=data["Salary Package"], data=data, color="lightblue")
        plt.title("CGPA vs Salary Package")
        _save("cgpa_vs_salary.png")
        charts.append("cgpa_vs_salary.png")

    if "CodingTestScore" in data.columns and "AptitudeTestScore" in data.columns:
        plt.figure(figsize=(12, 6), dpi=100)
        sns.scatterplot(x="CodingTestScore", y="AptitudeTestScore", data=data, color="lightblue")
        plt.title("CodingTestScore vs AptitudeTestScore")
        _save("coding_vs_aptitude.png")
        charts.append("coding_vs_aptitude.png")

    # 10. CATEGORICAL FEATURE COUNTS
    print("\n" + "=" * 80)
    print("10. Categorical feature counts")
    print("=" * 80)
    cat_cols = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CollegeTier"]
    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:
        print(f"\n-----{col}-----\n")
        plt.figure(figsize=(10, 5), dpi=125)
        order = data[col].value_counts().index
        sns.countplot(x=col, order=order, data=data)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        col_safe = col.replace(" ", "_").lower()
        fname = f"countplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # 11. GENDER VS PLACEMENT STATUS
    print("\n" + "=" * 80)
    print("11. GENDER VS PLACEMENT STATUS")
    print("=" * 80)
    if "Gender" in data.columns and "PlacementStatus" in data.columns:
        plt.figure(dpi=125)
        sns.countplot(x="Gender", data=data, hue="PlacementStatus")
        plt.title("Gender vs Placement Status")
        _save("gender_vs_placement.png")
        charts.append("gender_vs_placement.png")

    # 12. COLLEGE TIER VS PLACEMENT STATUS
    print("\n" + "=" * 80)
    print("12. COLLEGE TIER VS PLACEMENT STATUS")
    print("=" * 80)
    if "CollegeTier" in data.columns and "PlacementStatus" in data.columns:
        plt.figure(dpi=125)
        sns.countplot(x="CollegeTier", data=data, hue="PlacementStatus")
        plt.title("Placement Status by College Tier")
        _save("collegetier_vs_placement.png")
        charts.append("collegetier_vs_placement.png")

    # 13. AVERAGE CGPA TREND ACROSS SEMESTERS
    print("\n" + "=" * 80)
    print("13. AVERAGE CGPA TREND ACROSS SEMESTERS")
    print("=" * 80)
    sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9) if f"SGPA_Sem{i}" in data.columns]
    avg_sgpa = {}
    if sgpa_cols:
        avg_sgpa_series = data[sgpa_cols].mean()
        avg_sgpa = avg_sgpa_series.to_dict()
        print(avg_sgpa_series)

        plt.figure(figsize=(10, 6))
        plt.plot(avg_sgpa_series.index, avg_sgpa_series.values, marker="o")
        plt.title("Average SGPA Across Semesters")
        plt.xlabel("Semester")
        plt.ylabel("Average SGPA")
        _save("avg_sgpa_trend.png")
        charts.append("avg_sgpa_trend.png")

    # 14. SALARY PACKAGE ANALYSIS (UNI-VARIATE & BI-VARIATE)
    print("\n" + "=" * 80)
    print("14. SALARY PACKAGE ANALYSIS")
    print("=" * 80)
    salary_statistics = {}
    if "Salary Package" in data.columns:
        placed_data = data[data["PlacementStatus"] == 1] if "PlacementStatus" in data.columns else data
        salary_stats_series = placed_data["Salary Package"].describe()
        salary_statistics = salary_stats_series.to_dict()

        plt.figure(figsize=(10, 6), dpi=120)
        sns.histplot(placed_data["Salary Package"].dropna(), kde=True, bins=30)
        plt.title("Salary Package Distribution for Placed Students")
        plt.xlabel("Salary Package")
        plt.ylabel("Count")
        _save("salary_distribution.png")
        charts.append("salary_distribution.png")

        if "CollegeTier" in placed_data.columns:
            plt.figure(figsize=(10, 6), dpi=120)
            sns.boxplot(x="CollegeTier", y="Salary Package", data=placed_data)
            plt.title("Salary Package by College Tier")
            plt.xlabel("College Tier")
            plt.ylabel("Salary Package")
            _save("salary_by_collegetier.png")
            charts.append("salary_by_collegetier.png")

    # 15. PAIRPLOT (MULTI-VARIATE)
    print("\n" + "=" * 80)
    print("15. PAIRPLOT (MULTI-VARIATE)")
    print("=" * 80)
    pairplot_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
    pairplot_cols = [c for c in pairplot_cols if c in data.columns]

    if {"CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"}.issubset(data.columns):
        pairplot_df = data[pairplot_cols].dropna()
        if len(pairplot_df) > 1000:
            pairplot_df = pairplot_df.sample(n=1000, random_state=42)

        sns.pairplot(
            pairplot_df,
            vars=["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"],
            hue="PlacementStatus",
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 25},
        )
        plt.savefig(_chart_path("pairplot.png"), bbox_inches="tight")
        plt.close("all")
        charts.append("pairplot.png")

    # Numeric and Categorical columns classification
    numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(data.select_dtypes(include=["object", "category"]).columns)

    print("\n========== EDA COMPLETED ==========")
    print("Charts generated:", len(charts))

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicate_count,
        "missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0},
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "avg_sgpa": {str(k): float(v) for k, v in avg_sgpa.items()},
        "salary_statistics": {str(k): float(v) for k, v in salary_statistics.items()},
        "charts": charts,
    }

if __name__ == "__main__":
    results = run_eda()
    print("\n================================================")
    print("EDA RESULTS PREVIEW")
    print("================================================")
    print("Rows:", results["n_rows"])
    print("Columns:", results["n_cols"])
    print("Duplicate rows:", results["duplicate_count"])
    print("Missing values count:", results["missing"])
    print("Placement counts:", results["target_counts"])
    print("Charts generated:")
    for chart in results["charts"]:
        print(" -", chart)