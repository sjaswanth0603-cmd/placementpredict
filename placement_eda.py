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

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), dpi=100, bbox_inches="tight")
    plt.close("all")


def run_eda() -> dict:
    print("\n========== EDA STARTED ==========")
    data = load_data()
    charts = []

    # Missing values
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_pct"] > 0].sort_values(by="missing_count", ascending=False)

    if not missing_df.empty:
        plt.figure(figsize=(9, 5.2), dpi=105)
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"], color="#4ba3e3", edgecolor="#2980b9")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Percentage of Missing Values (%)")
        plt.title("Missing Values by Column", fontsize=13)
        _save("missing_values.png")
        charts.append("missing_values.png")

    duplicate_count = int(data.duplicated().sum())

    # Target distribution
    target_counts = {}
    if "PlacementStatus" in data.columns:
        target_counts = data["PlacementStatus"].value_counts().to_dict()
        plt.figure(figsize=(8, 5), dpi=105)
        sns.countplot(x="PlacementStatus", data=data, palette=["#aed6f1", "#4ba3e3"])
        plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
        plt.ylabel("Count")
        plt.title("Placement Status Distribution", fontsize=13)
        _save("target_distribution.png")
        charts.append("target_distribution.png")

    # Numeric distributions (clean 3x3 layout)
    hist_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "Salary Package"]
    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:
        fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2), dpi=105)
        axes = axes.flatten()
        for i, col in enumerate(hist_cols[:9]):
            axes[i].hist(data[col].dropna(), bins=18, color="#4c72b0", edgecolor="white")
            axes[i].set_title(col, fontsize=10)
            axes[i].tick_params(labelsize=8)
        for j in range(len(hist_cols[:9]), len(axes)):
            fig.delaxes(axes[j])
        _save("numeric_distribution.png")
        charts.append("numeric_distribution.png")

    # CGPA distribution with mean
    if "CGPA" in data.columns:
        plt.figure(figsize=(8.5, 5), dpi=105)
        sns.histplot(data["CGPA"], kde=True, color="#4ba3e3", edgecolor="#2980b9")
        plt.axvline(x=np.mean(data["CGPA"]), color="#e74c3c", linestyle="--", label=f"Mean: {np.mean(data['CGPA']):.2f}")
        plt.legend()
        plt.title("CGPA Distribution with Mean", fontsize=13)
        _save("cgpa_distribution_mean.png")
        charts.append("cgpa_distribution_mean.png")

    # Outlier detection (Boxplots)
    box_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "Salary Package"]
    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:
        plt.figure(figsize=(8.5, 4.2), dpi=105)
        sns.boxplot(x=data[col], color="#87ceeb")
        plt.title(f"Boxplot of {col}", fontsize=13)
        col_safe = col.replace(" ", "_").lower()
        fname = f"boxplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # Correlation Heatmap
    corr = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(9.5, 7.5), dpi=105)
    sns.heatmap(
        np.round(corr, 2),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap", fontsize=13)
    _save("correlation_heatmap.png")
    charts.append("correlation_heatmap.png")


    # CGPA vs Salary Package
    if "CGPA" in data.columns and "Salary Package" in data.columns:
        plt.figure(figsize=(9, 5.2), dpi=105)
        sample_data = data.sample(n=min(3000, len(data)), random_state=42)
        sns.regplot(
            x=sample_data["CGPA"],
            y=sample_data["Salary Package"],
            scatter_kws={"alpha": 0.4, "color": "#4ba3e3", "s": 18},
            line_kws={"color": "#1b6ca8", "lw": 2.5},
        )
        plt.title("CGPA vs Salary Package", fontsize=13)
        plt.xlabel("CGPA")
        plt.ylabel("Salary Package (LPA)")
        _save("cgpa_vs_salary.png")
        charts.append("cgpa_vs_salary.png")

    # CodingTestScore vs AptitudeTestScore
    if "CodingTestScore" in data.columns and "AptitudeTestScore" in data.columns:
        plt.figure(figsize=(9, 5.2), dpi=105)
        sample_data = data.sample(n=min(3000, len(data)), random_state=42)
        sns.scatterplot(
            x=sample_data["CodingTestScore"],
            y=sample_data["AptitudeTestScore"],
            alpha=0.4,
            color="#4ba3e3",
            s=18,
        )
        plt.title("CodingTestScore vs AptitudeTestScore", fontsize=13)
        plt.xlabel("CodingTestScore")
        plt.ylabel("AptitudeTestScore")
        _save("coding_vs_aptitude.png")
        charts.append("coding_vs_aptitude.png")

    # Categorical Feature Counts
    cat_cols = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CollegeTier"]
    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:
        plt.figure(figsize=(9, 5), dpi=105)
        order = data[col].value_counts().index
        sns.countplot(x=col, order=order, data=data, color="#4ba3e3", edgecolor="#2980b9")
        plt.title(f"Distribution of {col}", fontsize=13)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        col_safe = col.replace(" ", "_").lower()
        fname = f"countplot_{col_safe}.png"
        _save(fname)
        charts.append(fname)

    # Gender vs Placement Status
    if "Gender" in data.columns and "PlacementStatus" in data.columns:
        plt.figure(figsize=(8.5, 5), dpi=105)
        sns.countplot(x="Gender", data=data, hue="PlacementStatus", palette=["#aed6f1", "#4ba3e3"])
        plt.title("Gender vs Placement Status", fontsize=13)
        _save("gender_vs_placement.png")
        charts.append("gender_vs_placement.png")

    # College Tier vs Placement Status
    if "CollegeTier" in data.columns and "PlacementStatus" in data.columns:
        plt.figure(figsize=(8.5, 5), dpi=105)
        sns.countplot(x="CollegeTier", data=data, hue="PlacementStatus", palette=["#aed6f1", "#4ba3e3"])
        plt.title("Placement Status by College Tier", fontsize=13)
        _save("collegetier_vs_placement.png")
        charts.append("collegetier_vs_placement.png")

    # Average SGPA trend across semesters
    sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9) if f"SGPA_Sem{i}" in data.columns]
    avg_sgpa = {}
    if sgpa_cols:
        avg_sgpa_series = data[sgpa_cols].mean()
        avg_sgpa = avg_sgpa_series.to_dict()
        plt.figure(figsize=(9, 5), dpi=105)
        plt.plot(avg_sgpa_series.index, avg_sgpa_series.values, marker="o", color="#4ba3e3", markerfacecolor="#1b6ca8", lw=2.5)
        plt.title("Average SGPA Across Semesters", fontsize=13)
        plt.xlabel("Semester")
        plt.ylabel("Average SGPA")
        _save("avg_sgpa_trend.png")
        charts.append("avg_sgpa_trend.png")

    # Salary Package Analysis
    salary_statistics = {}
    if "Salary Package" in data.columns:
        placed_data = data[data["PlacementStatus"] == 1] if "PlacementStatus" in data.columns else data
        salary_statistics = placed_data["Salary Package"].describe().to_dict()

        plt.figure(figsize=(9, 5), dpi=105)
        sns.histplot(placed_data["Salary Package"].dropna(), kde=True, bins=25, color="#4ba3e3", edgecolor="#2980b9")
        plt.title("Salary Package Distribution for Placed Students", fontsize=13)
        plt.xlabel("Salary Package (LPA)")
        plt.ylabel("Count")
        _save("salary_distribution.png")
        charts.append("salary_distribution.png")

        if "CollegeTier" in placed_data.columns:
            plt.figure(figsize=(9, 5), dpi=105)
            sns.boxplot(x="CollegeTier", y="Salary Package", data=placed_data, color="#87ceeb")
            plt.title("Salary Package by College Tier", fontsize=13)
            plt.xlabel("College Tier")
            plt.ylabel("Salary Package (LPA)")
            _save("salary_by_collegetier.png")
            charts.append("salary_by_collegetier.png")

    # Pairplot
    pairplot_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
    if {"CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"}.issubset(data.columns):
        pairplot_df = data[pairplot_cols].dropna()
        if len(pairplot_df) > 1000:
            pairplot_df = pairplot_df.sample(n=1000, random_state=42)

        g = sns.pairplot(
            pairplot_df,
            vars=["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"],
            hue="PlacementStatus",
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 22},
            height=2.0,
            aspect=1.2,
        )
        g.fig.suptitle("Multivariate Feature Pairplot", y=1.02, fontsize=13)
        plt.savefig(_chart_path("pairplot.png"), dpi=105, bbox_inches="tight")
        plt.close("all")
        charts.append("pairplot.png")


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
    print("Generated charts:", len(results["charts"]))