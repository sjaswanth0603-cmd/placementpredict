import os
from math import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from load_data import load_data


def run_linear_regression(reg_type: str = "without_reg", alpha: float = 1.0) -> dict:
    """Train and evaluate Linear Regression model with optional regularization.

    Args:
        reg_type: One of 'without_reg', 'lasso', or 'ridge'.
        alpha: Regularization penalty coefficient.

    Returns:
        dict: Evaluation metrics, model parameters, and chart path.
    """
    data = load_data()
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    target = "Salary Package"

    # Drop missing values across features and target
    clean_data = data[features + [target]].dropna()
    X = clean_data[features]
    y = clean_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model based on regularization mode
    reg_type = (reg_type or "without_reg").lower()
    if reg_type == "lasso":
        model = Lasso(alpha=alpha, random_state=42)
        model_name = "Lasso Regularization (L1)"
        chart_slug = "lasso"
    elif reg_type == "ridge":
        model = Ridge(alpha=alpha, random_state=42)
        model_name = "Ridge Regularization (L2)"
        chart_slug = "ridge"
    else:
        model = LinearRegression()
        model_name = "Linear Regression (Without Regularization)"
        chart_slug = "without_reg"
        reg_type = "without_reg"

    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Coefficients summary
    coefficients = [
        {"feature": feat, "weight": float(coef)}
        for feat, coef in zip(features, model.coef_)
    ]

    # Save visualization
    charts_dir = os.path.join(os.path.dirname(__file__), "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    filename = f"linear_regression_{chart_slug}.png"
    filepath = os.path.join(charts_dir, filename)

    plt.figure(figsize=(8.5, 5.2), dpi=110)
    plt.scatter(y_test, y_pred, alpha=0.45, color="#4ba3e3", edgecolors="#2980b9", s=22)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="#e74c3c", linestyle="--", lw=2)
    plt.xlabel("Actual Salary Package")
    plt.ylabel("Predicted Salary Package")
    plt.title(f"{model_name}\nActual vs Predicted (R²: {r2:.4f})", fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close()



    return {
        "model_name": model_name,
        "reg_type": reg_type,
        "alpha": alpha if reg_type in ("lasso", "ridge") else None,
        "features": features,
        "target": target,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "graph": f"charts/{filename}",
    }


# ============================================================
# SALARY PACKAGE PREDICTOR (FOR PLACED STUDENTS)
# ============================================================

_SALARY_REG_MODEL = None


def get_salary_regression_model():
    """Load or train the Ridge regression model specifically on placed students' salary data."""
    global _SALARY_REG_MODEL
    if _SALARY_REG_MODEL is not None:
        return _SALARY_REG_MODEL

    data = load_data()
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    target = "Salary Package"

    # Train specifically on placed students who received salary offers
    placed_data = data[(data["PlacementStatus"] == 1) & (data[target] > 0)][
        features + [target]
    ].dropna()
    X = placed_data[features]
    y = placed_data[target]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    _SALARY_REG_MODEL = {"model": model, "features": features}
    return _SALARY_REG_MODEL


def predict_salary_package(student_metrics: dict) -> dict:
    """Predict the expected starting salary package (CTC in LPA) for a placed student.

    Args:
        student_metrics: Dictionary containing CGPA, AptitudeTestScore, CodingTestScore, MockInterviewScore.

    Returns:
        dict: Projected salary in LPA, tier category, formatted string, and humanized insight.
    """
    reg_bundle = get_salary_regression_model()
    model = reg_bundle["model"]
    features = reg_bundle["features"]

    def safe_float(key, default):
        try:
            return float(student_metrics.get(key, default))
        except (ValueError, TypeError):
            return default

    input_df = pd.DataFrame(
        [
            {
                "CGPA": safe_float("CGPA", 7.5),
                "AptitudeTestScore": safe_float("AptitudeTestScore", 70.0),
                "CodingTestScore": safe_float("CodingTestScore", 65.0),
                "MockInterviewScore": safe_float("MockInterviewScore", 60.0),
            }
        ],
        columns=features,
    )

    raw_prediction = float(model.predict(input_df)[0])

    # Clamp prediction to realistic boundaries [3.0 LPA to 26.0 LPA]
    salary_lpa = round(max(3.0, min(26.0, raw_prediction)), 2)

    # Determine salary bracket and compensation tier
    if salary_lpa >= 15.0:
        tier_name = "Super Dream Tier"
        tier_class = "tier-super-dream"
        tier_badge = "Super Dream Tier (Rs. 15 - 26 LPA)"
        insight = (
            f"Outstanding compensation projection! Backed by your high CGPA ({input_df['CGPA'].iloc[0]:.2f}) "
            f"and interview metrics, you are competitive for premium Product MNC and High-Frequency Tech roles."
        )
    elif salary_lpa >= 8.0:
        tier_name = "Dream Tier"
        tier_class = "tier-dream"
        tier_badge = "Dream Tier (Rs. 8 - 15 LPA)"
        insight = (
            f"Strong compensation projection! Your scores position you well for mid-to-high tier IT "
            f"consulting firms and product engineering cohorts."
        )
    else:
        tier_name = "Standard Tier"
        tier_class = "tier-standard"
        tier_badge = "Standard Tier (Rs. 3 - 8 LPA)"
        insight = (
            f"Solid foundational campus offer. To step up into Dream Tier packages (Rs. 8+ LPA), "
            f"focus on boosting your Coding Test score and Mock Interview performance."
        )

    # Calculate meter percentage (scaled across 3 LPA to 26 LPA range)
    meter_pct = round(((salary_lpa - 3.0) / (26.0 - 3.0)) * 100, 1)

    return {
        "salary_lpa": salary_lpa,
        "formatted": f"Rs. {salary_lpa:.1f} LPA",
        "tier_name": tier_name,
        "tier_badge": tier_badge,
        "tier_class": tier_class,
        "meter_pct": meter_pct,
        "insight": insight,
    }


if __name__ == "__main__":
    for mode in ["without_reg", "lasso", "ridge"]:
        res = run_linear_regression(reg_type=mode)
        print(f"[{res['model_name']}] R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}")

    sample_test = {"CGPA": 8.8, "AptitudeTestScore": 85, "CodingTestScore": 82, "MockInterviewScore": 80}
    sal = predict_salary_package(sample_test)
    print(f"\nSample Salary Prediction: {sal['formatted']} ({sal['tier_name']})")
    print(sal['insight'])
