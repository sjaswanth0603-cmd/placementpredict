import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_data


def run_logistic_regression(reg_type: str = "without_reg", c_val: float = 1.0) -> dict:
    """Train and evaluate Logistic Regression model with optional regularization.

    Args:
        reg_type: One of 'without_reg', 'lasso', or 'ridge'.
        c_val: Inverse regularization parameter C.

    Returns:
        dict: Evaluation metrics, confusion matrix, coefficients, and chart path.
    """
    data = load_data()
    features = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
    target = "PlacementStatus"

    clean_data = data[features + [target]].dropna()
    X = clean_data[features]
    y = clean_data[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    import warnings
    reg_type = (reg_type or "without_reg").lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if reg_type == "lasso":
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=c_val, max_iter=1000, random_state=42
            )
            model_name = "Logistic Regression with Lasso Regularization (L1)"
            chart_slug = "lasso"
        elif reg_type == "ridge":
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", C=c_val, max_iter=1000, random_state=42
            )
            model_name = "Logistic Regression with Ridge Regularization (L2)"
            chart_slug = "ridge"
        else:
            model = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000, random_state=42
            )
            model_name = "Logistic Regression (Without Regularization)"
            chart_slug = "without_reg"
            reg_type = "without_reg"

        model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    coefficients = [
        {"feature": feat, "weight": float(coef)}
        for feat, coef in zip(features, model.coef_[0])
    ]

    # Save Confusion Matrix Heatmap
    charts_dir = os.path.join(os.path.dirname(__file__), "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    filename = f"logistic_regression_{chart_slug}.png"
    filepath = os.path.join(charts_dir, filename)

    plt.figure(figsize=(7.2, 5.2), dpi=110)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Placed (0)", "Placed (1)"],
        yticklabels=["Not Placed (0)", "Placed (1)"],
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"{model_name}\nConfusion Matrix (Accuracy: {acc * 100:.2f}%)", fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close()



    return {
        "model_name": model_name,
        "reg_type": reg_type,
        "c_val": c_val if reg_type in ("lasso", "ridge") else None,
        "features": features,
        "target": target,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "coefficients": coefficients,
        "intercept": float(model.intercept_[0]),
        "graph": f"charts/{filename}",
    }


if __name__ == "__main__":
    for mode in ["without_reg", "lasso", "ridge"]:
        res = run_logistic_regression(reg_type=mode)
        print(f"[{res['model_name']}] Accuracy: {res['accuracy'] * 100:.2f}%")