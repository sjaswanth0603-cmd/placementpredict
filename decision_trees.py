import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from load_data import load_data

# Readable display names for each supported algorithm
ALGORITHM_DISPLAY_NAMES = {
    "bagging": "Bagging Classifier (Ensemble of Decision Trees)",
    "id3": "ID3 (Decision Tree - Information Gain)",
    "adaboost": "AdaBoost Classifier",
    "xgboost": "XGBoost Classifier",
    "lightgbm": "LightGBM Classifier",
    "random_forest": "Random Forest Classifier",
    "gradient_boost": "Gradient Boosting Classifier",
}

# The 8 academic and technical metrics used to predict placement
FEATURE_NAMES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore",
    "SoftSkillsRating",
    "Internships",
    "Projects",
    "Certifications",
]

TARGET_NAME = "PlacementStatus"

# In-memory cache for trained models so single-student predictions respond instantly
_TRAINED_MODELS = {}


def get_trained_model(algorithm: str = "bagging"):
    """Fetch or train a model for the given algorithm, caching it in memory."""
    algo = (algorithm or "bagging").lower()
    if algo not in ALGORITHM_DISPLAY_NAMES:
        algo = "bagging"

    if algo in _TRAINED_MODELS:
        return _TRAINED_MODELS[algo]

    # Load dataset and prepare clean training data
    data = load_data()
    clean_data = data[FEATURE_NAMES + [TARGET_NAME]].dropna()
    X = clean_data[FEATURE_NAMES]
    y = clean_data[TARGET_NAME].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize the chosen classifier
    if algo == "bagging":
        # Bagging (Bootstrap Aggregation) trains multiple decision trees on random
        # bootstrap subsamples with replacement, then averages their votes.
        # This significantly reduces model variance and stabilizes performance.
        base_tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=6,
            random_state=42
        )
        model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=35,
            random_state=42,
            n_jobs=-1
        )
    elif algo == "adaboost":
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
    elif algo == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=5, eval_metric="logloss", random_state=42
        )
    elif algo == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=50, max_depth=5, random_state=42, verbose=-1
        )
    elif algo == "random_forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    elif algo == "gradient_boost":
        model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    else:
        # Classical ID3 style decision tree using Entropy / Information Gain
        algo = "id3"
        model = DecisionTreeClassifier(
            criterion="entropy", max_depth=6, random_state=42
        )

    # Train the model
    model.fit(X_train, y_train)

    # Cache the trained components
    cached_bundle = {
        "model": model,
        "algo": algo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "raw_data": data,
    }
    _TRAINED_MODELS[algo] = cached_bundle
    return cached_bundle


def run_decision_trees(algorithm: str = "bagging") -> dict:
    """Train and evaluate Decision Tree & ensemble models for student placement.

    Args:
        algorithm: One of 'bagging', 'id3', 'adaboost', 'xgboost', 'lightgbm',
                   'random_forest', or 'gradient_boost'.

    Returns:
        dict: Comprehensive metrics, confusion matrix, feature importances,
              sample student predictions, and chart file path.
    """
    bundle = get_trained_model(algorithm)
    model = bundle["model"]
    algo = bundle["algo"]
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    data = bundle["raw_data"]

    model_name = ALGORITHM_DISPLAY_NAMES.get(algo, "Decision Tree Algorithm")

    # Evaluate test set predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Compute key classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    # Extract feature importances
    feature_importances = []
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif (
        hasattr(model, "estimators_")
        and len(model.estimators_) > 0
        and hasattr(model.estimators_[0], "feature_importances_")
    ):
        # In a BaggingClassifier, average feature importances across all base trees
        raw_importances = np.mean(
            [tree.feature_importances_ for tree in model.estimators_], axis=0
        )
    else:
        raw_importances = None

    if raw_importances is not None:
        feature_importances = sorted(
            [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in zip(FEATURE_NAMES, raw_importances)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

    # Generate and save feature importance chart
    charts_dir = os.path.join(os.path.dirname(__file__), "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    filename = f"dt_{algo}.png"
    filepath = os.path.join(charts_dir, filename)

    plt.figure(figsize=(8.5, 5.2), dpi=110)
    if feature_importances:
        feats = [item["feature"] for item in reversed(feature_importances)]
        imps = [item["importance"] for item in reversed(feature_importances)]
        plt.barh(feats, imps, color="#3498db", edgecolor="#2980b9")
        plt.xlabel("Relative Importance Score")
        plt.title(f"{model_name}\nFeature Importance (Accuracy: {acc * 100:.2f}%)", fontsize=12)
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Placed (0)", "Placed (1)"],
            yticklabels=["Not Placed (0)", "Placed (1)"],
        )
        plt.title(f"{model_name} Confusion Matrix", fontsize=12)

    plt.tight_layout()
    plt.savefig(filepath, dpi=110, bbox_inches="tight")
    plt.close()

    # Build human-readable sample predictions for test set students
    # This lets the user see whether actual students in the dataset are placed or not
    sample_indices = X_test.head(10).index
    sample_students = []
    for i, original_idx in enumerate(sample_indices):
        student_id = (
            int(data.loc[original_idx, "StudentID"])
            if "StudentID" in data.columns
            else i + 1
        )
        actual = int(y_test.iloc[i])
        predicted = int(y_pred[i])
        prob_placed = (
            float(y_proba[i][1]) * 100 if y_proba is not None else (100.0 if predicted == 1 else 0.0)
        )

        actual_salary = (
            float(data.loc[original_idx, "Salary Package"])
            if "Salary Package" in data.columns
            else 0.0
        )
        salary_display = (
            f"Rs. {actual_salary:.1f} LPA"
            if (actual == 1 and actual_salary > 0)
            else "—"
        )

        sample_students.append({
            "student_id": student_id,
            "cgpa": float(X_test.iloc[i]["CGPA"]),
            "coding": float(X_test.iloc[i]["CodingTestScore"]),
            "aptitude": float(X_test.iloc[i]["AptitudeTestScore"]),
            "mock_interview": float(X_test.iloc[i]["MockInterviewScore"]),
            "actual_status": "Placed" if actual == 1 else "Not Placed",
            "predicted_status": "Placed" if predicted == 1 else "Not Placed",
            "probability": round(prob_placed, 1),
            "salary_display": salary_display,
            "is_correct": actual == predicted,
        })

    return {
        "model_name": model_name,
        "algorithm": algo,
        "features": FEATURE_NAMES,
        "target": TARGET_NAME,
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
        "feature_importances": feature_importances,
        "sample_students": sample_students,
        "graph": f"charts/{filename}",
    }


def predict_student_status(student_input: dict, algorithm: str = "bagging") -> dict:
    """Predict whether a student will be placed or not based on academic and technical scores.

    Args:
        student_input: Dictionary with student metrics:
                       CGPA, AptitudeTestScore, CodingTestScore, MockInterviewScore,
                       SoftSkillsRating, Internships, Projects, Certifications.
        algorithm: Algorithm to use for prediction (defaults to 'bagging').

    Returns:
        dict: Human-friendly placement verdict ('Placed' or 'Not Placed'),
              probability score, confidence level, and actionable feedback.
    """
    bundle = get_trained_model(algorithm)
    model = bundle["model"]
    algo = bundle["algo"]

    # Parse and safely clamp input values to expected ranges
    def safe_float(key, default, min_val, max_val):
        try:
            val = float(student_input.get(key, default))
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default

    student_data = {
        "CGPA": safe_float("CGPA", 7.5, 0.0, 10.0),
        "AptitudeTestScore": safe_float("AptitudeTestScore", 70.0, 0.0, 100.0),
        "CodingTestScore": safe_float("CodingTestScore", 65.0, 0.0, 100.0),
        "MockInterviewScore": safe_float("MockInterviewScore", 60.0, 0.0, 100.0),
        "SoftSkillsRating": safe_float("SoftSkillsRating", 3.5, 1.0, 5.0),
        "Internships": safe_float("Internships", 1.0, 0.0, 10.0),
        "Projects": safe_float("Projects", 2.0, 0.0, 20.0),
        "Certifications": safe_float("Certifications", 1.0, 0.0, 20.0),
    }

    # Convert to DataFrame matching training feature columns
    input_df = pd.DataFrame([student_data], columns=FEATURE_NAMES)

    # Perform prediction
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

    if probabilities is not None:
        placed_probability = float(probabilities[1]) * 100
    else:
        placed_probability = 100.0 if prediction == 1 else 0.0

    is_placed = prediction == 1
    status = "Placed" if is_placed else "Not Placed"

    # Determine confidence description
    if placed_probability >= 80 or placed_probability <= 20:
        confidence = "High Confidence"
    elif placed_probability >= 65 or placed_probability <= 35:
        confidence = "Moderate Confidence"
    else:
        confidence = "Borderline Decision"

    # Generate humanized recommendations and profile strengths
    strengths = []
    improvements = []

    if student_data["CGPA"] >= 8.0:
        strengths.append(f"Excellent academic standing with CGPA of {student_data['CGPA']:.2f}")
    elif student_data["CGPA"] < 6.5:
        improvements.append(f"Academic CGPA ({student_data['CGPA']:.2f}) is lower than average campus cutoff")

    if student_data["MockInterviewScore"] >= 65:
        strengths.append(f"Strong interview readiness score ({student_data['MockInterviewScore']:.1f}/100)")
    else:
        improvements.append(f"Mock interview performance ({student_data['MockInterviewScore']:.1f}/100) needs practice")

    if student_data["CodingTestScore"] >= 70:
        strengths.append(f"Impressive coding test benchmark ({student_data['CodingTestScore']:.1f}/100)")
    elif student_data["CodingTestScore"] < 50:
        improvements.append(f"Coding test score ({student_data['CodingTestScore']:.1f}/100) requires targeted DSA prep")

    if student_data["Internships"] >= 1:
        strengths.append(f"Valuable industrial exposure with {int(student_data['Internships'])} internship(s)")
    else:
        improvements.append("No prior internship experience on record")

    if student_data["Projects"] >= 2:
        strengths.append(f"Hands-on experience demonstrated with {int(student_data['Projects'])} portfolio projects")

    # Formulate humanized summary advice
    if is_placed:
        summary_text = (
            f"Great news! The {ALGORITHM_DISPLAY_NAMES.get(algo, 'model')} predicts this student is "
            f"likely to be PLACED with a {placed_probability:.1f}% probability. "
            f"The candidate has strong fundamentals, especially in interview performance and academics."
        )
    else:
        summary_text = (
            f"Based on the analysis, this student is currently at risk of being NOT PLACED "
            f"({placed_probability:.1f}% placement chance). Focusing on mock interview drills and "
            f"improving coding test scores will significantly boost their odds."
        )

    return {
        "status": status,
        "is_placed": is_placed,
        "probability": round(placed_probability, 1),
        "confidence": confidence,
        "algorithm_used": ALGORITHM_DISPLAY_NAMES.get(algo, algo),
        "student_data": student_data,
        "strengths": strengths,
        "improvements": improvements,
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATING DECISION TREES & BAGGING ENSEMBLE MODELS")
    print("=" * 70)

    # Train and compare algorithms including Bagging
    algorithms_to_test = [
        "bagging",
        "id3",
        "random_forest",
        "adaboost",
        "xgboost",
        "lightgbm",
        "gradient_boost",
    ]

    for alg in algorithms_to_test:
        res = run_decision_trees(algorithm=alg)
        print(
            f"[{res['model_name']:<45}] "
            f"Accuracy: {res['accuracy'] * 100:.2f}% | "
            f"F1: {res['f1_score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("TESTING SINGLE STUDENT PLACEMENT PREDICTIONS")
    print("=" * 70)

    # Sample student A: Strong candidate
    student_a = {
        "CGPA": 8.5,
        "AptitudeTestScore": 85.0,
        "CodingTestScore": 80.0,
        "MockInterviewScore": 75.0,
        "SoftSkillsRating": 4.5,
        "Internships": 2,
        "Projects": 3,
        "Certifications": 2,
    }
    pred_a = predict_student_status(student_a, algorithm="bagging")
    print(f"\nStudent A (High Achiever): {pred_a['status']} ({pred_a['probability']}% chance)")
    print(f"Advice: {pred_a['summary_text']}")

    # Sample student B: Needs improvement candidate
    student_b = {
        "CGPA": 5.8,
        "AptitudeTestScore": 45.0,
        "CodingTestScore": 40.0,
        "MockInterviewScore": 38.0,
        "SoftSkillsRating": 2.5,
        "Internships": 0,
        "Projects": 1,
        "Certifications": 0,
    }
    pred_b = predict_student_status(student_b, algorithm="bagging")
    print(f"\nStudent B (At-Risk Profile): {pred_b['status']} ({pred_b['probability']}% chance)")
    print(f"Advice: {pred_b['summary_text']}")
