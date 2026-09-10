from flask import Flask, render_template, request
from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from Linear_Regression_Sklearn import run_linear_regression, predict_salary_package
from Logistic_Regression_Scalar import run_logistic_regression
from decision_trees import run_decision_trees, predict_student_status

app = Flask(__name__)


# ============================================================
# HOME & DATA LOADING
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html", active="data-loading", summary=summary, error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():
    error = None
    eda_output = None
    try:
        eda_output = run_eda()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html", active="eda", results=eda_output, error=error
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():
    error = None
    preprocessing_output = None
    try:
        preprocessing_output = run_preprocessing()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=preprocessing_output,
        error=error,
    )


# ============================================================
# LINEAR REGRESSION (WITHOUT REG, LASSO, RIDGE)
# ============================================================

@app.route("/linear-regression")
def linear_regression():
    error = None
    results = None
    reg_type = request.args.get("reg_type", "without_reg").lower()
    if reg_type not in ("without_reg", "lasso", "ridge"):
        reg_type = "without_reg"

    try:
        results = run_linear_regression(reg_type=reg_type)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        selected_reg=reg_type,
        error=error,
    )


# ============================================================
# LOGISTIC REGRESSION (WITHOUT REG, LASSO, RIDGE)
# ============================================================

@app.route("/logistic-regression")
def logistic_regression():
    error = None
    results = None
    reg_type = request.args.get("reg_type", "without_reg").lower()
    if reg_type not in ("without_reg", "lasso", "ridge"):
        reg_type = "without_reg"

    try:
        results = run_logistic_regression(reg_type=reg_type)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "logistic_regression.html",
        active="logistic-regression",
        results=results,
        selected_reg=reg_type,
        error=error,
    )


# ============================================================
# DECISION TREES & BAGGING ENSEMBLE
# ============================================================

@app.route("/decision-trees")
def decision_trees():
    error = None
    results = None
    algorithm = request.args.get("algorithm", "bagging").lower()
    valid_algos = (
        "bagging",
        "id3",
        "adaboost",
        "xgboost",
        "lightgbm",
        "random_forest",
        "gradient_boost",
    )
    if algorithm not in valid_algos:
        algorithm = "bagging"

    try:
        results = run_decision_trees(algorithm=algorithm)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "decision_trees.html",
        active="decision-trees",
        results=results,
        selected_algo=algorithm,
        error=error,
    )


# ============================================================
# STUDENT PLACEMENT PREDICTOR (DEDICATED DASHBOARD PAGE)
# ============================================================

@app.route("/placement-predictor", methods=["GET", "POST"])
def placement_predictor():
    error = None
    prediction_result = None
    sample_students = None

    if request.method == "POST":
        algorithm = request.form.get("algorithm", "bagging").lower()
    else:
        algorithm = request.args.get("algorithm", "bagging").lower()

    valid_algos = (
        "bagging",
        "id3",
        "adaboost",
        "xgboost",
        "lightgbm",
        "random_forest",
        "gradient_boost",
    )
    if algorithm not in valid_algos:
        algorithm = "bagging"

    student_inputs = {
        "CGPA": "7.8",
        "AptitudeTestScore": "72",
        "CodingTestScore": "68",
        "MockInterviewScore": "65",
        "SoftSkillsRating": "3.8",
        "Internships": "1",
        "Projects": "2",
        "Certifications": "2",
    }

    if request.method == "POST":
        for field in student_inputs.keys():
            if field in request.form:
                student_inputs[field] = request.form.get(field, student_inputs[field])

        try:
            prediction_result = predict_student_status(
                student_input=student_inputs, algorithm=algorithm
            )
            # If the student is placed, compute their projected salary package (LPA)
            if prediction_result and prediction_result.get("is_placed"):
                prediction_result["salary_info"] = predict_salary_package(student_inputs)
            else:
                prediction_result["salary_info"] = None
        except Exception as e:
            error = f"Error evaluating student profile: {e}"

    # Load sample test set student comparisons for the selected algorithm
    try:
        model_eval = run_decision_trees(algorithm=algorithm)
        sample_students = model_eval.get("sample_students", [])
    except Exception:
        sample_students = []

    return render_template(
        "predictor.html",
        active="placement-predictor",
        selected_algo=algorithm,
        student_inputs=student_inputs,
        prediction_result=prediction_result,
        sample_students=sample_students,
        error=error,
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)