from flask import Flask, render_template

from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from Linear_Regression_Sklearn import run_linear_regression


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

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
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
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
        "eda.html",
        active="eda",
        results=eda_output,
        error=error
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
        error=error
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/linear-regression")
def linear_regression():

    error = None
    results = None

    try:

        results = run_linear_regression()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"


    return render_template(
        "linear_regression.html",
        active="linear-regression",
        results=results,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )