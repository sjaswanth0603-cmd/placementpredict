import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt

from load_data import load_data


# ============================================================
# LINEAR REGRESSION
# ============================================================

def run_linear_regression():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_data()


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    target = "Salary Package"


    # --------------------------------------------------------
    # SELECT DATA
    # --------------------------------------------------------

    x = data[features]

    y = data[target]


    # --------------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    data = pd.concat([x, y], axis=1).dropna()


    x = data[features]

    y = data[target]


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = LinearRegression()


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model.fit(
        x_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        x_test
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test,
        y_pred
    )


    root_mse = sqrt(mse)


    r2 = r2_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED GRAPH
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_test,
        y_pred
    )

    plt.xlabel(
        "Actual Salary Package"
    )

    plt.ylabel(
        "Predicted Salary Package"
    )

    plt.title(
        "Actual vs Predicted Salary Package"
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.tight_layout()


    # Save graph for dashboard

    graph_path = "static/charts/linear_regression.png"

    plt.savefig(
        graph_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "model": model,

        "features": features,

        "target": target,

        "training_rows": len(x_train),

        "testing_rows": len(x_test),

        "mse": mse,

        "rmse": root_mse,

        "r2": r2,

        "graph": "charts/linear_regression.png"
    }