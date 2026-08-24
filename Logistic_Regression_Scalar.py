import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(filename):

    data = pd.read_csv(filename)

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    X = data[features]

    y = data["PlacementStatus"]

    # Remove missing values
    data = pd.concat(
        [X, y],
        axis=1
    ).dropna()

    X = data[features]

    y = data["PlacementStatus"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    name
):

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Create Logistic Regression model
    model = LogisticRegression(
        max_iter=1000
    )

    # Train
    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    train_pred = model.predict(
        X_train_scaled
    )

    test_pred = model.predict(
        X_test_scaled
    )

    # Accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    print(
        name,
        "Train Accuracy:",
        round(train_accuracy, 4)
    )

    print(
        name,
        "Test Accuracy:",
        round(test_accuracy, 4)
    )

    # Confusion Matrix
    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            test_pred
        )
    )

    return model, scaler


def main():

    filename = "placement_predict_50k Dataset (3)(in).csv"

    # Load data
    X, y = load_data(
        filename
    )

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Train and evaluate
    model, scaler = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        "Logistic Regression"
    )


if __name__ == "__main__":

    main()