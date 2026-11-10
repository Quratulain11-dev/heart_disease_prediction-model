import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "heart_extended.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "heart_disease_model.joblib")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
NUMERIC_FEATURES = [column for column in FEATURES if column not in CATEGORICAL_FEATURES]


def load_training_data():
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["target", "source"])
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Training data contains no records.")
    if not data["target"].isin([0, 1]).all():
        raise ValueError("Training target must contain only 0 and 1.")

    data = data.drop_duplicates(subset=FEATURES + ["target"]).reset_index(drop=True)
    return data


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def make_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])


PARAMETERS = {
    "classifier__C": [0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"],
}


def find_best_model(X, y, groups, cv, preprocessor):
    search = GridSearchCV(
        make_model(preprocessor),
        PARAMETERS,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    return search


def main():
    data = load_training_data()
    X = data[FEATURES]
    y = data["target"].astype(int)
    groups = data["source"]
    cv = LeaveOneGroupOut()

    # Keep each cohort out of both hyperparameter selection and evaluation.
    probabilities = np.full(len(data), np.nan)
    for train_idx, test_idx in cv.split(X, y, groups):
        fold_search = find_best_model(
            X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx], cv,
            make_preprocessor(),
        )
        probabilities[test_idx] = fold_search.best_estimator_.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)
    print(f"Nested leave-one-cohort-out accuracy: {accuracy_score(y, predictions):.3f}")
    print(f"Nested leave-one-cohort-out balanced accuracy: {balanced_accuracy_score(y, predictions):.3f}")
    print(f"Nested leave-one-cohort-out F1: {f1_score(y, predictions):.3f}")
    print(f"Nested leave-one-cohort-out ROC-AUC: {roc_auc_score(y, probabilities):.3f}")
    majority_accuracy = y.value_counts(normalize=True).max()
    print(f"Majority-class baseline accuracy: {majority_accuracy:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))
    for source in sorted(groups.unique()):
        mask = groups.to_numpy() == source
        print(
            f"  {source}: n={int(mask.sum())}, "
            f"accuracy={accuracy_score(y[mask], predictions[mask]):.3f}"
        )

    final_search = find_best_model(
        X,
        y,
        groups,
        cv,
        make_preprocessor(),
    )
    best_model = final_search.best_estimator_
    print(
        "Selected Logistic Regression for full-data fit: "
        f"{final_search.best_params_} "
        f"(leave-one-cohort-out CV accuracy={final_search.best_score_:.3f})"
    )
    bundle = {
        "model": best_model,
        "model_name": "Logistic Regression",
        "features": FEATURES,
        "cat_cols": CATEGORICAL_FEATURES,
        "input_mode": "raw",
        "training_rows": len(data),
        "training_sources": sorted(groups.unique()),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
