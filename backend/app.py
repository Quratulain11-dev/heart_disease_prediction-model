import math
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "heart_disease_model.joblib")


# Load the model bundle saved by the notebook
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
cat_cols = bundle.get("cat_cols", [])
input_mode = bundle.get("input_mode", "encoded")
THRESHOLD = 0.5

FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Numeric ranges (same as the frontend)
RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
}

# Allowed values for categorical fields
CHOICES = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {0, 1, 2, 3},
    "restecg": {0, 1, 2},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3},
    "thal": {1, 2, 3},
}

app = Flask(__name__)

FRONTEND_ORIGINS = {
    "http://localhost:5501",
    "http://127.0.0.1:5501",
}


@app.after_request
def add_local_frontend_cors(response):
    origin = request.headers.get("Origin")
    if origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers.add("Vary", "Origin")
    return response


def make_features(patient, features, cat_cols):
    """Same encoding used in the notebook."""
    row = {f: 0 for f in features}
    for col, value in patient.items():
        if col in cat_cols:
            key = f"{col}_{int(value)}"
            if key in row:
                row[key] = 1
        elif col in row:
            row[col] = value
    return pd.DataFrame([row])[features]


def validate(data):
    """Returns (cleaned_patient, errors)."""
    errors = []
    patient = {}

    for field in FIELDS:
        if field not in data or data[field] in ("", None):
            errors.append(f"{field} is required.")
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue

        if not math.isfinite(value):
            errors.append(f"{field} must be a finite number.")
            continue

        if field in CHOICES:
            if value != int(value) or int(value) not in CHOICES[field]:
                errors.append(f"{field} must be one of {sorted(CHOICES[field])}.")
                continue
        elif field in RANGES:
            lo, hi = RANGES[field]
            if not (lo <= value <= hi):
                errors.append(f"{field} must be between {lo} and {hi}.")
                continue

        patient[field] = value

    return patient, errors


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify(error="Send patient data to this endpoint with a POST request."), 405

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="A JSON object with patient data is required."), 400

    patient, errors = validate(data)
    if errors:
        return jsonify(error=" ".join(errors)), 400

    if input_mode == "raw":
        X_new = pd.DataFrame([{field: patient[field] for field in features}])
    else:
        X_new = make_features(patient, features, cat_cols)
    prob = float(model.predict_proba(X_new)[0, 1])

    return jsonify(
        probability=prob,
        prediction=int(prob >= THRESHOLD),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    