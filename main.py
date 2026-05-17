import os
import time
import random
from typing import Optional

import mlflow
import numpy as np
from fastapi import FastAPI, Query
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:/app/mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hw8-monitoring-iris-service")

app = FastAPI(title="HW8 ML Monitoring Service", version=APP_VERSION)

REQUEST_COUNT = Counter(
    "ml_service_requests_total",
    "Total number of HTTP requests processed by the ML service",
    ["endpoint", "status"],
)
ERROR_COUNT = Counter(
    "ml_service_errors_total",
    "Total number of failed requests in the ML service",
    ["endpoint"],
)
LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10),
)
PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Number of predictions by predicted class",
    ["class_name"],
)
MODEL_ACCURACY = Gauge("ml_model_accuracy", "Latest measured model accuracy on validation data")
DATA_DRIFT_SCORE = Gauge("ml_data_drift_score", "Synthetic data drift score, higher is worse")
DATA_QUALITY_INCIDENTS = Gauge("data_quality_incidents_total", "Number of detected Data Quality incidents")
MODEL_VERSION = Gauge("ml_model_version_info", "Model version as numeric build marker")

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
class_names = load_iris().target_names.tolist()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
MODEL_ACCURACY.set(float(accuracy_score(y_val, val_pred)))
MODEL_VERSION.set(1)


def _drift_score(reference: np.ndarray, current: np.ndarray) -> float:
    ref_mean = reference.mean(axis=0)
    ref_std = reference.std(axis=0) + 1e-8
    cur_mean = current.mean(axis=0)
    return float(np.mean(np.abs((cur_mean - ref_mean) / ref_std)))


@app.get("/")
def root():
    return {
        "service": "HW8 ML Monitoring Service",
        "version": APP_VERSION,
        "endpoints": ["/predict", "/simulate_drift", "/simulate_dq_incident", "/metrics"],
    }


@app.get("/predict")
def predict(
    sepal_length: float = Query(5.1),
    sepal_width: float = Query(3.5),
    petal_length: float = Query(1.4),
    petal_width: float = Query(0.2),
    slow: bool = Query(False, description="Set true to simulate latency alert"),
    fail: bool = Query(False, description="Set true to simulate error-rate alert"),
):
    start = time.time()
    endpoint = "/predict"

    try:
        if slow:
            time.sleep(2.2)
        if fail:
            raise RuntimeError("Synthetic failure for monitoring demo")

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_idx = int(model.predict(features)[0])
        pred_proba = model.predict_proba(features)[0].tolist()
        pred_class = class_names[pred_idx]

        PREDICTION_COUNT.labels(class_name=pred_class).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()

        with mlflow.start_run(run_name="prediction", nested=True):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("app_version", APP_VERSION)
            mlflow.log_metric("prediction_class_index", pred_idx)
            mlflow.log_metric("request_latency_seconds", time.time() - start)

        return {
            "prediction": pred_class,
            "class_index": pred_idx,
            "probabilities": dict(zip(class_names, pred_proba)),
            "features": dict(zip(feature_names, features[0].tolist())),
        }
    except Exception as exc:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        raise exc
    finally:
        LATENCY.observe(time.time() - start)


@app.post("/simulate_drift")
def simulate_drift(multiplier: float = 5.0):
    """Creates a synthetic current batch shifted away from reference data."""
    current = X_val.copy() * multiplier
    score = _drift_score(X_train, current)
    DATA_DRIFT_SCORE.set(score)
    with mlflow.start_run(run_name="drift_check", nested=True):
        mlflow.log_metric("data_drift_score", score)
        mlflow.log_param("current_multiplier", multiplier)
    return {
        "reference_batch": "iris train split",
        "current_batch": f"iris validation split multiplied by {multiplier}",
        "data_drift_score": score,
        "is_drift_detected": score > 1.0,
    }


@app.post("/simulate_dq_incident")
def simulate_dq_incident(count: int = 1):
    DATA_QUALITY_INCIDENTS.set(count)
    with mlflow.start_run(run_name="dq_incident", nested=True):
        mlflow.log_metric("data_quality_incidents_total", count)
    return {"data_quality_incidents_total": count, "alert_expected": count > 0}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
