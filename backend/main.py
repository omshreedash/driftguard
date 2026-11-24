from fastapi import FastAPI, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import os
import re
import threading
import pandas as pd
import numpy as np
import joblib

try:
    from database import SessionLocal, InferenceLog
except ImportError:
    from backend.database import SessionLocal, InferenceLog

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
REF_DATA_PATH = BASE_DIR / "reference_data.csv"

app = FastAPI(title="DriftGuard MLOps Engine")

cors_origins_env = os.getenv("CORS_ORIGINS", "*").strip()
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def normalize_slashes_middleware(request: Request, call_next):
    # Collapse repeated slashes (e.g. //drift-report when VITE_API_URL has a trailing slash)
    path = request.scope.get("path", "/")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


FEATURES = ["amount", "distance_from_home", "hour_of_day", "tx_frequency_24h"]


def _bootstrap_baseline_artifacts():
    """Generate baseline model and reference dataset if not already present."""
    rng = np.random.default_rng(42)
    n_samples = 2000
    data = pd.DataFrame({
        "amount": rng.exponential(scale=80, size=n_samples),
        "distance_from_home": rng.normal(loc=15, scale=8, size=n_samples),
        "hour_of_day": rng.integers(8, 23, size=n_samples),
        "tx_frequency_24h": rng.poisson(lam=2, size=n_samples),
    })
    prob = (
        (data["amount"] > 200).astype(int) * 0.4
        + (data["distance_from_home"] > 30).astype(int) * 0.3
        + (data["tx_frequency_24h"] > 5).astype(int) * 0.3
    )
    data["is_fraud"] = (prob + rng.normal(0, 0.1, n_samples) > 0.5).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data[FEATURES], data["is_fraud"])

    joblib.dump(model, MODEL_PATH)
    data.to_csv(REF_DATA_PATH, index=False)


class ModelRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        if not MODEL_PATH.exists() or not REF_DATA_PATH.exists():
            _bootstrap_baseline_artifacts()
        self.model = joblib.load(MODEL_PATH)
        self.reference_df = pd.read_csv(REF_DATA_PATH)
        self.version = 1
        self.is_retraining = False
        self.last_trained = datetime.now(timezone.utc).isoformat()

    def _positive_proba(self, probs: np.ndarray) -> np.ndarray:
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
            return probs[:, idx]
        return np.zeros(len(probs), dtype=float)

    def predict(self, df: pd.DataFrame):
        ordered_df = df[FEATURES]
        with self._lock:
            pred = int(self.model.predict(ordered_df)[0])
            prob = float(self._positive_proba(self.model.predict_proba(ordered_df))[0])
            ver = f"v1.{self.version - 1}"
        return pred, prob, ver

    def predict_batch(self, df: pd.DataFrame):
        ordered_df = df[FEATURES]
        with self._lock:
            preds = [int(p) for p in self.model.predict(ordered_df)]
            probs = [float(p) for p in self._positive_proba(self.model.predict_proba(ordered_df))]
            ver = f"v1.{self.version - 1}"
        return preds, probs, ver

    def get_reference(self) -> pd.DataFrame:
        with self._lock:
            return self.reference_df.copy()

    def swap(self, new_model, new_ref_df: pd.DataFrame):
        with self._lock:
            self.model = new_model
            self.reference_df = new_ref_df
            self.version += 1
            self.last_trained = datetime.now(timezone.utc).isoformat()

registry = ModelRegistry()

class TransactionInput(BaseModel):
    amount: float
    distance_from_home: float
    hour_of_day: int
    tx_frequency_24h: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_prediction_bg(tx: dict, pred: int, prob: float):
    db = SessionLocal()
    try:
        entry = InferenceLog(**tx, prediction=pred, fraud_probability=prob)
        db.add(entry)
        db.commit()
    finally:
        db.close()


def log_predictions_batch_bg(rows: list[dict]):
    db = SessionLocal()
    try:
        db.add_all([InferenceLog(**row) for row in rows])
        db.commit()
    finally:
        db.close()


def retrain_worker():
    registry.is_retraining = True
    db = SessionLocal()
    try:
        logs = db.query(InferenceLog).order_by(InferenceLog.id.desc()).limit(500).all()
        if len(logs) < 20:
            return

        new_data = pd.DataFrame([
            {
                "amount": row.amount,
                "distance_from_home": row.distance_from_home,
                "hour_of_day": row.hour_of_day,
                "tx_frequency_24h": row.tx_frequency_24h,
                "is_fraud": int(
                    (row.amount > 250 and row.distance_from_home > 35)
                    or row.tx_frequency_24h > 6
                ),
            }
            for row in logs
        ])

        old_ref = registry.get_reference()
        combined_df = pd.concat([old_ref.tail(1400), new_data], ignore_index=True)

        X = combined_df[FEATURES]
        y = combined_df["is_fraud"]

        new_model = RandomForestClassifier(n_estimators=50, random_state=42)
        new_model.fit(X, y)

        joblib.dump(new_model, MODEL_PATH)
        combined_df.to_csv(REF_DATA_PATH, index=False)

        registry.swap(new_model, combined_df)
    finally:
        registry.is_retraining = False
        db.close()


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "service": "DriftGuard MLOps Engine",
        "status": "ok",
        "model_version": f"v1.{registry.version - 1}",
        "endpoints": ["/health", "/predict", "/predict-batch", "/drift-report", "/retrain", "/reset", "/docs"],
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "model_version": f"v1.{registry.version - 1}",
        "is_retraining": registry.is_retraining,
    }


@app.post("/predict")
def predict(tx: TransactionInput, background_tasks: BackgroundTasks):
    payload = tx.model_dump()
    df = pd.DataFrame([payload])
    pred, prob, model_version = registry.predict(df)

    background_tasks.add_task(log_prediction_bg, payload, pred, prob)
    return {
        "is_fraud": pred,
        "fraud_probability": round(prob, 4),
        "model_version": model_version,
    }


@app.post("/predict-batch")
def predict_batch(transactions: List[TransactionInput], background_tasks: BackgroundTasks):
    if not transactions:
        return {"count": 0, "model_version": f"v1.{registry.version - 1}", "predictions": []}

    payloads = [tx.model_dump() for tx in transactions]
    df = pd.DataFrame(payloads)
    preds, probs, model_version = registry.predict_batch(df)

    log_rows = [
        {**payload, "prediction": pred, "fraud_probability": prob}
        for payload, pred, prob in zip(payloads, preds, probs)
    ]
    background_tasks.add_task(log_predictions_batch_bg, log_rows)

    return {
        "count": len(payloads),
        "model_version": model_version,
        "predictions": [
            {"is_fraud": pred, "fraud_probability": round(prob, 4)}
            for pred, prob in zip(preds, probs)
        ],
    }


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if registry.is_retraining:
        return {"status": "ALREADY_RUNNING", "model_version": f"v1.{registry.version - 1}"}
    background_tasks.add_task(retrain_worker)
    return {"status": "RETRAINING_STARTED", "current_version": f"v1.{registry.version - 1}"}

@app.get("/drift-report")
def get_drift_report(window: int = 100, db: Session = Depends(get_db)):
    logs = db.query(InferenceLog).order_by(InferenceLog.id.desc()).limit(window).all()
    ref_df = registry.get_reference()

    base_meta = {
        "model_version": f"v1.{registry.version - 1}",
        "is_retraining": registry.is_retraining,
        "last_trained": registry.last_trained,
    }

    if len(logs) < 15:
        return {**base_meta, "status": "INSUFFICIENT_DATA", "sample_count": len(logs), "features": {}}

    current_df = pd.DataFrame([{f: getattr(row, f) for f in FEATURES} for row in logs])
    feature_reports = {}
    drifted_count = 0

    for feat in FEATURES:
        ref_vals = ref_df[feat].values
        curr_vals = current_df[feat].values

        ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
        is_drifted = bool(p_value < 0.05 and ks_stat > 0.15)
        if is_drifted:
            drifted_count += 1

        min_val = float(min(ref_vals.min(), curr_vals.min()))
        max_val = float(max(ref_vals.max(), curr_vals.max()))
        if max_val <= min_val:
            max_val = min_val + 1.0
        bins = np.linspace(min_val, max_val, 12)
        ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_vals, bins=bins, density=True)

        histogram_data = [
            {
                "bin": f"{round(bins[i], 1)}",
                "baseline": round(float(ref_hist[i]), 4),
                "production": round(float(curr_hist[i]), 4),
            }
            for i in range(len(ref_hist))
        ]

        feature_reports[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drifted": is_drifted,
            "ref_mean": round(float(ref_vals.mean()), 2),
            "curr_mean": round(float(curr_vals.mean()), 2),
            "histogram": histogram_data,
        }

    overall_status = "CRITICAL_DRIFT" if drifted_count >= 2 else ("WARNING" if drifted_count == 1 else "HEALTHY")
    fraud_rate = round(float(np.mean([l.prediction for l in logs]) * 100), 2)

    return {
        **base_meta,
        "status": overall_status,
        "sample_count": len(logs),
        "drifted_features": drifted_count,
        "predicted_fraud_rate": fraud_rate,
        "features": feature_reports,
    }

@app.post("/reset")
def reset_logs(db: Session = Depends(get_db)):
    db.query(InferenceLog).delete()
    db.commit()
    return {"message": "Production telemetry cleared."}