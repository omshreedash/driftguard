from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp
from datetime import datetime, timezone
import threading
import pandas as pd
import numpy as np
import joblib

from database import SessionLocal, InferenceLog

app = FastAPI(title="DriftGuard MLOps Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = ["amount", "distance_from_home", "hour_of_day", "tx_frequency_24h"]

class ModelRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self.model = joblib.load("fraud_model.pkl")
        self.reference_df = pd.read_csv("reference_data.csv")
        self.version = 1
        self.is_retraining = False
        self.last_trained = datetime.now(timezone.utc).isoformat()

    def predict(self, df: pd.DataFrame):
        with self._lock:
            pred = int(self.model.predict(df)[0])
            prob = float(self.model.predict_proba(df)[0][1])
            ver = f"v1.{self.version - 1}"
        return pred, prob, ver

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

        joblib.dump(new_model, "fraud_model.pkl")
        combined_df.to_csv("reference_data.csv", index=False)

        registry.swap(new_model, combined_df)
    finally:
        registry.is_retraining = False
        db.close()

@app.post("/predict")
def predict(tx: TransactionInput, background_tasks: BackgroundTasks):
    df = pd.DataFrame([tx.model_dump()])
    pred, prob, model_version = registry.predict(df)

    background_tasks.add_task(log_prediction_bg, tx.model_dump(), pred, prob)
    return {
        "is_fraud": pred,
        "fraud_probability": round(prob, 4),
        "model_version": model_version,
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

        min_val = min(ref_vals.min(), curr_vals.min())
        max_val = max(ref_vals.max(), curr_vals.max())
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