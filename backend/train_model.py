import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    "amount": np.random.exponential(scale=80, size=n_samples),
    "distance_from_home": np.random.normal(loc=15, scale=8, size=n_samples),
    "hour_of_day": np.random.randint(8, 23, size=n_samples),
    "tx_frequency_24h": np.random.poisson(lam=2, size=n_samples)
})

prob = (
    (data["amount"] > 200).astype(int) * 0.4 +
    (data["distance_from_home"] > 30).astype(int) * 0.3 +
    (data["tx_frequency_24h"] > 5).astype(int) * 0.3
)
data["is_fraud"] = (prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

X = data[["amount", "distance_from_home", "hour_of_day", "tx_frequency_24h"]]
y = data["is_fraud"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")
data.to_csv("reference_data.csv", index=False)
print("✅ Baseline model and reference_data.csv generated successfully.")