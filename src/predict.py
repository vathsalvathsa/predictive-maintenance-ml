"""Load model and expose predict_single / predict_batch."""

import joblib
import pandas as pd
import numpy as np
from src.features import add_features

MODEL_PATH = "models/best_model.joblib"
_cache = {}


def load_model(path=MODEL_PATH):
    if path not in _cache:
        _cache[path] = joblib.load(path)
    return _cache[path]


def _prep(df, features):
    if "machine_id" not in df.columns:
        df = df.copy(); df["machine_id"] = 1
    if "timestamp_index" not in df.columns:
        df = df.copy(); df["timestamp_index"] = range(len(df))
    df = add_features(df)
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
    return df[features]


def predict_single(reading):
    bundle = load_model()
    model, features = bundle["model"], bundle["features"]
    df = pd.DataFrame([reading])
    X = _prep(df, features)
    prob = float(model.predict_proba(X)[0, 1])
    return {"failure_probability": prob,
            "risk": "HIGH" if prob >= 0.5 else "LOW",
            "model_used": bundle["name"]}


def predict_batch(df):
    bundle = load_model()
    model, features = bundle["model"], bundle["features"]
    X = _prep(df, features)
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["failure_probability"] = probs
    out["risk"] = np.where(probs >= 0.5, "HIGH", "LOW")
    return out
