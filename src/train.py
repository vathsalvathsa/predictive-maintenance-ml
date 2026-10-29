"""Train KNN baseline vs LR vs RF. Save best by Recall."""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

from src.features import add_features, get_feature_columns

RAW_PATH = "data/sensor_data.csv"
FEATURED_PATH = "data/sensor_data_features.csv"
FIG_DIR = "docs/figures"
MODEL_DIR = "models"
TARGET = "failure_within_24h"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare():
    if not os.path.exists(FEATURED_PATH):
        df = pd.read_csv(RAW_PATH)
        df = add_features(df)
        df.to_csv(FEATURED_PATH, index=False)
    else:
        df = pd.read_csv(FEATURED_PATH)
    features = get_feature_columns(df)
    return df, features


def build_models():
    return {
        "KNN (baseline)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
    }


def plot_confusion(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fail", "Fail"],
                yticklabels=["No Fail", "Fail"])
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"cm_{name.split()[0].lower()}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def plot_comparison(results):
    names = list(results.keys())
    recalls = [results[n]["recall"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    aucs = [results[n]["roc_auc"] for n in names]
    x = np.arange(len(names)); w = 0.25
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w, recalls, w, label="Recall")
    plt.bar(x, f1s, w, label="F1")
    plt.bar(x + w, aucs, w, label="ROC-AUC")
    plt.xticks(x, names, rotation=10)
    plt.ylim(0, 1.05); plt.ylabel("Score")
    plt.title("Model Comparison"); plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def main():
    ensure_dirs()
    df, features = load_and_prepare()
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    results = {}
    best_name, best_model, best_recall = None, None, -1
    for name, model in build_models().items():
        print(f"\n=== Training: {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"recall": recall, "f1": f1, "roc_auc": auc,
                         "report": classification_report(y_test, y_pred,
                                                          output_dict=True)}
        print(classification_report(y_test, y_pred, digits=3))
        print(f"Recall={recall:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
        plot_confusion(name, y_test, y_pred)
        if recall > best_recall:
            best_name, best_model, best_recall = name, model, recall
    plot_comparison(results)
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump({"model": best_model, "features": features, "name": best_name},
                os.path.join(MODEL_DIR, "best_model.joblib"))
    print(f"\nBest model by Recall: {best_name} (Recall={best_recall:.3f})")


if __name__ == "__main__":
    main()
