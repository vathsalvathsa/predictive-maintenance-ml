"""Streamlit dashboard for predictive maintenance."""

import streamlit as st
import pandas as pd
from src.predict import predict_single, predict_batch, load_model

st.set_page_config(page_title="Predictive Maintenance",
                   page_icon="🔧", layout="wide")

st.title("🔧 Predictive Maintenance System")
st.caption("Predict equipment failure 24 hours in advance using sensor data")

try:
    bundle = load_model()
    st.sidebar.success(f"Model: {bundle['name']}")
except Exception as e:
    st.error(f"Model not found. Run `python src/train.py` first.\n\n{e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models compared:** KNN (baseline) · Logistic Regression · Random Forest\n\n"
    "**Selection criterion:** Recall")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Alerts", "Model Info"])

with tab1:
    st.subheader("Enter Current Sensor Readings")
    col1, col2, col3 = st.columns(3)
    temperature = col1.number_input("Temperature (°C)", value=70.0, step=0.1)
    vibration   = col2.number_input("Vibration (g)", value=0.35, step=0.01)
    pressure    = col3.number_input("Pressure (kPa)", value=100.0, step=0.1)
    rpm         = col1.number_input("RPM", value=1500.0, step=1.0)
    current     = col2.number_input("Current (A)", value=10.0, step=0.1)
    if st.button("Predict Risk", type="primary"):
        result = predict_single({"temperature": temperature, "vibration": vibration,
            "pressure": pressure, "rpm": rpm, "current": current})
        c1, c2 = st.columns(2)
        c1.metric("Failure Probability", f"{result['failure_probability']:.2%}")
        c2.metric("Risk Level", result["risk"])
        if result["risk"] == "HIGH":
            st.error("🚨 HIGH RISK — Schedule maintenance immediately.")
        else:
            st.success("✅ LOW RISK — Equipment operating normally.")

with tab2:
    st.subheader("Batch Alert Dashboard")
    if st.button("Load Sample Batch"):
        df = pd.read_csv("data/sensor_data.csv").head(500)
        out = predict_batch(df)
        high = out[out["risk"] == "HIGH"].sort_values(
            "failure_probability", ascending=False)
        st.metric("High-risk records", len(high))
        st.dataframe(high.head(50), use_container_width=True)

with tab3:
    st.subheader("Model Comparison Results")
    try:
        import json
        with open("models/results.json") as f:
            results = json.load(f)
        rows = [{"Model": n, "Recall": round(r["recall"], 3),
                 "F1": round(r["f1"], 3), "ROC-AUC": round(r["roc_auc"], 3)}
                for n, r in results.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.image("docs/figures/model_comparison.png",
                 caption="Model comparison — Recall / F1 / ROC-AUC")
        for m in ["knn", "logistic", "random"]:
            try: st.image(f"docs/figures/cm_{m}.png", width=350)
            except Exception: pass
    except Exception as e:
        st.info(f"Run `python src/train.py` to generate results.\n\n{e}")
