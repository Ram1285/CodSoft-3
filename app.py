import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("fraud_detection_pipeline.pkl")
METADATA_PATH = Path("feature_metadata.json")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="",
    layout="wide",
)

st.title("Credit Card Fraud Detection")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files were not found. Run `python train_model.py` before starting the app.")
    st.stop()

model = load_model()
metadata = load_metadata()
feature_columns = metadata["feature_columns"]
feature_defaults = metadata["feature_defaults"]
feature_mins = metadata["feature_mins"]
feature_maxs = metadata["feature_maxs"]

st.subheader("Transaction Features")

input_values = {}
columns = st.columns(3)

for index, feature in enumerate(feature_columns):
    with columns[index % 3]:
        default_value = float(feature_defaults.get(feature, 0.0))
        min_value = float(feature_mins.get(feature, default_value))
        max_value = float(feature_maxs.get(feature, default_value))

        input_values[feature] = st.number_input(
            label=feature,
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            format="%.6f",
        )

if st.button("Check Transaction", type="primary"):
    transaction = pd.DataFrame([input_values], columns=feature_columns)
    prediction = int(model.predict(transaction)[0])

    if prediction == 1:
        st.error("Fraudulent Transaction")
    else:
        st.success("Genuine Transaction")

    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(transaction)[0][1]
        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
