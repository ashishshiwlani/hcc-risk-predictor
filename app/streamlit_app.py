"""
HCC Risk Prediction — Streamlit Clinical Dashboard.

Features:
  - Lab value input sliders with clinical reference ranges shown inline
  - Real-time risk gauge (Low / Medium / High) with probability bar chart
  - SHAP waterfall bar chart: top features driving the prediction
  - Clinical context panel: what each biomarker means at current level
  - Batch CSV upload for scoring multiple patients at once

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import load_model_artefacts, predict_patient, batch_predict
from src.synthetic_data import generate_hcc_dataset
from src.feature_engineering import extract_features
from src.model import HCCRiskModel, RISK_LABELS
from src.train import train


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HCC Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_model(model_dir: str = "models"):
    """Load or train the model on first run."""
    booster_path = os.path.join(model_dir, "hcc_booster.lgb")
    if not os.path.exists(booster_path):
        st.info("No saved model found — training on synthetic data (takes ~10s)...")
        train(n_patients=3000, output_dir=model_dir)
    return load_model_artefacts(model_dir)


model, scaler = get_model()


# ---------------------------------------------------------------------------
# Helper: risk colour
# ---------------------------------------------------------------------------

RISK_COLOURS = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
}

RISK_BG = {
    "Low":    "#d5f5e3",
    "Medium": "#fef9e7",
    "High":   "#fde8e6",
}


def risk_badge(category: str) -> str:
    colour = RISK_COLOURS.get(category, "#888")
    bg     = RISK_BG.get(category, "#eee")
    return (
        f'<span style="background:{bg};color:{colour};'
        f'font-weight:bold;padding:4px 14px;border-radius:20px;'
        f'border:2px solid {colour};font-size:1.1em">'
        f'{category} Risk</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar — lab value inputs
# ---------------------------------------------------------------------------

st.sidebar.title("🔬 Patient Lab Values")
st.sidebar.caption("Adjust sliders to match the patient's results. "
                   "Reference ranges shown in parentheses.")

st.sidebar.markdown("### 📊 Liver Function")
afp        = st.sidebar.slider("AFP (ng/mL)  [normal < 10]",        0.0, 1000.0, 5.0,  step=0.5)
alt        = st.sidebar.slider("ALT U/L  [normal 7–56]",             0,   600,    30,   step=1)
ast        = st.sidebar.slider("AST U/L  [normal 10–40]",            0,   600,    28,   step=1)
bilirubin  = st.sidebar.slider("Bilirubin mg/dL  [normal 0.1–1.2]",  0.1, 20.0,  0.7,  step=0.1)
albumin    = st.sidebar.slider("Albumin g/dL  [normal 3.5–5.5]",     1.0, 5.5,   4.2,  step=0.1)
ggt        = st.sidebar.slider("GGT U/L  [normal 8–61]",             0,   800,    35,   step=1)
inr        = st.sidebar.slider("INR  [normal 0.8–1.2]",              0.8, 5.0,   1.0,  step=0.1)

st.sidebar.markdown("### 🩸 Haematology & Renal")
platelets  = st.sidebar.slider("Platelets ×10⁹/L  [normal 150–400]", 10,  500,   200,  step=5)
creatinine = st.sidebar.slider("Creatinine mg/dL  [normal 0.6–1.2]", 0.3, 8.0,   0.9,  step=0.1)

st.sidebar.markdown("### 👤 Demographics & History")
age            = st.sidebar.slider("Age (years)",  18, 90, 55)
male           = st.sidebar.checkbox("Male sex", value=False)
hcv_positive   = st.sidebar.checkbox("HCV antibody positive", value=False)
hbv_positive   = st.sidebar.checkbox("HBV surface antigen positive", value=False)
cirrhosis      = st.sidebar.checkbox("Liver cirrhosis (diagnosed)", value=False)
diabetes       = st.sidebar.checkbox("Diabetes mellitus", value=False)
alcohol_use    = st.sidebar.checkbox("Alcohol use (>14 units/week)", value=False)


# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------

patient = {
    "age":          age,
    "afp":          afp,
    "alt":          alt,
    "ast":          ast,
    "bilirubin":    bilirubin,
    "albumin":      albumin,
    "platelets":    platelets,
    "ggt":          ggt,
    "inr":          inr,
    "creatinine":   creatinine,
    "male":         int(male),
    "hcv_positive": int(hcv_positive),
    "hbv_positive": int(hbv_positive),
    "cirrhosis":    int(cirrhosis),
    "diabetes":     int(diabetes),
    "alcohol_use":  int(alcohol_use),
}

result = predict_patient(patient, model, scaler)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("🔬 HCC Risk Predictor")
st.caption(
    "Hepatocellular carcinoma (HCC) risk stratification from blood panel and "
    "clinical history. **For research use only — not a clinical diagnostic tool.**"
)

# ── Risk badge ──────────────────────────────────────────────────────────────
col_badge, col_prob = st.columns([1, 2])

with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(risk_badge(result.risk_category), unsafe_allow_html=True)
    st.markdown(
        f"<br><span style='font-size:0.9em;color:#666'>"
        f"Confidence: {result.probabilities[result.risk_category]*100:.1f}%"
        f"</span>",
        unsafe_allow_html=True,
    )

with col_prob:
    fig_prob = go.Figure(go.Bar(
        x=list(result.probabilities.values()),
        y=list(result.probabilities.keys()),
        orientation="h",
        marker_color=[RISK_COLOURS[k] for k in result.probabilities],
        text=[f"{v*100:.1f}%" for v in result.probabilities.values()],
        textposition="outside",
    ))
    fig_prob.update_layout(
        title="Predicted Probabilities",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=200,
        margin=dict(l=20, r=40, t=40, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

st.divider()

# ── SHAP explanation ─────────────────────────────────────────────────────────
st.subheader("🧠 Why this prediction?")
st.caption(
    f"SHAP values show which lab results drove the **{result.risk_category} risk** "
    f"classification. Positive = pushes toward this class."
)

if result.explanation:
    feat_names  = [e[0] for e in result.explanation]
    shap_values = [e[1] for e in result.explanation]
    colours     = [RISK_COLOURS["High"] if v > 0 else RISK_COLOURS["Low"]
                   for v in shap_values]

    fig_shap = go.Figure(go.Bar(
        x=shap_values,
        y=feat_names,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.3f}" for v in shap_values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title=f"Top feature contributions (predicted class: {result.risk_category})",
        xaxis_title="SHAP value (log-odds contribution)",
        height=350,
        margin=dict(l=20, r=60, t=50, b=20),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# ── Computed scores ───────────────────────────────────────────────────────────
st.subheader("📐 Derived Clinical Scores")
col_fib4, col_apri, col_ast_alt = st.columns(3)

fib4_val = (age * ast) / (platelets * np.sqrt(max(alt, 0.1))) if platelets > 0 else 0
apri_val = (ast / 40.0) / platelets * 100 if platelets > 0 else 0
ast_alt_ratio = ast / max(alt, 0.1)

with col_fib4:
    fib4_colour = "#e74c3c" if fib4_val > 3.25 else ("#f39c12" if fib4_val > 1.45 else "#2ecc71")
    st.metric("FIB-4 Index", f"{fib4_val:.2f}")
    st.caption("< 1.45 = low fibrosis | > 3.25 = advanced fibrosis")
    if fib4_val > 3.25:
        st.warning("⚠️ High — suggests advanced hepatic fibrosis")
    elif fib4_val > 1.45:
        st.info("ℹ️ Intermediate — consider liver biopsy")
    else:
        st.success("✓ Low — fibrosis unlikely")

with col_apri:
    st.metric("APRI Score", f"{apri_val:.2f}")
    st.caption("< 0.5 = low fibrosis | > 1.5 = significant fibrosis")
    if apri_val > 1.5:
        st.warning("⚠️ High — significant fibrosis likely")
    elif apri_val > 0.5:
        st.info("ℹ️ Intermediate")
    else:
        st.success("✓ Low")

with col_ast_alt:
    st.metric("AST:ALT Ratio", f"{ast_alt_ratio:.2f}")
    st.caption("> 2.0 suggests alcoholic hepatitis or cirrhosis")
    if ast_alt_ratio > 2.0:
        st.warning("⚠️ Elevated — consider alcohol use or cirrhosis")
    else:
        st.success("✓ Normal range")

st.divider()

# ── Risk factor summary ───────────────────────────────────────────────────────
col_risk, col_protect = st.columns(2)

with col_risk:
    st.subheader("⬆️ Top Risk-Increasing Factors")
    if result.top_risk_factors:
        for feat, val in result.top_risk_factors:
            st.markdown(f"- **{feat.upper()}**: contribution `+{val:.3f}`")
    else:
        st.write("No strong risk-increasing factors identified.")

with col_protect:
    st.subheader("⬇️ Top Protective Factors")
    if result.top_protective:
        for feat, val in result.top_protective:
            st.markdown(f"- **{feat.upper()}**: contribution `−{val:.3f}`")
    else:
        st.write("No strong protective factors identified.")

st.divider()

# ── Batch upload ──────────────────────────────────────────────────────────────
st.subheader("📂 Batch Scoring (CSV Upload)")
st.caption(
    "Upload a CSV with columns matching the feature names. "
    "Download the template to see the expected format."
)

# Template download
template_data = {col: [0.0] for col in [
    "age", "afp", "alt", "ast", "bilirubin", "albumin", "platelets",
    "ggt", "inr", "creatinine", "male", "hcv_positive", "hbv_positive",
    "cirrhosis", "diabetes", "alcohol_use",
]}
template_df = pd.DataFrame(template_data)
st.download_button(
    "⬇️ Download CSV template",
    data=template_df.to_csv(index=False),
    file_name="hcc_batch_template.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Upload patient CSV", type="csv")
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    st.write(f"Loaded {len(batch_df)} patients.")

    patients_list = batch_df.to_dict(orient="records")
    results = batch_predict(patients_list, model, scaler)

    out_df = batch_df.copy()
    out_df["risk_label"]    = [r.risk_label for r in results]
    out_df["risk_category"] = [r.risk_category for r in results]
    out_df["prob_low"]      = [r.probabilities["Low"]    for r in results]
    out_df["prob_medium"]   = [r.probabilities["Medium"] for r in results]
    out_df["prob_high"]     = [r.probabilities["High"]   for r in results]

    st.dataframe(out_df[["risk_category", "prob_low", "prob_medium", "prob_high"]
                         + list(batch_df.columns)].head(50))

    st.download_button(
        "⬇️ Download scored CSV",
        data=out_df.to_csv(index=False),
        file_name="hcc_predictions.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: LightGBM multiclass classifier | "
    "Explainability: Native LightGBM SHAP (pred_contrib=True) | "
    "Dataset: Synthetic (clinically-grounded ranges) | "
    "**Research prototype — not validated for clinical use**"
)
