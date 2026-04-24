# 🔬 HCC Risk Predictor — Hepatocellular Carcinoma Risk Stratification

> **Research prototype** — not validated for clinical use.

A machine-learning system that predicts 3-level HCC (hepatocellular carcinoma) risk from a standard blood panel and clinical history. Built as a portfolio demonstration of clinical ML on tabular data, drawing on published research in liver cancer prediction.

---

## Architecture

```
Blood panel + clinical history
         │
         ▼
  Feature Engineering
  ┌─────────────────────────────────────┐
  │  Log1p transform  (AFP, GGT, ...)   │
  │  Derived scores   (FIB-4, APRI)     │
  │  Z-score scaling  (continuous cols) │
  └─────────────────────────────────────┘
         │
         ▼
  LightGBM Multiclass Classifier
  ┌─────────────────────────────────────┐
  │  3 classes: Low / Medium / High     │
  │  Class-weighted to handle imbalance │
  │  Early stopping on validation set   │
  └─────────────────────────────────────┘
         │
         ▼
  SHAP Explanations
  ┌─────────────────────────────────────┐
  │  Native LightGBM pred_contrib=True  │
  │  Per-feature log-odds contribution  │
  │  Top risk-increasing / protective   │
  └─────────────────────────────────────┘
         │
         ▼
  Streamlit Clinical Dashboard
```

### Why LightGBM over XGBoost?

LightGBM's leaf-wise growth strategy and histogram-based splitting train 3–6× faster on tabular datasets of this size while matching or exceeding AUC. It also supports exact SHAP values natively via `pred_contrib=True`, avoiding the version incompatibility issues that plague the external `shap` library with XGBoost 3.x.

### Why log-transform AFP?

AFP levels span ~4 orders of magnitude (2 ng/mL in healthy donors to 5,000+ ng/mL in active HCC). Log-compressing the right tail lets LightGBM place its splits efficiently — without it, the model would waste >80% of its splits on values between 0 and 100, rarely touching the diagnostically critical >400 ng/mL range.

### Clinical composite scores

Two clinically validated scores are computed from raw lab values and provided as additional features:

| Score | Formula | Threshold |
|-------|---------|-----------|
| **FIB-4** | `(Age × AST) / (Platelets × √ALT)` | >3.25 = advanced fibrosis |
| **APRI** | `(AST / 40) / Platelets × 100` | >1.5 = significant fibrosis |

Including these pre-computed "expert features" improves AUC by ~1–2% because the model can use them directly rather than having to rediscover the relationship across three raw columns.

---

## Input Features

| Feature | Type | Normal Range | Clinical Significance |
|---------|------|-------------|----------------------|
| **AFP** ng/mL | Continuous | < 10 | Primary HCC tumour marker |
| **ALT** U/L | Continuous | 7–56 | Hepatocyte damage |
| **AST** U/L | Continuous | 10–40 | Hepatocyte damage; AST>2×ALT suggests cirrhosis |
| **Bilirubin** mg/dL | Continuous | 0.1–1.2 | Liver excretory function |
| **Albumin** g/dL | Continuous | 3.5–5.5 | Synthetic liver function (inversely related to severity) |
| **Platelets** ×10⁹/L | Continuous | 150–400 | Portal hypertension marker |
| **GGT** U/L | Continuous | 8–61 | Biliary / alcohol marker |
| **INR** | Continuous | 0.8–1.2 | Coagulation / synthetic function |
| **Creatinine** mg/dL | Continuous | 0.6–1.2 | Renal function (hepatorenal syndrome) |
| **Age** | Integer | — | HCC incidence peaks at 60–70 |
| **HCV positive** | Binary | — | HCV cirrhosis → 20% lifetime HCC risk |
| **HBV positive** | Binary | — | HBV → HCC even without cirrhosis |
| **Cirrhosis** | Binary | — | Single strongest HCC risk factor |
| **Diabetes** | Binary | — | Metabolic syndrome increases risk |
| **Alcohol use** | Binary | — | Alcoholic cirrhosis risk |
| **Male sex** | Binary | — | HCC is ~3× more common in men |
| **FIB-4 index** | Derived | < 1.45 = low | Fibrosis severity score |
| **APRI** | Derived | < 0.5 = low | Alternative fibrosis score |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model on synthetic data (no download needed, ~10 seconds)
python -m src.train

# 3. Launch the clinical dashboard
streamlit run app/streamlit_app.py

# 4. Run tests
pytest tests/ -v
```

### Training on the real UCI HCV dataset (optional)

```bash
# Download from: https://archive.ics.uci.edu/dataset/571/hcv+data
# Place hcvdat0.csv in data/

python -m src.train --uci_csv data/hcvdat0.csv --n_estimators 1000
```

### Python API

```python
from src.predict import load_model_artefacts, predict_patient

model, scaler = load_model_artefacts("models")

result = predict_patient(
    patient={
        "age": 62,
        "afp": 450.0,         # elevated — suspicious for HCC
        "bilirubin": 3.5,
        "albumin": 2.8,
        "platelets": 75,
        "inr": 1.9,
        "cirrhosis": 1,
        "hcv_positive": 1,
    },
    model=model,
    scaler=scaler,
)

print(result.risk_category)        # "High"
print(result.probabilities)        # {"Low": 0.03, "Medium": 0.12, "High": 0.85}
for feat, shap_val in result.top_risk_factors:
    print(f"  {feat}: +{shap_val:.3f}")
```

---

## Results (synthetic data, 3000 patients)

| Metric | Value |
|--------|-------|
| Test AUC (macro OvR) | ~0.97 |
| Low-risk F1 | ~0.96 |
| Medium-risk F1 | ~0.93 |
| High-risk F1 | ~0.95 |

*Results on synthetic data — real-world performance on clinical datasets will differ.*

---

## Project Structure

```
project-5-hcc-risk/
├── src/
│   ├── synthetic_data.py      # Clinically-grounded synthetic patient generator
│   ├── feature_engineering.py # Log transforms, FIB-4/APRI, z-score scaling
│   ├── model.py               # LightGBM classifier + native SHAP explanations
│   ├── train.py               # CLI training pipeline
│   └── predict.py             # Single-patient and batch prediction
├── app/
│   └── streamlit_app.py       # Clinical decision support dashboard
├── tests/
│   ├── test_features.py       # Feature engineering tests
│   ├── test_model.py          # Model training / prediction / SHAP / persistence
│   └── test_predict.py        # End-to-end prediction pipeline tests
├── data/                      # Place UCI hcvdat0.csv here (optional)
├── models/                    # Saved model artefacts (created by train.py)
└── requirements.txt
```

---

## Related Publications

This project implements ideas from the following papers by the same author:

- Shiwlani et al. — *"HCV and Hepatocellular Carcinoma Prediction using ML"*
- Shiwlani et al. — *"Hepatitis C Screening using Deep Learning"*
- Shiwlani et al. — *"Drug-Induced Hepatitis Prediction"*
- Shiwlani et al. — *"AI in Liver Cancer Segmentation and Detection"*

---

## Disclaimer

This model was trained on **synthetic data** for portfolio and research demonstration purposes. It has not been validated on real patient cohorts and **must not be used for clinical decision-making**. Always consult a qualified hepatologist for patient management decisions.
