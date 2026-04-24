"""
Synthetic HCC risk dataset generator.

Generates realistic liver panel data for three patient risk groups:
  - Low:    Healthy donors / mild hepatitis (no cirrhosis, AFP normal)
  - Medium: Chronic HCV/HBV with fibrosis / compensated cirrhosis
  - High:   Decompensated cirrhosis / confirmed HCC / elevated AFP

Biomarker reference ranges are grounded in clinical literature:
  - EASL 2018 HCC Clinical Practice Guidelines
  - AASLD 2023 HCC Guidance
  - Llovet et al. (2021) "Hepatocellular carcinoma", Nature Reviews Disease Primers

All values are sampled from clipped normal distributions that approximate
the skewed distributions seen in real EHR lab data.

Usage:
    >>> from src.synthetic_data import generate_hcc_dataset
    >>> df = generate_hcc_dataset(n_patients=2000, random_state=42)
    >>> df.to_csv("data/hcc_synthetic.csv", index=False)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Per-feature sampling parameters (mean, std, low_clip, high_clip)
# These are stratified by risk group (low / medium / high).
# ---------------------------------------------------------------------------

_FEATURE_PARAMS: dict = {
    # Alpha-fetoprotein (AFP) ng/mL — the primary HCC tumour marker
    # Normal <10; suspicious 10-400; HCC often >400
    "afp": {
        "low":    (4.5,   3.0,   1.0,   15.0),
        "medium": (35.0,  60.0,  5.0,   400.0),
        "high":   (450.0, 600.0, 20.0,  5000.0),
    },
    # ALT (alanine aminotransferase) U/L — hepatocyte damage
    # Normal 7-56; elevated in hepatitis/cirrhosis
    "alt": {
        "low":    (28.0,  12.0,  7.0,   80.0),
        "medium": (75.0,  45.0,  15.0,  400.0),
        "high":   (60.0,  40.0,  10.0,  350.0),   # may normalise in advanced disease
    },
    # AST (aspartate aminotransferase) U/L
    # Normal 10-40; AST>ALT ratio >2 suggests cirrhosis
    "ast": {
        "low":    (25.0,  10.0,  10.0,  60.0),
        "medium": (80.0,  50.0,  20.0,  450.0),
        "high":   (95.0,  70.0,  20.0,  500.0),
    },
    # Total bilirubin mg/dL — liver excretory function
    # Normal 0.1-1.2; elevated in cirrhosis / acute liver failure
    "bilirubin": {
        "low":    (0.6,   0.3,   0.1,   1.5),
        "medium": (1.8,   1.2,   0.4,   8.0),
        "high":   (4.0,   3.0,   0.5,   20.0),
    },
    # Albumin g/dL — synthetic liver function (inversely related to severity)
    # Normal 3.5-5.5; <3.5 indicates hepatic decompensation
    "albumin": {
        "low":    (4.4,   0.4,   3.5,   5.5),
        "medium": (3.5,   0.6,   2.0,   4.5),
        "high":   (2.8,   0.7,   1.5,   4.0),
    },
    # Platelet count 10^9/L — portal hypertension / hypersplenism marker
    # Normal 150-400; thrombocytopenia <150 common in cirrhosis
    "platelets": {
        "low":    (240.0, 50.0,  130.0, 400.0),
        "medium": (130.0, 55.0,  40.0,  250.0),
        "high":   (90.0,  50.0,  20.0,  200.0),
    },
    # GGT (gamma-glutamyl transferase) U/L — biliary / alcohol marker
    # Normal 8-61; elevated in hepatitis, cholestasis, alcohol use
    "ggt": {
        "low":    (30.0,  18.0,  8.0,   100.0),
        "medium": (120.0, 90.0,  20.0,  600.0),
        "high":   (180.0, 130.0, 20.0,  800.0),
    },
    # Prothrombin time (PT/INR) — coagulation / liver synthetic function
    # Normal INR 0.8-1.2; elevated in liver disease
    "inr": {
        "low":    (1.0,   0.1,   0.8,   1.3),
        "medium": (1.4,   0.3,   1.0,   2.5),
        "high":   (1.9,   0.5,   1.2,   4.0),
    },
    # Creatinine mg/dL — renal function (hepatorenal syndrome risk)
    # Normal 0.6-1.2; elevated in hepatorenal syndrome
    "creatinine": {
        "low":    (0.85,  0.18,  0.5,   1.3),
        "medium": (0.95,  0.30,  0.5,   2.5),
        "high":   (1.3,   0.7,   0.5,   5.0),
    },
    # Age years
    "age": {
        "low":    (42.0,  12.0,  18.0,  75.0),
        "medium": (55.0,  12.0,  25.0,  80.0),
        "high":   (63.0,  11.0,  30.0,  85.0),
    },
}

# Probability of binary features per risk group
_BINARY_PARAMS: dict = {
    # HCV antibody positive
    "hcv_positive": {"low": 0.03, "medium": 0.55, "high": 0.65},
    # HBV surface antigen positive
    "hbv_positive": {"low": 0.02, "medium": 0.25, "high": 0.30},
    # Cirrhosis (clinical diagnosis)
    "cirrhosis":    {"low": 0.00, "medium": 0.35, "high": 0.85},
    # Male sex (HCC is ~3x more common in males)
    "male":         {"low": 0.48, "medium": 0.60, "high": 0.70},
    # Diabetes mellitus (metabolic risk)
    "diabetes":     {"low": 0.08, "medium": 0.22, "high": 0.35},
    # Alcohol use (>14 units/week self-reported)
    "alcohol_use":  {"low": 0.10, "medium": 0.30, "high": 0.40},
}

_RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
_CLASS_PROPORTIONS = [0.55, 0.30, 0.15]   # realistic class imbalance


def _sample_feature(rng: np.random.Generator, params: tuple, n: int) -> np.ndarray:
    """Sample from a clipped normal distribution."""
    mean, std, lo, hi = params
    vals = rng.normal(mean, std, n)
    return np.clip(vals, lo, hi)


def generate_hcc_dataset(
    n_patients: int = 3000,
    random_state: int = 42,
    class_proportions: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic HCC risk dataset.

    Args:
        n_patients:         Total number of patients to generate.
        random_state:       Random seed for reproducibility.
        class_proportions:  [p_low, p_medium, p_high]; defaults to [0.55, 0.30, 0.15].

    Returns:
        DataFrame with columns:
            patient_id, age, male, hcv_positive, hbv_positive, cirrhosis,
            diabetes, alcohol_use, afp, alt, ast, bilirubin, albumin,
            platelets, ggt, inr, creatinine, fib4_index, apri,
            risk_label (0/1/2), risk_category (Low/Medium/High)

    Example:
        >>> df = generate_hcc_dataset(n_patients=500, random_state=0)
        >>> df["risk_category"].value_counts()
    """
    if class_proportions is None:
        class_proportions = _CLASS_PROPORTIONS

    rng = np.random.default_rng(random_state)
    proportions = np.array(class_proportions)
    proportions = proportions / proportions.sum()

    counts = np.round(proportions * n_patients).astype(int)
    # Ensure total matches n_patients exactly
    counts[-1] = n_patients - counts[:-1].sum()

    rows = []
    patient_id = 1

    for risk_label, n in enumerate(counts):
        group = ["low", "medium", "high"][risk_label]

        # Continuous lab values
        data: dict = {}
        for feat, params_by_group in _FEATURE_PARAMS.items():
            data[feat] = _sample_feature(rng, params_by_group[group], n)

        # Binary / categorical features
        for feat, prob_by_group in _BINARY_PARAMS.items():
            p = prob_by_group[group]
            data[feat] = rng.binomial(1, p, n).astype(int)

        # Derived composite scores (used as engineered features)
        # FIB-4 index: Age × AST / (Platelets × √ALT)
        # Score >3.25 suggests advanced fibrosis
        with np.errstate(divide="ignore", invalid="ignore"):
            fib4 = (data["age"] * data["ast"]) / (
                data["platelets"] * np.sqrt(np.maximum(data["alt"], 0.1))
            )
        data["fib4_index"] = np.clip(fib4, 0.0, 20.0)

        # APRI: AST / (ULN_AST × Platelets) × 100
        # ULN for AST = 40 U/L; score >2 suggests cirrhosis
        apri = (data["ast"] / 40.0) / data["platelets"] * 100
        data["apri"] = np.clip(apri, 0.0, 30.0)

        # Labels
        data["risk_label"] = np.full(n, risk_label, dtype=int)
        data["patient_id"] = np.arange(patient_id, patient_id + n)
        patient_id += n

        rows.append(pd.DataFrame(data))

    df = pd.concat(rows, ignore_index=True)
    df["risk_category"] = df["risk_label"].map(_RISK_LABELS)

    # Shuffle rows so risk classes are interleaved
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Round continuous values to realistic precision
    for col in ["afp", "alt", "ast", "bilirubin", "albumin", "platelets",
                "ggt", "inr", "creatinine", "fib4_index", "apri"]:
        decimals = 1 if col in ("bilirubin", "albumin", "inr", "creatinine",
                                "fib4_index", "apri") else 0
        df[col] = df[col].round(decimals)

    df["age"] = df["age"].round(0).astype(int)

    return df


def load_uci_hcv(csv_path: str) -> pd.DataFrame:
    """
    Load and harmonise the UCI HCV Prediction Dataset.

    The UCI dataset (hcvdat0.csv) has columns:
        Unnamed:0, Category, Age, Sex, ALB, ALP, ALT, AST,
        BIL, CHE, CHOL, CREA, GGT, PROT

    Categories (target):
        '0=Blood Donor', '0s=suspect Blood Donor',
        '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'

    We remap to 3-class HCC risk:
        Blood Donor / Suspect Donor → 0 (Low)
        Hepatitis                   → 1 (Medium)
        Fibrosis / Cirrhosis        → 2 (High)

    Args:
        csv_path: Path to hcvdat0.csv

    Returns:
        DataFrame aligned with generate_hcc_dataset() schema.

    Download:
        https://archive.ics.uci.edu/dataset/571/hcv+data
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Unnamed: 0": "patient_id",
        "Age": "age",
        "ALB": "albumin",
        "ALP": "ggt",       # closest match (ALP not in our schema; use as proxy)
        "ALT": "alt",
        "AST": "ast",
        "BIL": "bilirubin",
        "CHE": "afp",       # cholinesterase used as AFP proxy (both drop in disease)
        "CHOL": "creatinine",
        "CREA": "creatinine",
        "GGT": "ggt",
        "PROT": "albumin",
    })

    # Sex encoding: 'm' → 1, 'f' → 0
    df["male"] = (df["Sex"].str.strip().str.lower() == "m").astype(int)

    # Risk label mapping
    risk_map = {
        "0=Blood Donor": 0,
        "0s=suspect Blood Donor": 0,
        "1=Hepatitis": 1,
        "2=Fibrosis": 2,
        "3=Cirrhosis": 2,
    }
    df["risk_label"] = df["Category"].map(risk_map)
    df = df.dropna(subset=["risk_label"])
    df["risk_label"] = df["risk_label"].astype(int)
    df["risk_category"] = df["risk_label"].map({0: "Low", 1: "Medium", 2: "High"})

    # Fill missing values with median
    numeric_cols = ["albumin", "alt", "ast", "bilirubin", "ggt", "creatinine", "age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Add absent columns with neutral defaults
    for col, default in [
        ("hcv_positive", 1), ("hbv_positive", 0), ("cirrhosis", 0),
        ("diabetes", 0), ("alcohol_use", 0), ("platelets", 200),
        ("inr", 1.0), ("afp", 5.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Derived scores
    with np.errstate(divide="ignore", invalid="ignore"):
        fib4 = (df["age"] * df["ast"]) / (
            df["platelets"] * np.sqrt(np.maximum(df["alt"], 0.1))
        )
    df["fib4_index"] = np.clip(fib4, 0.0, 20.0)
    df["apri"] = np.clip((df["ast"] / 40.0) / df["platelets"] * 100, 0.0, 30.0)

    return df


if __name__ == "__main__":
    df = generate_hcc_dataset(n_patients=3000)
    out_path = "data/hcc_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} patients to {out_path}")
    print(df["risk_category"].value_counts())
    print(df.describe().round(2))
