"""
Feature engineering for the HCC risk prediction pipeline.

Transformations applied:
  1. Log-transform skewed lab values (AFP, bilirubin, GGT, APRI)
  2. Compute/verify composite scores (FIB-4, APRI)
  3. Encode binary flags (already 0/1 — kept as-is)
  4. Standard-scale all continuous features
  5. Return aligned numpy array with stable column ordering

Design choice — log transform:
  Lab values like AFP span 4 orders of magnitude (2 ng/mL to 5,000 ng/mL).
  LightGBM splits on raw values; log-transforming compresses the right tail and
  lets the model find a meaningful threshold near AFP ~400 ng/mL with fewer
  splits.  We log1p (log(x+1)) to handle zero-valued inputs safely.

Design choice — FIB-4 and APRI as features:
  These validated clinical scores (published in J Hepatology 2006, Hepatology
  2003) are linear combinations of the raw lab values.  Including them gives
  LightGBM a pre-computed "expert feature" it can use directly, which tends to
  improve AUC by ~1-2% on structured clinical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

# ---------------------------------------------------------------------------
# Stable column ordering — NEVER reorder; model weights depend on positions
# ---------------------------------------------------------------------------

# Continuous columns that receive log1p transformation
LOG_TRANSFORM_COLS: List[str] = ["afp", "ggt", "bilirubin", "apri"]

# All continuous feature columns (in fixed order)
CONTINUOUS_COLS: List[str] = [
    "age", "afp", "alt", "ast", "bilirubin", "albumin",
    "platelets", "ggt", "inr", "creatinine", "fib4_index", "apri",
]

# Binary (0/1) columns
BINARY_COLS: List[str] = [
    "male", "hcv_positive", "hbv_positive",
    "cirrhosis", "diabetes", "alcohol_use",
]

# Final feature names in the exact order returned by extract_features()
FEATURE_NAMES: List[str] = CONTINUOUS_COLS + BINARY_COLS

# Number of features (used in tests)
N_FEATURES: int = len(FEATURE_NAMES)   # 18


def _compute_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Re)compute FIB-4 and APRI from raw columns.

    Called both during training and inference so the scores are always
    internally consistent, even if the input DataFrame already contains them.
    """
    df = df.copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        fib4 = (df["age"] * df["ast"]) / (
            df["platelets"] * np.sqrt(np.maximum(df["alt"], 0.1))
        )
    df["fib4_index"] = np.clip(fib4, 0.0, 20.0).fillna(0.0)

    apri = (df["ast"] / 40.0) / df["platelets"] * 100
    df["apri"] = np.clip(apri, 0.0, 30.0).fillna(0.0)

    return df


def extract_features(
    df: pd.DataFrame,
    fit_scaler: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str], dict]:
    """
    Transform a patient DataFrame into a feature matrix.

    Args:
        df:          DataFrame with columns matching FEATURE_NAMES (plus
                     patient_id, risk_label, risk_category which are ignored).
        fit_scaler:  If None, compute mean/std from df (training mode).
                     If provided dict {"mean": ..., "std": ...}, use those
                     values (inference mode).

    Returns:
        X:          np.ndarray shape (n_samples, N_FEATURES), dtype float64
        feat_names: List[str] — FEATURE_NAMES (stable ordering)
        scaler:     dict {"mean": np.ndarray, "std": np.ndarray}
                    (same dict passed back in; use to scale future rows)

    Raises:
        ValueError: if any required column is missing from df.

    Example:
        >>> X_train, names, scaler = extract_features(train_df)
        >>> X_test, _, _          = extract_features(test_df, fit_scaler=scaler)
    """
    # Validate columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # (Re)compute derived scores for consistency
    df = _compute_derived_scores(df)

    working = df[FEATURE_NAMES].copy().astype(np.float64)

    # Log1p-transform skewed lab values
    for col in LOG_TRANSFORM_COLS:
        working[col] = np.log1p(working[col].clip(lower=0.0))

    # Standard scaling (z-score) on continuous columns only
    # Binary columns are left as 0/1
    cont_idx = [FEATURE_NAMES.index(c) for c in CONTINUOUS_COLS]
    X = working.values  # (n, 18)

    if fit_scaler is None:
        mean = X[:, cont_idx].mean(axis=0)
        std  = X[:, cont_idx].std(axis=0)
        std  = np.where(std == 0, 1.0, std)     # avoid divide-by-zero
        scaler = {"mean": mean, "std": std}
    else:
        mean = fit_scaler["mean"]
        std  = fit_scaler["std"]
        scaler = fit_scaler

    X = X.copy()
    X[:, cont_idx] = (X[:, cont_idx] - mean) / std

    return X.astype(np.float64), FEATURE_NAMES, scaler


def prepare_single_patient(patient: dict, scaler: dict) -> np.ndarray:
    """
    Convert a single patient dict to a scaled feature vector.

    Handles missing binary fields by defaulting to 0 (absent risk factor).
    Handles missing continuous fields by defaulting to population medians
    (reasonable clinical default when a test result is unavailable).

    Args:
        patient: dict with lab values (e.g. from Streamlit form inputs).
        scaler:  Scaler dict from extract_features() call on training data.

    Returns:
        np.ndarray shape (1, N_FEATURES), dtype float64

    Example:
        >>> vec = prepare_single_patient(
        ...     {"age": 58, "afp": 12.3, "alt": 45, ...},
        ...     scaler=scaler
        ... )
    """
    # Population medians used as defaults for missing fields
    _DEFAULTS: dict = {
        "age": 55.0,
        "afp": 5.0,
        "alt": 30.0,
        "ast": 28.0,
        "bilirubin": 0.7,
        "albumin": 4.2,
        "platelets": 200.0,
        "ggt": 35.0,
        "inr": 1.0,
        "creatinine": 0.9,
        "fib4_index": 1.0,
        "apri": 0.5,
        "male": 0,
        "hcv_positive": 0,
        "hbv_positive": 0,
        "cirrhosis": 0,
        "diabetes": 0,
        "alcohol_use": 0,
    }

    row = {k: patient.get(k, v) for k, v in _DEFAULTS.items()}
    df_single = pd.DataFrame([row])
    X, _, _ = extract_features(df_single, fit_scaler=scaler)
    return X
