"""
Single-patient HCC risk prediction with SHAP explanation.

Usage:
    from src.predict import predict_patient, load_model_artefacts

    model, scaler = load_model_artefacts("models")
    result = predict_patient(
        patient={"age": 62, "afp": 85.0, "alt": 72, "ast": 90,
                 "bilirubin": 2.1, "albumin": 3.2, "platelets": 98,
                 "ggt": 145, "inr": 1.6, "creatinine": 1.1,
                 "cirrhosis": 1, "hcv_positive": 1},
        model=model,
        scaler=scaler,
    )
    print(result["risk_category"])   # "High"
    print(result["explanation"])     # list of (feature_name, shap_value) tuples
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from src.feature_engineering import (
    FEATURE_NAMES,
    N_FEATURES,
    prepare_single_patient,
)
from src.model import HCCRiskModel, RISK_LABELS


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Complete prediction result for a single patient.

    Attributes:
        risk_label:     Integer class (0=Low, 1=Medium, 2=High).
        risk_category:  Human-readable label ("Low" / "Medium" / "High").
        probabilities:  Dict {"Low": p_low, "Medium": p_med, "High": p_high}.
        explanation:    Sorted list of (feature_name, shap_value) tuples.
                        Positive = pushes toward predicted class.
                        Ordered by |shap_value| descending.
        top_risk_factors:   Top 3 features increasing risk.
        top_protective:     Top 3 features decreasing risk.
    """
    risk_label: int
    risk_category: str
    probabilities: dict
    explanation: List[Tuple[str, float]]
    top_risk_factors: List[Tuple[str, float]]
    top_protective: List[Tuple[str, float]]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_artefacts(model_dir: str = "models") -> Tuple[HCCRiskModel, dict]:
    """
    Load trained model and scaler from disk.

    Args:
        model_dir: Directory containing hcc_booster.lgb, hcc_metadata.pkl,
                   and scaler.pkl.

    Returns:
        (HCCRiskModel, scaler_dict)

    Raises:
        FileNotFoundError: if any artefact is missing.
    """
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            f"Run 'python -m src.train' first."
        )

    model  = HCCRiskModel.load(model_dir)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_patient(
    patient: dict,
    model: HCCRiskModel,
    scaler: dict,
    top_n: int = 5,
) -> PredictionResult:
    """
    Run HCC risk prediction for a single patient.

    Args:
        patient: Dict of lab values. Missing keys default to population medians.
                 Recognised keys (all optional):
                   age, afp, alt, ast, bilirubin, albumin, platelets, ggt,
                   inr, creatinine, fib4_index, apri,
                   male, hcv_positive, hbv_positive, cirrhosis,
                   diabetes, alcohol_use
        model:   Fitted HCCRiskModel.
        scaler:  Scaler dict from training.
        top_n:   How many top-contributing features to return.

    Returns:
        PredictionResult dataclass.

    Example:
        >>> result = predict_patient(
        ...     {"age": 62, "afp": 450, "cirrhosis": 1},
        ...     model, scaler
        ... )
        >>> print(result.risk_category)
        High
        >>> for feat, val in result.top_risk_factors:
        ...     print(f"  {feat}: +{val:.3f}")
    """
    # Prepare feature vector — shape (1, N_FEATURES)
    X = prepare_single_patient(patient, scaler)

    # Predict probabilities
    proba = model.predict_proba(X)[0]   # (3,)
    risk_label = int(np.argmax(proba))
    risk_category = RISK_LABELS[risk_label]

    probabilities = {
        label: round(float(p), 4)
        for label, p in zip(RISK_LABELS, proba)
    }

    # SHAP explanation for predicted class
    shap_vals = model.explain(X)[0]   # (N_FEATURES,)

    # Sort by absolute magnitude
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    explanation = [
        (FEATURE_NAMES[i], round(float(shap_vals[i]), 4))
        for i in sorted_idx[:top_n]
    ]

    # Split into risk-increasing (positive) and protective (negative)
    top_risk_factors = [
        (name, val) for name, val in explanation if val > 0
    ][:3]

    top_protective = [
        (name, abs(val)) for name, val in explanation if val < 0
    ][:3]

    return PredictionResult(
        risk_label=risk_label,
        risk_category=risk_category,
        probabilities=probabilities,
        explanation=explanation,
        top_risk_factors=top_risk_factors,
        top_protective=top_protective,
    )


def batch_predict(
    patients: list,
    model: HCCRiskModel,
    scaler: dict,
) -> list:
    """
    Run predictions for a list of patient dicts.

    Args:
        patients: List of patient dicts (same format as predict_patient).
        model:    Fitted HCCRiskModel.
        scaler:   Scaler dict.

    Returns:
        List of PredictionResult objects (same order as input).
    """
    return [predict_patient(p, model, scaler) for p in patients]
