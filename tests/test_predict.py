"""
Tests for single-patient prediction and batch prediction.

No GPU, no external files needed.
"""

import pytest
import numpy as np
import os
import sys
import pickle
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_hcc_dataset
from src.feature_engineering import extract_features, N_FEATURES
from src.model import HCCRiskModel, RISK_LABELS
from src.predict import predict_patient, batch_predict, PredictionResult
from sklearn.model_selection import train_test_split


# ============================================================================
# Shared fixture: trained model + scaler
# ============================================================================

@pytest.fixture(scope="module")
def trained_model_and_scaler():
    df = generate_hcc_dataset(n_patients=400, random_state=7)
    y  = df["risk_label"].values

    df_train, df_val, y_train, y_val = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=7
    )
    X_train, _, scaler = extract_features(df_train)
    X_val,   _, _      = extract_features(df_val, fit_scaler=scaler)

    model = HCCRiskModel(n_estimators=50, random_state=7)
    model.fit(X_train, y_train, X_val, y_val)

    return model, scaler


@pytest.fixture
def low_risk_patient():
    return {
        "age": 35, "afp": 3.0, "alt": 22, "ast": 20,
        "bilirubin": 0.5, "albumin": 4.5, "platelets": 280,
        "ggt": 25, "inr": 0.95, "creatinine": 0.8,
        "male": 0, "hcv_positive": 0, "hbv_positive": 0,
        "cirrhosis": 0, "diabetes": 0, "alcohol_use": 0,
    }


@pytest.fixture
def high_risk_patient():
    return {
        "age": 68, "afp": 600.0, "alt": 80, "ast": 120,
        "bilirubin": 4.5, "albumin": 2.5, "platelets": 60,
        "ggt": 250, "inr": 2.1, "creatinine": 1.8,
        "male": 1, "hcv_positive": 1, "hbv_positive": 0,
        "cirrhosis": 1, "diabetes": 1, "alcohol_use": 1,
    }


# ============================================================================
# PredictionResult dataclass tests
# ============================================================================

class TestPredictionResult:

    def test_result_has_required_fields(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)

        assert hasattr(result, "risk_label")
        assert hasattr(result, "risk_category")
        assert hasattr(result, "probabilities")
        assert hasattr(result, "explanation")
        assert hasattr(result, "top_risk_factors")
        assert hasattr(result, "top_protective")

    def test_risk_label_is_valid_class(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        assert result.risk_label in {0, 1, 2}

    def test_risk_category_is_valid_string(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        assert result.risk_category in RISK_LABELS

    def test_risk_label_and_category_consistent(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        assert RISK_LABELS[result.risk_label] == result.risk_category

    def test_probabilities_have_all_classes(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        for label in RISK_LABELS:
            assert label in result.probabilities

    def test_probabilities_sum_to_one(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_probabilities_in_range(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        for p in result.probabilities.values():
            assert 0.0 <= p <= 1.0


# ============================================================================
# Explanation tests
# ============================================================================

class TestExplanation:

    def test_explanation_is_list_of_tuples(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        assert isinstance(result.explanation, list)
        for item in result.explanation:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_explanation_feature_names_valid(self, trained_model_and_scaler, low_risk_patient):
        from src.feature_engineering import FEATURE_NAMES
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        for feat_name, _ in result.explanation:
            assert feat_name in FEATURE_NAMES

    def test_explanation_shap_values_finite(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        for _, shap_val in result.explanation:
            assert np.isfinite(shap_val)

    def test_top_risk_factors_positive_shap(self, trained_model_and_scaler, high_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(high_risk_patient, model, scaler)
        for _, val in result.top_risk_factors:
            assert val > 0

    def test_top_protective_positive_magnitude(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        result = predict_patient(low_risk_patient, model, scaler)
        for _, val in result.top_protective:
            assert val >= 0   # stored as absolute value


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:

    def test_empty_patient_dict(self, trained_model_and_scaler):
        """Empty patient dict should use defaults and return valid result."""
        model, scaler = trained_model_and_scaler
        result = predict_patient({}, model, scaler)
        assert result.risk_label in {0, 1, 2}
        assert isinstance(result.risk_category, str)

    def test_partial_patient_dict(self, trained_model_and_scaler):
        """Partial dict should fill missing values with defaults."""
        model, scaler = trained_model_and_scaler
        result = predict_patient({"afp": 500.0, "cirrhosis": 1}, model, scaler)
        assert result.risk_label in {0, 1, 2}

    def test_high_afp_tends_toward_high_risk(self, trained_model_and_scaler):
        """
        High AFP (>400 ng/mL) with cirrhosis should give higher High-risk
        probability than a healthy patient.

        This is not guaranteed for every model seed, but with 500+ trees
        on well-separated synthetic data it should hold reliably.
        """
        model, scaler = trained_model_and_scaler

        healthy = predict_patient(
            {"afp": 3.0, "cirrhosis": 0, "albumin": 4.5},
            model, scaler,
        )
        sick = predict_patient(
            {"afp": 600.0, "cirrhosis": 1, "albumin": 2.2,
             "bilirubin": 4.5, "platelets": 55, "inr": 2.2},
            model, scaler,
        )

        assert sick.probabilities["High"] > healthy.probabilities["High"], (
            "High AFP + cirrhosis should yield higher High-risk probability"
        )


# ============================================================================
# Batch prediction tests
# ============================================================================

class TestBatchPredict:

    def test_batch_returns_list(self, trained_model_and_scaler, low_risk_patient, high_risk_patient):
        model, scaler = trained_model_and_scaler
        results = batch_predict([low_risk_patient, high_risk_patient], model, scaler)
        assert isinstance(results, list)

    def test_batch_length_matches_input(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        patients = [low_risk_patient] * 10
        results = batch_predict(patients, model, scaler)
        assert len(results) == 10

    def test_batch_all_valid_results(self, trained_model_and_scaler, low_risk_patient):
        model, scaler = trained_model_and_scaler
        results = batch_predict([low_risk_patient] * 5, model, scaler)
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.risk_label in {0, 1, 2}

    def test_batch_empty_list(self, trained_model_and_scaler):
        model, scaler = trained_model_and_scaler
        results = batch_predict([], model, scaler)
        assert results == []
