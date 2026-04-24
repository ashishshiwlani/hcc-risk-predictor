"""
Tests for LightGBM HCC model — training, prediction, SHAP, persistence.

All tests run on CPU with a small synthetic dataset. No GPU, no downloads.
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
from src.model import HCCRiskModel, RISK_LABELS, N_CLASSES
from sklearn.model_selection import train_test_split


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def dataset():
    """Small synthetic dataset split into train/val/test."""
    df = generate_hcc_dataset(n_patients=500, random_state=42)
    y  = df["risk_label"].values

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=42
    )
    df_train, df_val, y_train, y_val = train_test_split(
        df_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    X_train, _, scaler = extract_features(df_train)
    X_val,   _, _      = extract_features(df_val,  fit_scaler=scaler)
    X_test,  _, _      = extract_features(df_test, fit_scaler=scaler)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
    }


@pytest.fixture(scope="module")
def fitted_model(dataset):
    """Trained HCCRiskModel for re-use across tests."""
    model = HCCRiskModel(n_estimators=50, random_state=42)   # fewer rounds for speed
    model.fit(
        dataset["X_train"], dataset["y_train"],
        dataset["X_val"],   dataset["y_val"],
    )
    return model


# ============================================================================
# Model initialisation tests
# ============================================================================

class TestModelInit:

    def test_default_params(self):
        model = HCCRiskModel()
        assert model.n_estimators == 500
        assert model.random_state == 42
        assert not model.is_fitted

    def test_custom_params(self):
        model = HCCRiskModel(n_estimators=200, learning_rate=0.1, num_leaves=63)
        assert model.n_estimators == 200
        assert model.lgb_params["learning_rate"] == 0.1
        assert model.lgb_params["num_leaves"] == 63

    def test_not_fitted_raises_on_predict(self, dataset):
        model = HCCRiskModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(dataset["X_test"][:1])


# ============================================================================
# Training tests
# ============================================================================

class TestModelFit:

    def test_fit_returns_self(self, dataset):
        model = HCCRiskModel(n_estimators=30, random_state=0)
        result = model.fit(dataset["X_train"], dataset["y_train"])
        assert result is model

    def test_is_fitted_after_fit(self, fitted_model):
        assert fitted_model.is_fitted

    def test_booster_not_none_after_fit(self, fitted_model):
        assert fitted_model.booster is not None

    def test_best_iteration_positive(self, fitted_model):
        assert fitted_model.best_iteration > 0

    def test_fit_without_validation(self, dataset):
        """Fitting without validation data should also work."""
        model = HCCRiskModel(n_estimators=20, random_state=0)
        model.fit(dataset["X_train"], dataset["y_train"])
        assert model.is_fitted


# ============================================================================
# Prediction tests
# ============================================================================

class TestPredictProba:

    def test_output_shape(self, fitted_model, dataset):
        proba = fitted_model.predict_proba(dataset["X_test"])
        assert proba.shape == (len(dataset["X_test"]), N_CLASSES)

    def test_probabilities_sum_to_one(self, fitted_model, dataset):
        proba = fitted_model.predict_proba(dataset["X_test"])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_probabilities_in_0_1(self, fitted_model, dataset):
        proba = fitted_model.predict_proba(dataset["X_test"])
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_single_sample_shape(self, fitted_model, dataset):
        proba = fitted_model.predict_proba(dataset["X_test"][:1])
        assert proba.shape == (1, N_CLASSES)

    def test_predict_returns_valid_classes(self, fitted_model, dataset):
        preds = fitted_model.predict(dataset["X_test"])
        assert set(preds).issubset({0, 1, 2})

    def test_predict_argmax_matches_proba(self, fitted_model, dataset):
        proba = fitted_model.predict_proba(dataset["X_test"])
        preds = fitted_model.predict(dataset["X_test"])
        np.testing.assert_array_equal(preds, np.argmax(proba, axis=1))


# ============================================================================
# SHAP explanation tests
# ============================================================================

class TestExplain:

    def test_shap_output_shape(self, fitted_model, dataset):
        shap_vals = fitted_model.explain(dataset["X_test"])
        assert shap_vals.shape == (len(dataset["X_test"]), N_FEATURES)

    def test_shap_single_sample(self, fitted_model, dataset):
        shap_vals = fitted_model.explain(dataset["X_test"][:1])
        assert shap_vals.shape == (1, N_FEATURES)

    def test_shap_finite_values(self, fitted_model, dataset):
        shap_vals = fitted_model.explain(dataset["X_test"][:10])
        assert np.all(np.isfinite(shap_vals)), "SHAP values must be finite"

    def test_shap_non_all_zero(self, fitted_model, dataset):
        """SHAP values should not all be zero — model must have learned something."""
        shap_vals = fitted_model.explain(dataset["X_test"][:10])
        assert np.any(shap_vals != 0.0)


# ============================================================================
# Evaluation tests
# ============================================================================

class TestEvaluate:

    def test_evaluate_returns_dict(self, fitted_model, dataset):
        metrics = fitted_model.evaluate(dataset["X_test"], dataset["y_test"])
        assert isinstance(metrics, dict)

    def test_evaluate_has_required_keys(self, fitted_model, dataset):
        metrics = fitted_model.evaluate(dataset["X_test"], dataset["y_test"])
        for key in ["auc_ovr", "log_loss_val", "report", "confusion_mat"]:
            assert key in metrics

    def test_auc_in_valid_range(self, fitted_model, dataset):
        metrics = fitted_model.evaluate(dataset["X_test"], dataset["y_test"])
        assert 0.5 <= metrics["auc_ovr"] <= 1.0, (
            f"AUC {metrics['auc_ovr']} is below 0.5 — model is worse than random"
        )

    def test_auc_above_baseline(self, fitted_model, dataset):
        """Expect at least 0.75 AUC on synthetic data — sanity check."""
        metrics = fitted_model.evaluate(dataset["X_test"], dataset["y_test"])
        assert metrics["auc_ovr"] >= 0.75, (
            f"AUC {metrics['auc_ovr']} is too low — model may not be learning"
        )

    def test_confusion_matrix_shape(self, fitted_model, dataset):
        metrics = fitted_model.evaluate(dataset["X_test"], dataset["y_test"])
        assert metrics["confusion_mat"].shape == (N_CLASSES, N_CLASSES)


# ============================================================================
# Save / load tests
# ============================================================================

class TestPersistence:

    def test_save_creates_files(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_model.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "hcc_booster.lgb"))
            assert os.path.exists(os.path.join(tmpdir, "hcc_metadata.pkl"))

    def test_save_unfitted_raises(self):
        model = HCCRiskModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="not fitted"):
                model.save(tmpdir)

    def test_load_roundtrip(self, fitted_model, dataset):
        """Save then load; predictions should be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_model.save(tmpdir)
            loaded = HCCRiskModel.load(tmpdir)

            proba_orig   = fitted_model.predict_proba(dataset["X_test"])
            proba_loaded = loaded.predict_proba(dataset["X_test"])
            np.testing.assert_array_almost_equal(proba_orig, proba_loaded, decimal=5)

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                HCCRiskModel.load(tmpdir)

    def test_loaded_model_is_fitted(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_model.save(tmpdir)
            loaded = HCCRiskModel.load(tmpdir)
            assert loaded.is_fitted

    def test_loaded_model_feature_names(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_model.save(tmpdir)
            loaded = HCCRiskModel.load(tmpdir)
            assert loaded.feature_names == fitted_model.feature_names
