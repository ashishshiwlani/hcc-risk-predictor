"""
Tests for feature engineering pipeline.

All tests run without GPU or external datasets.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_hcc_dataset
from src.feature_engineering import (
    extract_features,
    prepare_single_patient,
    FEATURE_NAMES,
    N_FEATURES,
    CONTINUOUS_COLS,
    BINARY_COLS,
    LOG_TRANSFORM_COLS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def small_df():
    """100-patient synthetic DataFrame for fast tests."""
    return generate_hcc_dataset(n_patients=100, random_state=0)


@pytest.fixture(scope="module")
def feature_matrix(small_df):
    """Pre-computed feature matrix + scaler."""
    X, names, scaler = extract_features(small_df)
    return X, names, scaler


# ============================================================================
# Feature name and shape tests
# ============================================================================

class TestFeatureNames:

    def test_feature_names_constant(self):
        """FEATURE_NAMES must be stable (model weights depend on ordering)."""
        assert isinstance(FEATURE_NAMES, list)
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_n_features_correct(self):
        assert N_FEATURES == len(CONTINUOUS_COLS) + len(BINARY_COLS)

    def test_continuous_cols_subset(self):
        for col in CONTINUOUS_COLS:
            assert col in FEATURE_NAMES

    def test_binary_cols_subset(self):
        for col in BINARY_COLS:
            assert col in FEATURE_NAMES

    def test_log_transform_cols_are_continuous(self):
        for col in LOG_TRANSFORM_COLS:
            assert col in CONTINUOUS_COLS


# ============================================================================
# extract_features tests
# ============================================================================

class TestExtractFeatures:

    def test_output_shape(self, small_df, feature_matrix):
        X, names, scaler = feature_matrix
        assert X.shape == (len(small_df), N_FEATURES)

    def test_output_dtype_float64(self, feature_matrix):
        X, _, _ = feature_matrix
        assert X.dtype == np.float64

    def test_feature_names_returned(self, feature_matrix):
        _, names, _ = feature_matrix
        assert names == FEATURE_NAMES

    def test_scaler_keys(self, feature_matrix):
        _, _, scaler = feature_matrix
        assert "mean" in scaler
        assert "std" in scaler

    def test_no_nan_in_output(self, feature_matrix):
        X, _, _ = feature_matrix
        assert not np.any(np.isnan(X)), "Feature matrix must not contain NaN"

    def test_no_inf_in_output(self, feature_matrix):
        X, _, _ = feature_matrix
        assert not np.any(np.isinf(X)), "Feature matrix must not contain Inf"

    def test_binary_cols_still_binary_after_scaling(self, small_df, feature_matrix):
        """Binary features should remain 0/1 (not z-scored)."""
        X, names, _ = feature_matrix
        for col in BINARY_COLS:
            idx = names.index(col)
            vals = X[:, idx]
            unique = set(np.unique(vals))
            assert unique.issubset({0.0, 1.0}), (
                f"Binary column '{col}' has unexpected values: {unique}"
            )

    def test_continuous_cols_are_scaled(self, small_df, feature_matrix):
        """Scaled continuous features should have mean ≈ 0, std ≈ 1."""
        X, names, _ = feature_matrix
        for col in CONTINUOUS_COLS[:5]:   # test first 5 to keep runtime short
            idx = names.index(col)
            vals = X[:, idx]
            assert abs(vals.mean()) < 0.5,  f"{col} mean not near 0 after scaling"
            assert 0.5 < vals.std() < 2.0,  f"{col} std not near 1 after scaling"

    def test_inference_uses_train_scaler(self, small_df, feature_matrix):
        """When fit_scaler is provided, the same scaler is returned unchanged."""
        X_train, names, scaler_train = feature_matrix
        # Use a single row as "test" set
        single_df = small_df.iloc[:1]
        X_test, _, scaler_test = extract_features(single_df, fit_scaler=scaler_train)

        np.testing.assert_array_equal(scaler_train["mean"], scaler_test["mean"])
        np.testing.assert_array_equal(scaler_train["std"],  scaler_test["std"])

    def test_missing_column_raises_value_error(self):
        """Omitting a required column should raise ValueError."""
        bad_df = pd.DataFrame({"age": [50], "alt": [30]})   # incomplete
        with pytest.raises(ValueError, match="missing required columns"):
            extract_features(bad_df)

    def test_consistent_output_shape_single_row(self, small_df, feature_matrix):
        """Single-row extraction must produce (1, N_FEATURES)."""
        _, _, scaler = feature_matrix
        X, _, _ = extract_features(small_df.iloc[:1], fit_scaler=scaler)
        assert X.shape == (1, N_FEATURES)


# ============================================================================
# prepare_single_patient tests
# ============================================================================

class TestPrepareSinglePatient:

    def test_output_shape(self, feature_matrix):
        _, _, scaler = feature_matrix
        vec = prepare_single_patient({"age": 55, "afp": 10.0}, scaler)
        assert vec.shape == (1, N_FEATURES)

    def test_output_dtype(self, feature_matrix):
        _, _, scaler = feature_matrix
        vec = prepare_single_patient({}, scaler)
        assert vec.dtype == np.float64

    def test_empty_dict_uses_defaults(self, feature_matrix):
        """Empty patient dict should produce a valid feature vector using defaults."""
        _, _, scaler = feature_matrix
        vec = prepare_single_patient({}, scaler)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_full_dict_same_as_single_row_df(self, small_df, feature_matrix):
        """
        prepare_single_patient and extract_features(single_row_df) should give
        the same result when all columns are provided.
        """
        _, _, scaler = feature_matrix
        row = small_df.iloc[0]
        patient_dict = row.to_dict()

        vec_from_func = prepare_single_patient(patient_dict, scaler)
        vec_from_df, _, _ = extract_features(
            small_df.iloc[:1], fit_scaler=scaler
        )

        np.testing.assert_array_almost_equal(vec_from_func, vec_from_df, decimal=6)


# ============================================================================
# Derived score tests
# ============================================================================

class TestDerivedScores:

    def test_fib4_in_expected_range(self, small_df):
        """FIB-4 should be between 0 and 20."""
        assert small_df["fib4_index"].between(0, 20).all()

    def test_apri_in_expected_range(self, small_df):
        """APRI should be between 0 and 30."""
        assert small_df["apri"].between(0, 30).all()

    def test_fib4_high_for_high_risk(self):
        """High-risk patients should have higher median FIB-4."""
        df = generate_hcc_dataset(n_patients=500, random_state=1)
        low_fib4  = df[df["risk_label"] == 0]["fib4_index"].median()
        high_fib4 = df[df["risk_label"] == 2]["fib4_index"].median()
        assert high_fib4 > low_fib4, (
            f"High-risk median FIB-4 ({high_fib4:.2f}) should exceed "
            f"low-risk ({low_fib4:.2f})"
        )
