"""
Training script for the HCC risk prediction model.

Usage:
    # Train on synthetic data (no download needed)
    python -m src.train

    # Train with custom settings
    python -m src.train --n_patients 5000 --n_estimators 1000 --output_dir models/

    # Train on UCI HCV dataset (download from UCI first)
    python -m src.train --uci_csv path/to/hcvdat0.csv

Outputs:
    models/hcc_booster.lgb    — trained LightGBM booster
    models/hcc_metadata.pkl   — feature names, params, best iteration
    models/scaler.pkl         — standard scaler parameters
    models/training_report.txt — evaluation metrics
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_hcc_dataset, load_uci_hcv
from src.feature_engineering import extract_features
from src.model import HCCRiskModel, RISK_LABELS


def train(
    n_patients: int = 3000,
    uci_csv: str = None,
    n_estimators: int = 500,
    output_dir: str = "models",
    random_state: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.15,
) -> dict:
    """
    Full training pipeline: data → features → train → evaluate → save.

    Args:
        n_patients:    Number of synthetic patients (ignored if uci_csv provided).
        uci_csv:       Optional path to UCI hcvdat0.csv for real data training.
        n_estimators:  Max LightGBM boosting rounds.
        output_dir:    Directory to save model artefacts.
        random_state:  Random seed for reproducibility.
        test_size:     Fraction of data held out for final evaluation.
        val_size:      Fraction of training data used for early stopping.

    Returns:
        dict with evaluation metrics (auc_ovr, log_loss_val, report).
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    if uci_csv:
        print(f"Loading UCI HCV dataset from {uci_csv}...")
        df = load_uci_hcv(uci_csv)
        print(f"  Loaded {len(df)} patients from UCI dataset")
    else:
        print(f"Generating {n_patients} synthetic patients...")
        df = generate_hcc_dataset(n_patients=n_patients, random_state=random_state)

    print(f"  Class distribution:\n{df['risk_category'].value_counts().to_string()}")

    y = df["risk_label"].values

    # ── 2. Train / val / test split ───────────────────────────────────────────
    # Stratify on y to preserve class proportions in every split
    X_temp, X_test_df, y_temp, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"\nSplit sizes — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    X_train, feature_names, scaler = extract_features(X_train_df)
    X_val,   _, _                  = extract_features(X_val_df,   fit_scaler=scaler)
    X_test,  _, _                  = extract_features(X_test_df,  fit_scaler=scaler)

    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Feature names: {feature_names}")

    # ── 4. Train model ────────────────────────────────────────────────────────
    print("\nTraining LightGBM model...")
    model = HCCRiskModel(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
    print(f"  Best iteration: {model.best_iteration}")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    metrics = model.evaluate(X_test, y_test)

    print(f"\n  Test AUC (macro OvR): {metrics['auc_ovr']:.4f}")
    print(f"  Test log-loss:        {metrics['log_loss_val']:.4f}")
    print("\n  Classification Report:")
    print(metrics["report"])
    print("\n  Confusion Matrix (Low / Medium / High):")
    print(metrics["confusion_mat"])

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model.save(output_dir)

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved → {scaler_path}")

    # Save evaluation report
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("HCC Risk Model — Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset:         {'UCI HCV' if uci_csv else 'Synthetic'}\n")
        f.write(f"Patients:        {len(df)}\n")
        f.write(f"Train/Val/Test:  {len(y_train)}/{len(y_val)}/{len(y_test)}\n")
        f.write(f"Best iteration:  {model.best_iteration}\n\n")
        f.write(f"Test AUC (OvR):  {metrics['auc_ovr']:.4f}\n")
        f.write(f"Test log-loss:   {metrics['log_loss_val']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["report"])
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics["confusion_mat"]))
    print(f"✓ Report saved  → {report_path}")

    print("\n✅ Training complete!")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HCC risk prediction model"
    )
    parser.add_argument(
        "--n_patients", type=int, default=3000,
        help="Number of synthetic patients to generate (default: 3000)"
    )
    parser.add_argument(
        "--uci_csv", type=str, default=None,
        help="Path to UCI HCV hcvdat0.csv (optional; overrides synthetic data)"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=500,
        help="Max LightGBM boosting rounds (default: 500)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models",
        help="Directory for saved model files (default: models/)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    train(
        n_patients=args.n_patients,
        uci_csv=args.uci_csv,
        n_estimators=args.n_estimators,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
