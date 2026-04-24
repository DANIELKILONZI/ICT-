#!/usr/bin/env python
"""
Train the XGBoost ML signal filter from labeled backtest data.

Reads the CSV produced by scripts/backtest_setup4.py, trains an XGBoost
classifier on the 10-feature vector defined in python/ml/signal_filter.py,
evaluates on a held-out test split, and saves the model to disk.

Once the model file exists, set `ml.enabled: true` in config.yaml to activate
the ML filter in the live trading system.

Usage:
    # Train from default backtest output
    python scripts/train_ml.py

    # Custom input / output paths
    python scripts/train_ml.py --input data/gbpusd_labels.csv --output ml/models/xgb_filter.pkl

    # Use a smaller held-out test fraction
    python scripts/train_ml.py --test-size 0.15

Workflow:
    1. Run scripts/backtest_setup4.py to produce a labeled CSV.
    2. Run this script to train the model.
    3. Set `ml.enabled: true` in config.yaml.
    4. The live system will automatically load and apply the filter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Feature columns must match build_feature_vector() in python/ml/signal_filter.py
FEATURE_COLS = [
    "d1_trend",
    "h1_trend",
    "atr_m5",
    "hour_utc",
    "m5_sweep",
    "price_zone",
    "h1_ob",
    "h1_fvg",
    "m5_fvg",
    "confluence_score",
]

_MIN_SAMPLES_WARNING = 50   # warn when training set is smaller than this


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost ML filter from labeled backtest data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/backtest_labels.csv",
        help="Labeled CSV produced by scripts/backtest_setup4.py",
    )
    parser.add_argument(
        "--output",
        default="ml/models/xgb_filter.pkl",
        help="Destination path for the trained model file",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of samples held out for evaluation (0.0–0.5)",
    )
    args = parser.parse_args()

    # ── Import dependencies ────────────────────────────────────────────────────
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        print(f"ERROR: Required package not installed: {exc}")
        print("Install with: pip install scikit-learn pandas numpy")
        sys.exit(1)

    # ── Load labeled data ──────────────────────────────────────────────────────
    input_path = REPO_ROOT / args.input
    if not input_path.exists():
        print(
            f"ERROR: Labeled data not found at {input_path}\n"
            "  Run scripts/backtest_setup4.py first to generate it."
        )
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Keep only resolved trades (drop OPEN rows if any slipped through)
    df = df[df["label"].isin([0, 1])].copy()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing feature columns in input CSV: {missing}")
        sys.exit(1)
    if "label" not in df.columns:
        print("ERROR: 'label' column not found in input CSV.")
        sys.exit(1)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(int)

    wins   = int(y.sum())
    losses = int((y == 0).sum())
    print(f"Dataset: {len(df)} samples  (wins={wins}  losses={losses})")

    if len(df) < _MIN_SAMPLES_WARNING:
        print(
            f"\nWARNING: Only {len(df)} labeled trades.  The model is unlikely to "
            f"generalize well.\n"
            f"  Gather at least {_MIN_SAMPLES_WARNING}–200 resolved trades before "
            f"enabling ML filtering in production.\n"
        )

    if wins == 0 or losses == 0:
        print(
            "ERROR: All samples belong to one class – cannot train a useful classifier."
        )
        sys.exit(1)

    # ── Train / test split ────────────────────────────────────────────────────
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y,
        )
    except ValueError:
        # Stratification fails when a class has fewer samples than n_splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42,
        )
    print(f"Train={len(X_train)}  Test={len(X_test)}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    out_path = REPO_ROOT / args.output
    from python.ml.signal_filter import train_model  # noqa: PLC0415
    train_model(np.array(X_train), np.array(y_train), save_path=out_path)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    try:
        import joblib
        model   = joblib.load(out_path)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
            auc_str = f"{auc:.4f}"
        except ValueError:
            auc_str = "n/a (single class in test split)"

        print(f"Test accuracy : {accuracy:.2%}")
        print(f"ROC-AUC       : {auc_str}")
        print("\nClassification report:")
        print(
            classification_report(
                y_test, y_pred,
                target_names=["LOSS (0)", "WIN (1)"],
                zero_division=0,
            )
        )

        # ── Feature importances ────────────────────────────────────────────────
        if hasattr(model, "feature_importances_"):
            print("Feature importances (descending):")
            pairs = sorted(
                zip(FEATURE_COLS, model.feature_importances_),
                key=lambda x: -x[1],
            )
            for name, imp in pairs:
                bar = "█" * max(1, int(imp * 50))
                print(f"  {name:<20s} {imp:.4f}  {bar}")

    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Post-training evaluation failed: {exc}")

    # ── Usage instructions ─────────────────────────────────────────────────────
    print(f"\nModel saved → {out_path}")
    print(
        "\nNext steps:\n"
        "  1. Review the win-rate and ROC-AUC above.\n"
        "  2. If results look reasonable, enable in config.yaml:\n"
        "       ml:\n"
        "         enabled: true\n"
        "  3. Tune `ml.confidence_threshold` (default 0.65) if needed.\n"
        "  4. Re-train periodically as new labeled trades accumulate."
    )


if __name__ == "__main__":
    main()
