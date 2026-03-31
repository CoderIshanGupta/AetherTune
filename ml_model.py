"""
Activity-based Music Recommendation Model
Improvements:
- Vectorized label generation (no row-wise apply)
- Hyperparameter-tuned RandomForest with class_weight balancing
- Hamming loss + per-label classification report instead of subset accuracy
- Label imbalance diagnostics
- Pipeline-based scaler+model for safe inference
- Reproducible age seed
- Graceful directory creation
- Type-annotated, docstring-documented functions
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH   = "data/spotify_dataset.csv"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "activity_pipeline.pkl")

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "age",
]

LABELS = ["studying", "driving", "meditating", "exercising"]

RNG_SEED = 42  # single seed used everywhere for reproducibility

# ── Label generation (vectorized) ────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary activity labels using vectorized boolean operations.
    Avoids row-wise apply() for a significant speed-up on large datasets.
    """
    df = df.copy()
    df["studying"]   = ((df["energy"] < 0.6) & (df["speechiness"] < 0.4)).astype(int)
    df["driving"]    = ((df["energy"] > 0.5) & (df["tempo"] > 90)).astype(int)
    df["meditating"] = ((df["acousticness"] > 0.6) & (df["energy"] < 0.4)).astype(int)
    df["exercising"] = ((df["energy"] > 0.7) & (df["tempo"] > 120)).astype(int)
    return df

# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_label_balance(y: pd.DataFrame) -> None:
    """Print positive-class ratio for each label to detect imbalance."""
    print("\n── Label balance (positive rate) ──")
    for col in y.columns:
        rate = y[col].mean()
        print(f"  {col:>12}: {rate:.2%}")
    print()

# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Return a sklearn Pipeline that scales features and fits a
    hyperparameter-tuned MultiOutputClassifier.

    Key RF improvements vs. defaults:
      - n_estimators=300  : more trees → lower variance
      - max_features="sqrt": default for classifiers, stated explicitly
      - min_samples_leaf=4 : reduces overfitting on noisy simulated labels
      - class_weight="balanced": compensates for label imbalance
      - n_jobs=-1          : parallel training across all CPU cores
    """
    from sklearn.preprocessing import StandardScaler

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RNG_SEED,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputClassifier(rf, n_jobs=-1)),
    ])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Print Hamming loss (fraction of wrong labels) and a per-label
    classification report. Both are more informative than subset accuracy
    for multi-label problems.
    """
    y_pred = pipeline.predict(X_test)
    hl = hamming_loss(y_test, y_pred)
    print(f"Hamming loss (lower=better): {hl:.4f}")
    print("\nPer-label classification report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Simulated age with fixed seed for reproducibility
    rng = np.random.default_rng(RNG_SEED)
    df["age"] = rng.integers(15, 60, size=len(df))

    # ── Feature & label prep ───────────────────────────────────────────────
    df = generate_labels(df)

    X = df[FEATURES]
    y = df[LABELS]

    print_label_balance(y)

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, shuffle=True
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("Training pipeline …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    evaluate(pipeline, X_test, y_test)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)   # compress=3 ≈ 50% smaller file
    print(f"\nPipeline saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
