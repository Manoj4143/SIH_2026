"""
Thunderstorm Nowcasting Model Training Pipeline:
Trains Scikit-learn Random Forest or XGBoost Classifier to predict convective initiation
and severe storm persistence (dBZ >= 40) at 30-min lead time.
"""

import os
import sys
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Ensure ml package in path
ML_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ROOT))

from pipelines.feature_engineering import AtmosphericFeatureExtractor


def train_thunderstorm_model():
    print("Starting Thunderstorm Nowcast model training pipeline...")
    fe = AtmosphericFeatureExtractor()
    df = fe.create_synthetic_training_dataset(num_samples=2500)

    feature_cols = [
        "current_max_dbz",
        "dbz_tendency_10m",
        "dbz_tendency_20m",
        "vil_kg_m2",
        "satellite_ir_kelvin",
        "is_overshooting_top",
        "lightning_flash_rate_15m",
        "cape_j_kg",
        "deep_layer_shear",
        "energy_shear_index",
    ]
    target_col = "target_thunderstorm_30m"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        from xgboost import XGBClassifier
        print("Using XGBoost Classifier for nowcasting...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("XGBoost not available; using Scikit-learn GradientBoostingClassifier...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save trained artifact
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "thunderstorm_xgb_v1.joblib"
    joblib.dump(model, out_path)
    print(f"Thunderstorm model successfully serialized to: {out_path}")


if __name__ == "__main__":
    train_thunderstorm_model()
