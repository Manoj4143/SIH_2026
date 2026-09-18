"""
Lightning Strike Nowcast Training Pipeline:
Trains Scikit-learn / XGBoost model to predict lightning stroke occurrence and jump events.
"""

import sys
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ML_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ROOT))

from pipelines.feature_engineering import AtmosphericFeatureExtractor


def train_lightning_model():
    print("Starting Lightning Nowcast model training pipeline...")
    fe = AtmosphericFeatureExtractor()
    df = fe.create_synthetic_training_dataset(num_samples=2500)

    feature_cols = [
        "current_max_dbz",
        "dbz_tendency_10m",
        "vil_kg_m2",
        "satellite_ir_kelvin",
        "is_overshooting_top",
        "lightning_flash_rate_15m",
        "cape_j_kg",
        "deep_layer_shear",
    ]
    target_col = "target_lightning_30m"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        from xgboost import XGBClassifier
        print("Using XGBoost Classifier for lightning prediction...")
        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        print("Using Scikit-learn RandomForestClassifier for lightning prediction...")
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "lightning_xgb_v1.joblib"
    joblib.dump(model, out_path)
    print(f"Lightning model successfully serialized to: {out_path}")


if __name__ == "__main__":
    train_lightning_model()
