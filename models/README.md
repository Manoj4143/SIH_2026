# Model Registry: AI Weather Nowcast

This directory contains trained serializations of Scikit-learn and XGBoost atmospheric nowcasting models.

## Expected Artifacts

- `thunderstorm_xgb_v1.joblib`: Severe convection prediction model ($Z \ge 40\text{ dBZ}$)
- `lightning_xgb_v1.joblib`: Lightning occurrence (0–60 min) classifier
- `metadata.json`: Model version, training date, hyperparameters, and meteorological evaluation metrics (POD, FAR, CSI).

## Training New Models
Run the training pipeline scripts in `ml/`:
```bash
python ../ml/pipelines/train_thunderstorm.py
python ../ml/pipelines/train_lightning.py
```
Models will be saved directly into this directory.
