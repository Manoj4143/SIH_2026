# Machine Learning Pipeline: AI Weather Nowcast

This directory contains the AI/ML pipelines for **Thunderstorm and Lightning Nowcasting (0–2 hours lead time)** using multi-sensor atmospheric observations.

## Methodology & Feature Engineering

### 1. Radar Features
- **Max Reflectivity ($Z_{\text{max}}$ in dBZ)**: Identifies strong updrafts and core precipitation.
- **Echo Top Height**: Altitude where reflectivity drops below 18 dBZ.
- **Vertically Integrated Liquid (VIL in $\text{kg/m}^2$)**: Measures liquid water content; sudden VIL surges indicate severe hail/storm risk.
- **Temporal Trends**: $\Delta Z / \Delta t$ over 10- and 20-minute windows.

### 2. Satellite Features (INSAT / GOES)
- **Cloud Top Brightness Temperature (CTBT)** in Thermal IR channel (10.8 $\mu\text{m}$).
- **Water Vapor (WV) - IR difference**: Indicates convective cloud penetration into the tropopause.
- **Cooling Rate**: $\Delta T / \Delta t$ indicating vigorous vertical cloud development.

### 3. Lightning Features
- **Flash Count & Density**: Total strokes per $5 \times 5\text{ km}$ grid in past 15 min.
- **Lightning Jump**: Rapid acceleration in total flash rate ($> 2\sigma$ above running mean), which precedes severe surface wind/hail by 15–30 minutes.

### 4. NWP Environmental Indices (WRF/GFS)
- **CAPE** (Convective Available Potential Energy): Atmospheric instability index.
- **CIN** (Convective Inhibition): Energy barrier to storm initiation.
- **0–6 km Bulk Wind Shear**: Propels cell organization into multicell or supercell clusters.

---

## Pipelines

| Script | Purpose |
| :--- | :--- |
| `pipelines/data_preprocessing.py` | Cleaning, interpolation, coordinate reprojection, and radar clutter filtering |
| `pipelines/feature_engineering.py` | Atmospheric index calculations (CAPE, VIL, lightning density, gradients) |
| `pipelines/train_thunderstorm.py` | Training XGBoost & Scikit-learn models for reflectivity $\ge 40\text{ dBZ}$ prediction |
| `pipelines/train_lightning.py` | Training gradient boosted classifiers for lightning occurrence (0–60 min) |
| `pipelines/evaluate.py` | Meteorology metrics: CSI (Critical Success Index), POD, FAR, Brier Score |

## Running Pipelines

```bash
cd ml
pip install -r requirements.txt
python pipelines/train_thunderstorm.py
python pipelines/train_lightning.py
```
Trained artifacts will be saved to `../models/`.
