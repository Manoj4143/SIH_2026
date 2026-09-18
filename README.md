# AI Weather Nowcast

> **AIML-based Nowcasting of thunderstorm and lightning using atmospheric observations including multiple radars, satellite, lightning and model data.**

---

## 🌩️ Overview

**AI Weather Nowcast** is a modern, modular, production-style platform designed to provide ultra-short-term (0–2 hours) high-resolution predictions of severe weather events:
- **Thunderstorm Initiation, Growth, and Decay** (convective cell tracking and dBZ reflectivity evolution).
- **Lightning Stroke Density & Strike Probabilities** (Cloud-to-Ground and Intra-Cloud events).
- **Multi-Sensor Fusion**: Combines Doppler Weather Radars (DWR), Geostationary Satellite imagery (INSAT-3D/3DR / GOES), Lightning Detection Networks (LDN), and Numerical Weather Prediction (NWP - WRF/GFS) parameters.

---

## 🏛️ Project Architecture

```
ai-weather-nowcast/
├── frontend/             # React (Vite) + Leaflet interactive weather map dashboard
├── backend/              # Python FastAPI REST API with async processing
├── ml/                   # Atmospheric ML pipelines (Scikit-learn, XGBoost)
├── data/                 # Raw and processed observation data store
├── database/             # SQLite (SQLAlchemy ORM) metadata and prediction store
├── models/               # Serialized model artifacts (.joblib / .json)
└── README.md             # Project documentation and developer guide
```

### Component Details

| Module | Description | Tech Stack |
| :--- | :--- | :--- |
| **Frontend** | Interactive geospatial nowcast dashboard, radar reflectivity layer viewer, lightning stroke overlays, risk alerts | React 18, Vite, Leaflet, OpenStreetMap |
| **Backend** | Fast, asynchronous REST API serving geospatial nowcasts, sensor feeds, and prediction models | FastAPI, Uvicorn, Pydantic v2 |
| **AI/ML** | Feature engineering (CAPE, shear, max-Z, VIL), cell tracking, and gradient boosted nowcasters | Python, Scikit-learn, XGBoost, NumPy, Pandas |
| **Database** | Lightweight relational storage for stations, radar scans, strike logs, and historical predictions | SQLite, SQLAlchemy ORM |
| **Data Lake** | Directory structure for multi-radar volume sweeps, satellite NetCDF/HDF, and lightning point data | Structured File System |
| **Models** | Model registry for serialized weights, training metrics, and feature schemas | Joblib, XGBoost JSON |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.10+ (Tested on Python 3.12)
- **Node.js**: 18+ (Tested on Node 22)
- **VS Code**: Recommended for development

---

### 1. Backend Setup

```bash
# Navigate to backend
cd ai-weather-nowcast/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Initialize the SQLite database
cd ../database
python init_db.py
cd ../backend

# Start FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Documentation (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

### 2. Frontend Setup

```bash
# Navigate to frontend
cd ai-weather-nowcast/frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
- Dashboard UI: [http://localhost:5173](http://localhost:5173)

---

### 3. ML Pipeline Setup

```bash
# Navigate to ML directory
cd ai-weather-nowcast/ml

# Install ML dependencies
pip install -r requirements.txt

# Run feature engineering & sample model training
python pipelines/train_thunderstorm.py
python pipelines/train_lightning.py
```

---

## 💻 Running in VS Code

This repository contains pre-configured `.vscode/` configurations:
1. Open the project in VS Code:
   ```bash
   code ai-weather-nowcast
   ```
2. Press **`Ctrl+Shift+P`** and run:
   - `Tasks: Run Task` -> `Start All (Backend + Frontend)`
3. Or go to the **Run & Debug** panel (`Ctrl+Shift+D`):
   - Select **Full Stack: Backend + Frontend** and press `F5`.

---

## 🛰️ Atmospheric Data Sources

- **Doppler Weather Radars (DWR)**: Reflectivity ($Z$), Radial Velocity ($V$), Spectrum Width ($W$), Vertically Integrated Liquid (VIL).
- **Satellite (INSAT-3D/3DR, GOES)**: Thermal Infrared (TIR1/TIR2), Water Vapor (WV) channels, Cloud Top Brightness Temperature (CTBT).
- **Lightning Detection Network (LDN)**: Time, latitude, longitude, peak current (kA), polarity, stroke type (CG/IC).
- **Numerical Weather Prediction (NWP)**: Convective Available Potential Energy (CAPE), Convective Inhibition (CIN), 0–6 km bulk wind shear.

---

## 📈 ML Problem Formulation

Nowcasting is framed as a spatial-temporal classification & regression task:
1. **Thunderstorm Cell Prediction (0–120 min)**:
   - Target: Likelihood of radar reflectivity exceeding severe threshold ($Z \ge 40\text{ dBZ}$) at lead times $t + 15, 30, 60, 120\text{ min}$.
   - Model: Gradient Boosted Trees (XGBoost) with spatio-temporal lag features.
2. **Lightning Strike Prediction (0–60 min)**:
   - Target: Binary occurrence and density of lightning strikes in $5\text{ km} \times 5\text{ km}$ grid cells.
   - Evaluation Metrics: Critical Success Index (CSI), Probability of Detection (POD), False Alarm Ratio (FAR), Brier Score.

---

## 📄 License
MIT License.
