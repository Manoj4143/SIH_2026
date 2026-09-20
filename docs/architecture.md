# System Architecture Specification (ARCHITECTURE.md)

## 1. High-Level Architecture & Application Flow
* **Multi-Source Data Ingestion:** Automated workers fetch raw inputs every 15 minutes, including Doppler Weather Radar volumes, INSAT-3D IR/WV channels, lightning strike coordinates, and NWP thermodynamic indices.
* **FOSS Data Processing Pipeline:** Raw inputs are processed using Py-ART and pysteps for ground clutter removal, reprojected to EPSG:3857, and resampled to a uniform $128 \times 128$ spatial grid ($1\text{ km/pixel}$ resolution).
* **5D Tensor Assembly:** Preprocessed channels are normalized and concatenated into an 8-channel PyTorch tensor $[B \times T_{in} \times C \times H \times W] = [2 \times 4 \times 8 \times 128 \times 128]$ containing dBZ, radial velocity, IR, WV, lightning KDE, CAPE, CIN, and wind magnitude.
* **Deep Learning Nowcasting Engine:** A hybrid model featuring a Depthwise Separable Conv2D Spatial Encoder, a 2-Layer ConvLSTM Bottleneck, and a Skip-Connected U-Net Decoder processes input tensors using Automatic Mixed Precision (FP16) on an RTX 3050 GPU (2.48 GB VRAM peak)[cite: 1, 2].
* **Dual Output Generation:** The model predicts 6 discrete sequence frames ($t+15\text{m}$ to $t+180\text{m}$) across two output heads: Radar Reflectivity ($\ge 35\text{ dBZ}$) and Lightning Strike Density ($0.0 - 1.0$).
* **FastAPI Backend & PostGIS:** Inference outputs are converted to Cloud-Optimized GeoTIFFs (COG) and PNG map tiles while risk polygons are stored in PostGIS for spatial querying.
* **Leaflet.js Web Dashboard:** Serves real-time map tile overlays, interactive 0–3 hour time sliders, and dynamic geofenced alert banners to end-users.

---

## 2. Technology Stack
* **Deep Learning Framework:** PyTorch with CUDA 12 and Automatic Mixed Precision (`torch.cuda.amp`) for memory-efficient FP16 execution.
* **Meteorological & Geospatial Data Libraries:** Py-ART (ARM Radar Toolkit), pysteps, Rasterio, GDAL, and NumPy for radar processing, optical flow baselines, grid alignment, and spatial reprojection.
* **Backend API Framework:** FastAPI running on Uvicorn asynchronous server for tile streaming and REST endpoint handling.
* **Database & Spatial Vector Engine:** PostgreSQL with PostGIS extension for geo-indexing, polygon bounding box queries, and active alert state management.
* **Frontend Web Stack:** Leaflet.js mapping engine with CartoDB Positron base tiles, HTML5 canvas, JavaScript (ES6+), and Chart.js for risk gauge widgets.
* **Containerization & Infrastructure:** Docker, Docker-Compose, and NVIDIA Container Toolkit for low-overhead local GPU deployment.

---

## 3. Directory & Folder Structure
sih-26072-nowcast/
├── docker-compose.yml             # Multi-container orchestrator (API, Engine, UI)
├── README.md                      # Project setup & run instructions
├── architecture.md                # System architecture documentation
├── PRD.md                         # Product requirement document
├── data_pipeline/                 # Data Ingestion & Preprocessing Microservice
│   ├── ingestion/                 # Scripts for MOSDAC, IMD/NEXRAD, ERA5, Blitzortung
│   ├── preprocessing/             # Py-ART clutter filtering & spatial alignment scripts
│   ├── normalization/             # MinMax scaling & Gaussian KDE transform logic
│   └── loaders/                   # PyTorch dataset generators emitting [8, 128, 128] tensors
├── nowcast_engine/                # Deep Learning Inference & Training Module
│   ├── models/                    # Conv2D Encoder, ConvLSTM Bottleneck, U-Net Decoder
│   ├── loss/                      # Composite Focal Loss + SSIM Loss implementation
│   ├── weights/                   # Saved PyTorch checkpoint weights (.pt)
│   └── inference_worker.py        # 15-minute automated inference loop
├── backend_api/                   # FastAPI Web & Spatial Service
│   ├── app/
│   │   ├── main.py                # FastAPI entry point & CORS configuration
│   │   ├── api/                   # REST routes (/nowcast/latest, /radar/tile, /alerts)
│   │   ├── db/                    # PostgreSQL/PostGIS connection setup & schemas
│   │   └── services/              # COG to 32-bit PNG tile renderer
│   └── Dockerfile                 # Backend container configuration
└── web_dashboard/                 # Refactored Front-End Repository ('weatherwebsite')
    ├── public/                    # Static assets & icons
    ├── src/
    │   ├── components/            # Leaflet map canvas, time slider, risk gauges
    │   ├── css/                   # Responsive UI styling
    │   └── js/                    # API integration & time-playback logic
    ├── nginx.conf                 # Nginx web server configuration
    └── Dockerfile                 # Web dashboard container setup

---

## 4. Component Communication & Data Flow
* **Data Pipeline to Model Engine:** The preprocessing service transforms incoming raw sensor data into standardized HDF5 files every 15 minutes, emitting PyTorch tensors directly to shared GPU memory buffers[cite: 1].
* **Model Engine to Backend API:** The PyTorch inference worker outputs raw model array forecasts, writes Cloud-Optimized GeoTIFFs (COG) to temporary storage, and posts active hazard vector polygons directly to the PostGIS spatial database[cite: 1].
* **Backend API to PostGIS:** FastAPI issues spatial queries against PostGIS using bounding box (`bbox`) parameters to extract active severe weather risk polygons[cite: 1].
* **Backend API to Front-End Canvas:** The frontend Leaflet map layer queries `GET /api/v1/radar/tile/{z}/{x}/{y}` to dynamically fetch custom color-mapped 32-bit PNG raster map tiles[cite: 1].
* **Alert Trigger Protocol to Front-End:** The frontend polls `GET /api/v1/alerts/active` with user geolocation coordinates, automatically rendering top alert banners (Yellow, Orange, Red) and district gauge updates when severe weather thresholds are breached[cite: 1].