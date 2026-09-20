# Multi-Source Deep Learning Weather Nowcasting System (SIH 26072)

## 1. Project Overview
* **Problem Statement:** SIH Problem Statement 26072 – Multi-Source DL Weather Nowcasting System[cite: 3].
* **Team & Theme:** Developed under Smart India Hackathon 2026 by Team ASTRO under the Disaster Management theme[cite: 3].
* **Core Purpose:** High-resolution short-range (0–3 hours) real-time nowcasting for severe convective weather hazards at $1\text{ km} \times 1\text{ km}$ spatial resolution across 6 discrete 15-minute intervals ($t+15\text{m}$ to $t+180\text{m}$)[cite: 3].
* **Dual Hazard Predictions:** Simultaneously tracks convective thunderstorm cell movement ($\ge 35\text{ dBZ}$) and total lightning strike probability maps ($0.0 - 1.0$)[cite: 3].
* **Hardware Calibration:** Specifically optimized to run on budget hardware (NVIDIA RTX 3050 GPU with 4GB VRAM), capping peak PyTorch VRAM at 2.48 GB[cite: 3].
* **100% FOSS Stack:** Built entirely on Free and Open-Source Software including PyTorch, Py-ART, pysteps, FastAPI, PostgreSQL/PostGIS, Leaflet.js, and CartoDB Positron[cite: 3].

---

## 2. Key Features
* **Multi-Modal Data Fusion:** Ingests and normalizes 8 parallel channels from INSAT-3D/3DR satellites, Doppler Weather Radar (DWR), Blitzortung lightning network, and ERA5 NWP atmospheric reanalysis into a standardized $[2, 4, 8, 128, 128]$ PyTorch tensor[cite: 3].
* **Hybrid Deep Learning Architecture:** Features a Depthwise Separable Conv2D Spatial Encoder, a 2-Layer ConvLSTM Bottleneck for cell trajectory tracking, and a Skip-Connected U-Net Decoder executing via PyTorch Automatic Mixed Precision (FP16)[cite: 3].
* **FastAPI Microservice Engine:** Asynchronous REST server serving Cloud-Optimized GeoTIFFs (COG) and 32-bit PNG map tiles in under 850 ms[cite: 3].
* **Refactored Web Dashboard:** Interactive Leaflet.js map canvas featuring an animated 0–3 hour time slider, dynamic district risk gauges, and geofenced alert banners[cite: 3].
* **Automated 4-Tier Risk Matrix:** Dynamic alerts categorized into Green (Normal), Yellow (Advisory), Orange (Warning), and Red (Severe Risk) based on reflectivity and lightning probability thresholds[cite: 3].

---

## 3. Technology Stack
* **Deep Learning Framework:** PyTorch with CUDA 12 and Automatic Mixed Precision (`torch.cuda.amp`)[cite: 3].
* **Geospatial & Radar Processing:** Py-ART (ARM Radar Toolkit), pysteps, Rasterio, GDAL, NumPy[cite: 3].
* **Backend API & Database:** FastAPI running on Uvicorn asynchronous server, PostgreSQL with PostGIS extension[cite: 3].
* **Frontend UI Canvas:** Leaflet.js mapping engine, CartoDB Positron base tiles, HTML5 Canvas, JavaScript (ES6+), Chart.js[cite: 3].
* **Infrastructure & Containerization:** Docker, Docker-Compose, NVIDIA Container Toolkit[cite: 3].

---

## 4. Hardware Requirements
* **GPU:** NVIDIA RTX 3050 (or any CUDA-capable GPU with $\ge 4\text{ GB}$ VRAM)[cite: 3].
* **Peak Memory Usage:** Capped at 2.48 GB VRAM during full FP16 backpropagation and inference[cite: 3].
* **System RAM:** 16 GB minimum recommended.
* **Storage:** 20 GB free SSD storage for dataset caching, COG output rasters, and model checkpoints.

---

## 5. Quick Start & Setup Guide
* **Step 1: Clone the Repository**
  ```bash
  git clone [https://github.com/team-astro/sih-26072-nowcast.git](https://github.com/team-astro/sih-26072-nowcast.git)
  cd sih-26072-nowcast

```

* **Step 2: Environment Configuration**
```bash
cp .env.example .env
# Configure local database credentials and server settings

```


* **Step 3: Build and Launch Multi-Container Stack**
```bash
docker-compose up --build -d

```


* **Step 4: Verify Active Services**
* Web Dashboard UI: `http://localhost:80`
* FastAPI Interactive Docs: `http://localhost:8000/docs`
* PostGIS Database Container: `localhost:5432`



---

## 6. API Endpoint Summary

* **`GET /api/v1/nowcast/latest`:** Returns a 6-step probability forecast array and storm intensity trend vectors for specified latitude/longitude coordinates.


* **`GET /api/v1/radar/tile/{z}/{x}/{y}`:** Streams 32-bit PNG map tiles with custom meteorological color ramps.


* **`GET /api/v1/alerts/active`:** Serves a GeoJSON FeatureCollection of active severe risk polygons within a bounding box.


* **`POST /api/v1/inference/trigger`:** Manual ETL pipeline trigger endpoint for evaluation and live jury testing.



---

## 7. Directory Structure

```text
sih-26072-nowcast/
├── docker-compose.yml             # Multi-container orchestrator (API, Engine, UI)
├── README.md                      # Project setup & run instructions
├── architecture.md                # System architecture documentation
├── PRD.md                         # Product requirement document
├── rules.md                       # Development guidelines & standards
├── ALLSpecification.md           # Master technical specification file
├── TASK.md                        # Implementation task checklist
├── MEMORY.md                      # Persistent context memory
├── data_pipeline/                 # Data Ingestion & Preprocessing Microservice
├── nowcast_engine/                # Deep Learning Inference & Training Module
├── backend_api/                   # FastAPI Web & Spatial Service
└── web_dashboard/                 # Refactored Front-End Repository ('weatherwebsite')

```

---

## 8. License & Acknowledgments

* **License:** Open-source software released under the MIT License.


* **Data Sources:** ISRO MOSDAC, IMD, NOAA NEXRAD AWS, Blitzortung, ECMWF ERA5.


* **Competition:** Smart India Hackathon 2026 (Problem Statement 26072).



```

```