# Project Persistent Context Memory (MEMORY.md)

## 1. Core Project Identity & Scope
* **Problem Statement:** SIH Problem Statement 26072 – Multi-Source Deep Learning Weather Nowcasting System[cite: 3].
* **Team:** Team ASTRO (Theme: Disaster Management)[cite: 3].
* **Core Purpose:** Provide real-time, high-resolution short-range (0–3 hours) nowcasting for convective weather hazards across 6 discrete 15-minute frames ($t+15\text{m}$ to $t+180\text{m}$)[cite: 3].
* **Dual Outputs:** Convective storm cell movement/intensity ($\ge 35\text{ dBZ}$) and total lightning strike probability maps ($0.0 - 1.0$)[cite: 3].

## 2. Hardware & Runtime Constraints
* **Primary Target GPU:** NVIDIA RTX 3050 (4 GB VRAM ceiling)[cite: 3].
* **VRAM Cap:** Peak PyTorch VRAM consumption must remain under $2.48\text{ GB}$ out of 4.00 GB available[cite: 3].
* **Software Stack Mandate:** 100% Free and Open-Source Software (PyTorch FP16 AMP, Py-ART, pysteps, FastAPI, PostgreSQL/PostGIS, Leaflet.js, CartoDB Positron)[cite: 3].
* **Spatial Canvas Resolution:** Input domain locked at $128 \times 128$ spatial pixels ($1\text{ km/pixel}$ resolution)[cite: 3].

## 3. Data Pipeline & Tensor Memory Standard
* **Modalities & Providers:**
  * **INSAT-3D/3DR Satellite:** IR Brightness Temp ($10.8\mu\text{m}$) & Water Vapor ($6.8\mu\text{m}$) via ISRO MOSDAC[cite: 3].
  * **Doppler Weather Radar:** Reflectivity ($\text{dBZ}$) & Radial Velocity via IMD/NOAA NEXRAD AWS[cite: 3].
  * **Lightning:** Flash count coordinates via Blitzortung/IITM processed using Gaussian Kernel Density Estimation (KDE)[cite: 3].
  * **NWP Reanalysis:** CAPE, CIN, and Wind Magnitude ($850\text{ hPa}$) via ERA5 / Open-Meteo[cite: 3].
* **5D Input Tensor Format:** $[B \times T_{in} \times C \times H \times W] = [2 \times 4 \times 8 \times 128 \times 128]$ containing 8 normalized channels ($[\text{dBZ}, \text{Vel}, \text{IR}_{10.8}, \text{WV}_{6.8}, \text{Lightning}_{\text{KDE}}, \text{CAPE}, \text{CIN}, \text{Wind}_{\text{Mag}}]$)[cite: 3].
* **Spatial Coordinate System:** All raster and vector layers reprojected to EPSG:3857 (Web Mercator)[cite: 3].

## 4. Deep Learning Model Architecture
* **Topology:** Depthwise Separable Conv2D Spatial Encoder + 2-Layer ConvLSTM Bottleneck + Skip-Connected U-Net Decoder[cite: 3].
* **Precision:** PyTorch Automatic Mixed Precision (`torch.cuda.amp`) executing in FP16[cite: 3].
* **Loss Function:** Composite Focal Loss ($\gamma=2.0, \alpha=0.75$) + $0.5 \cdot (1 - \text{SSIM})$[cite: 3].
* **Performance Benchmarks:** Critical Success Index ($\text{CSI}) > 0.60$ at $\ge 35\text{ dBZ}$ reflectivity[cite: 3].

## 5. API & Map Tile Service Specifications
* **Tile Rendering Target:** Serve 32-bit PNG map tiles via FastAPI in $< 850\text{ ms}$[cite: 3].
* **Automated Schedule:** Background inference worker executes every 15 minutes in `nowcast_engine/`[cite: 3].
* **Core Endpoints:**
  * `GET /api/v1/nowcast/latest` – Location-based point nowcast array[cite: 3].
  * `GET /api/v1/radar/tile/{z}/{x}/{y}` – Dynamic raster map tile stream[cite: 3].
  * `GET /api/v1/alerts/active` – GeoJSON active hazard polygons from PostGIS[cite: 3].
  * `POST /api/v1/inference/trigger` – Manual trigger token for demonstration ETL[cite: 3].

## 6. Weather Risk Action Matrix
* **Green (Normal):** $< 20\text{ dBZ}$, Lightning Prob $< 20\%$ (Standard monitoring)[cite: 3].
* **Yellow (Advisory):** $20 - 35\text{ dBZ}$, Lightning Prob $20 - 55\%$ (Advisory web banner)[cite: 3].
* **Orange (Warning):** $35 - 45\text{ dBZ}$, Lightning Prob $55 - 80\%$ (District warning & push notification)[cite: 3].
* **Red (Severe):** $> 45\text{ dBZ}$, Lightning Prob $> 80\%$ (Emergency alert broadcast)[cite: 3].