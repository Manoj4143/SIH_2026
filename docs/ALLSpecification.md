# Comprehensive System Specification (ALLSpecification.md)

## 1. Project Overview & Operational Scope
* **Product Identification:** Multi-Source Deep Learning Weather Nowcasting System (SIH Problem Statement 26072).
* **Team & Context:** Developed under Smart India Hackathon 2026 by Team ASTRO within the Disaster Management category.
* **Primary Objective:** Deliver real-time, high-resolution short-range (0–3 hours) nowcasts at $1\text{ km} \times 1\text{ km}$ spatial resolution across 6 discrete 15-minute sequence intervals ($t+15\text{m}$ to $t+180\text{m}$).
* **Dual Hazard Outputs:** Simultaneously tracks thunderstorm cell movement ($\ge 35\text{ dBZ}$) and total lightning strike probability maps ($0.0 - 1.0$).
* **Hardware Footprint Optimization:** Specifically calibrated to execute on consumer-grade NVIDIA RTX 3050 hardware (4GB VRAM limit), capping peak PyTorch memory consumption at 2.48 GB.
* **FOSS Architecture Policy:** Built entirely using 100% Free and Open-Source Software (PyTorch, Py-ART, pysteps, FastAPI, PostgreSQL/PostGIS, Leaflet.js).

---

## 2. Multi-Modal Data Ingestion & Normalization Matrix
* **INSAT-3D/3DR Satellite Data:**
  * Ingests IR Brightness Temperature ($10.8\mu\text{m}$) and Water Vapor ($6.8\mu\text{m}$) channels at native 4 km / 15-min resolution from ISRO MOSDAC.
  * Clipped to $[180\text{K}, 320\text{K}]$, MinMax scaled to $[0, 1]$, and reprojected to EPSG:3857 grid.
* **Doppler Weather Radar (DWR):**
  * Ingests Reflectivity ($\text{dBZ}$) and Radial Velocity at native 1 km / 10-min resolution from IMD / NOAA NEXRAD AWS feeds.
  * Filtered via Py-ART clutter mitigation, converted from polar to $128 \times 128$ Cartesian grid, clipped to $[-10, 70\text{ dBZ}]$, and scaled to $[0, 1]$.
* **Lightning Strike Point Data:**
  * Ingests Flash Count & Location Coordinates from Blitzortung / IITM networks.
  * Processed via Gaussian Kernel Density Estimation (KDE) onto $128 \times 128$ spatial grid and log-transformed into a continuous density field.
* **NWP Atmospheric Reanalysis (ERA5 / Open-Meteo):**
  * Ingests CAPE, CIN, and U/V Wind vector parameters ($850\text{ hPa}$) at native 25 km / 1-hour resolution.
  * Spatial bicubic upsampling to $128 \times 128$ grid and linear temporal interpolation between 1-hour forecast cycles.
* **5D Input Tensor Standard:**
  * Concatenated every 15 minutes into a PyTorch tensor $\mathbf{X}_{in} \in \mathbb{R}^{B \times T_{in} \times C \times H \times W} = [2 \times 4 \times 8 \times 128 \times 128]$ containing 8 normalized channels ($[\text{dBZ}, \text{Vel}, \text{IR}_{10.8}, \text{WV}_{6.8}, \text{Lightning}_{\text{KDE}}, \text{CAPE}, \text{CIN}, \text{Wind}_{\text{Mag}}]$).

---

## 3. Deep Learning Architecture & Meteorological Metrics
* **Hybrid Model Topology:**
  * **Spatial Encoder:** Depthwise Separable Conv2D layers for low-compute spatial feature extraction.
  * **Spatiotemporal Bottleneck:** 2-Layer ConvLSTM cell structure for tracking cell motion trajectories.
  * **Decoder:** Skip-connected U-Net decoder using transposed convolutions to reconstruct $128 \times 128$ resolution frames.
  * **Dual Output Heads:** Concurrent 6-frame forecasts for Reflectivity ($\ge 35\text{ dBZ}$) and Lightning Density ($0.0 - 1.0$).
* **Mixed Precision Execution:** Uses PyTorch Automatic Mixed Precision (`torch.cuda.amp`) for FP16 inference on RTX 3050 GPUs.
* **Loss Function Formulation:**
  * Combined loss: $\mathcal{L}_{Total} = \mathcal{L}_{Focal}(\gamma=2.0, \alpha=0.75) + 0.5 \cdot \left(1 - \text{SSIM}(\hat{Y}, Y)\right)$.
* **Target Evaluation Benchmarks:**
  * Critical Success Index $\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}} > 0.60$ at $\ge 35\text{ dBZ}$ reflectivity.
  * Probability of Detection ($\text{POD}$) and False Alarm Ratio ($\text{FAR}$) performance validation[cite: 3].

---

## 4. System Directory & Microservice Architecture
* **Repository Hierarchy:**
```text
sih-26072-nowcast/
├── docker-compose.yml             # Multi-container orchestrator (API, Engine, UI)
├── README.md                      # Project setup & run instructions
├── architecture.md                # System architecture documentation
├── PRD.md                         # Product requirement document
├── rules.md                       # Development guidelines & standards
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