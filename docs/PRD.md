# Product Requirement Document (PRD): Multi-Source Deep Learning Weather Nowcasting System

## 1. Product Overview
* **Product Name:** Multi-Source Deep Learning Weather Nowcasting System (SIH Problem Statement 26072)[cite: 1, 2].
* **Theme & Category:** Disaster Management Software solution developed under Smart India Hackathon 2026 by Team ASTRO[cite: 2].
* **Core Purpose:** Provides high-resolution, short-range (0–3 hours) real-time nowcasting for convective weather hazards, specifically tracking thunderstorm cell movement and generating total lightning strike probability maps[cite: 1, 2].
* **Technology Stack & Constraints:** Built on a 100% Free and Open-Source (FOSS) software stack (PyTorch, Py-ART, FastAPI, PostGIS, Leaflet.js) calibrated to run on low-cost hardware (NVIDIA RTX 3050 GPU with 4GB VRAM)[cite: 1].
* **Multi-Modal Data Fusion:** Ingests and fuses data from Doppler Weather Radar (DWR), INSAT-3D/3DR satellites, lightning detection networks, and Numerical Weather Prediction (NWP) atmospheric indices[cite: 1, 2].

## 2. Problem Statement
* **High NWP Initialization Latency:** Traditional Numerical Weather Prediction (NWP) models require >2 hours for initialization and run on 6–12 hour forecast cycles, making them too slow to provide timely warnings for fast-forming convective storms[cite: 1, 2].
* **Inadequacy of Single-Source Data:** Relying on a single observation type (e.g., radar alone or satellite alone) is insufficient for accurately predicting sudden thunderstorm onset and lightning generation[cite: 1, 2].
* **High Impact of Severe Convective Events:** Unexpected thunderstorms and lightning cause widespread damage across human lives, agricultural crops, power grids, and aviation ground operations[cite: 1, 2].
* **Hardware & Deployment Cost Barriers:** State-of-the-art meteorological deep learning models typically demand high-end industrial GPUs, restricting affordable local deployment across regional centers[cite: 1].

## 3. Goals
* **High-Resolution Short-Range Predictions:** Generate short-range forecasts (0–3 hours) at $1\text{ km} \times 1\text{ km}$ spatial resolution across 6 discrete 15-minute intervals ($t+15\text{m}$ to $t+180\text{m}$)[cite: 1].
* **Dual Hazard Forecasting:** Predict radar reflectivity cell movement/intensity ($\ge 35\text{ dBZ}$) and total lightning strike probability maps ($0.0 - 1.0$) simultaneously[cite: 1, 2].
* **Strict Hardware Footprint Optimization:** Bound input tensor sizes to $128 \times 128$ spatial pixels to cap peak GPU VRAM consumption at 2.48 GB (out of 4.00 GB available on RTX 3050)[cite: 1].
* **High Operational Accuracy & Low Latency:** Achieve Critical Success Index ($\text{CSI}) > 0.60$ at $\ge 35\text{ dBZ}$ reflectivity while serving map tile REST API requests in $<850\text{ ms}$[cite: 1].
* **Automated Multi-Channel Alerting:** Automatically identify severe risk polygons and issue geofenced advisory warnings to end-users[cite: 1, 2].

## 4. Target Users
* **IMD Forecasters & Meteorologists:** Require objective AI prediction tools and high-resolution spatial overlay maps to issue regional weather updates[cite: 2].
* **Disaster Management Authorities:** Depend on localized geofenced risk maps to pre-position emergency services and initiate protective actions[cite: 2].
* **General Public:** Need early local hazard notifications, advisory web banners, and district risk gauges to seek timely shelter[cite: 1, 2].
* **Aviation Operations:** Rely on early lightning and reflectivity warnings for ground crew safety and flight handling adjustments[cite: 2].
* **Agricultural Community:** Seek advance notice on lightning and severe storm paths to protect crops and livestock[cite: 2].

## 5. Core Features
* **Multi-Source Ingestion Pipeline:** Automated ingestion and spatial alignment (EPSG:3857) of radar reflectivity/velocity, INSAT-3D IR/WV channels, lightning point observations (converted via Gaussian Kernel Density Estimation), and atmospheric thermodynamic indices (CAPE, CIN, Wind) into standardized 8-channel tensors[cite: 1].
* **Hybrid Deep Learning Model Architecture:** Combines a Depthwise Separable Conv2D spatial encoder, a 2-layer ConvLSTM bottleneck for tracking temporal trajectories, and a skip-connected U-Net decoder with dual output heads[cite: 1, 2].
* **Low-VRAM & Mixed-Precision Optimization Engine:** Uses PyTorch Automatic Mixed Precision (AMP FP16) to execute single-precision computations with minimal VRAM overhead on consumer GPUs[cite: 1].
* **Interactive Leaflet.js Web Canvas:** Interactive mapping interface with CartoDB Positron base maps, animated 0–3 hour time sliders with historical playback ($t-30\text{m}$ to $t+180\text{m}$), and dynamic district risk widgets[cite: 1].
* **Geofenced Risk Thresholds & Automated Alert Engine:** Categorizes severe weather into four risk tiers (Green: Normal, Yellow: Advisory, Orange: Warning, Red: Severe) based on dBZ levels and lightning probabilities to fire active alerts[cite: 1].
* **FastAPI Dynamic Tile Server:** Asynchronous REST microservice that encodes model inference arrays into Cloud-Optimized GeoTIFFs (COG) and 32-bit PNG map tiles with custom color ramps[cite: 1].
* **Containerized Deployment Stack:** Complete multi-stage `docker-compose.yml` configuration integrated with NVIDIA Container Toolkit for fast, isolated microservice deployment[cite: 1].