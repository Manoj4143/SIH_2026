# Development Rules & Coding Guidelines (RULES.md)

## 1. General Principles
* **FOSS Compliance:** Maintain 100% Free and Open-Source Software (FOSS) tooling; strictly avoid proprietary APIs or locked dependencies.
* **Low Hardware Footprint:** All model training, evaluation, and inference must execute under 2.5 GB peak VRAM on an NVIDIA RTX 3050 (4GB total VRAM budget).
* **Microservice Isolation:** Data ingestion, model execution, REST API handling, and UI rendering must remain strictly decoupled microservices.
* **Coordinate Consistency:** Every spatial layer across satellite, radar, and lightning datasets must be reprojected to EPSG:3857 (Web Mercator) before tensor assembly.
* **Deterministic Reproducibility:** Fixed random seeds must be set across PyTorch, NumPy, and Python standard libraries during training and test validation.

## 2. Development Rules
* **Version Control & Branching:** `main` is protected; all changes must pass through feature branches (`feature/*`, `bugfix/*`) with linear git history.
* **Environment Security:** Never commit credentials, database URI strings, or API secrets; all variables must be ingested via `.env` files.
* **Tensor Verification:** Input tensor dimensions must be validated programmatically at ingestion boundaries to guarantee `[B, T, C, H, W] = [2, 4, 8, 128, 128]` before model forwarding.
* **Automated Inference Loops:** The inference pipeline in `nowcast_engine/` must run asynchronously every 15 minutes, failing gracefully with fallback alerts if data feeds drop.
* **Containerized Parity:** Local development, testing, and production runtime environments must use the multi-container setup configured in `docker-compose.yml`.

## 3. Technology & Coding Standards
* **Python (Data Pipeline, ML Engine & Backend):**
  * Enforce strict type hinting (`mypy`) and adherence to PEP 8 standard formatting across all `.py` files.
  * Use PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast()`) for FP16 inference execution to optimize VRAM utilization.
  * Maintain non-blocking asynchronous routing in FastAPI using `async def` and Uvicorn workers for all tile and alert endpoints.
* **Geospatial & Radar Processing:**
  * Clamp or mask invalid values (e.g., radar non-reflectivity noise below $0\text{ dBZ}$) using Py-ART clutter mitigation routines.
  * Store output rasters exclusively as Cloud-Optimized GeoTIFFs (COG) to enable rapid HTTP range requests for map tiling.
* **JavaScript & Web Dashboard:**
  * Write modern ES6+ modular JavaScript without bloated heavy frameworks.
  * Avoid DOM memory leaks in the Leaflet.js map time-slider loop by explicitly purging old dynamic tile layers during timeline scrubbing.

## 4. Project Structure Rules
* **`data_pipeline/` Scope:** Handles raw radar/satellite ingestion, spatial warping, Gaussian KDE transforms for lightning, and HDF5 dataset generation.
* **`nowcast_engine/` Scope:** Contains neural network architectures (Conv2D Spatial Encoder, ConvLSTM, U-Net Decoder), checkpoint weights (`.pt`), and the 15-minute scheduled inference engine.
* **`backend_api/` Scope:** Contains FastAPI routes, PostGIS spatial queries, active hazard alert handlers, and dynamic PNG map tile renderers.
* **`web_dashboard/` Scope:** Contains static UI assets, Leaflet map initializations, time-slider controls, dynamic risk gauges, and Nginx configurations.
* **Inter-Service Communication Boundary:** Direct cross-service python imports are prohibited; microservices interact exclusively through REST APIs, shared persistent volume mounts (HDF5/COG rasters), or PostGIS query endpoints.