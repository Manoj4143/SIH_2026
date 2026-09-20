# Engineering Task Checklist (TASK.md)

## Phase 1: Data Engine & Preprocessing Microservice (Weeks 1–2)
- [ ] Automate data ingestion scripts for MOSDAC INSAT-3D, ERA5, and Blitzortung feeds[cite: 3].
- [ ] Build Py-ART Cartesian gridding scripts to process raw radar reflectivity into $128 \times 128$ grids[cite: 3].
- [ ] Implement Py-ART ground clutter filtering and Doppler velocity processing[cite: 3].
- [ ] Execute spatial re-projection from native coordinates (EPSG:4326) to Web Mercator (EPSG:3857)[cite: 3].
- [ ] Implement Gaussian Kernel Density Estimation (KDE) transform for lightning strike points[cite: 3].
- [ ] Construct dataset loaders to normalize and output 8-channel PyTorch tensors $[8, 128, 128]$[cite: 3].

## Phase 2: Deep Learning Model Architecture & Training (Weeks 3–4)
- [ ] Implement Depthwise Separable Conv2D Spatial Encoder in PyTorch[cite: 3].
- [ ] Construct 2-Layer ConvLSTM Bottleneck for temporal trajectory tracking[cite: 3].
- [ ] Implement Skip-Connected U-Net Decoder with Dual Output Heads (Reflectivity & Lightning Density)[cite: 3].
- [ ] Configure PyTorch Automatic Mixed Precision (`torch.cuda.amp`) FP16 training pipeline[cite: 3].
- [ ] Implement composite Focal Loss ($\gamma=2.0, \alpha=0.75$) + SSIM loss function[cite: 3].
- [ ] Train model on host NVIDIA RTX 3050 GPU, capping peak VRAM usage at $2.48\text{ GB}$[cite: 3].
- [ ] Evaluate trained checkpoints against meteorological metrics ($\text{CSI} > 0.60$ at $\ge 35\text{ dBZ}$, POD, FAR)[cite: 3].

## Phase 3: FastAPI Backend & PostGIS Integration (Weeks 5–6)
- [ ] Set up PostgreSQL database with PostGIS spatial extensions enabled[cite: 3].
- [ ] Build dynamic PNG tile renderer service converting Cloud-Optimized GeoTIFFs (COG) to map tiles in $<850\text{ ms}$[cite: 3].
- [ ] Implement `GET /api/v1/nowcast/latest` REST endpoint for point forecast queries[cite: 3].
- [ ] Implement `GET /api/v1/radar/tile/{z}/{x}/{y}` REST endpoint for map tile streaming[cite: 3].
- [ ] Implement `GET /api/v1/alerts/active` REST endpoint returning GeoJSON risk polygons[cite: 3].
- [ ] Implement `POST /api/v1/inference/trigger` manual ETL trigger endpoint[cite: 3].
- [ ] Implement automated 15-minute background inference worker in `nowcast_engine/`[cite: 3].

## Phase 4: Web Dashboard UI Refactoring (`weatherwebsite`) (Week 7)
- [ ] Replace static UI containers with interactive full-screen Leaflet.js canvas initialized with CartoDB Positron base tiles[cite: 3].
- [ ] Build animated horizon time slider supporting historical playback ($t-30\text{m}$) and AI forecasts ($t+180\text{m}$)[cite: 3].
- [ ] Implement layer toggle control for Radar Reflectivity and Lightning Strike Density maps[cite: 3].
- [ ] Build dynamic top alert banner component bound to severe weather risk levels (Green, Yellow, Orange, Red)[cite: 3].
- [ ] Integrate Chart.js district risk gauge widgets and storm velocity vector indicators[cite: 3].

## Phase 5: Containerization, Testing & Deployment (Week 8)
- [ ] Write multi-stage Dockerfiles for `backend_api`, `nowcast_engine`, and `web_dashboard` microservices[cite: 3].
- [ ] Configure `docker-compose.yml` orchestrator with NVIDIA Container Toolkit runtime support[cite: 3].
- [ ] Perform latency profiling, tile load benchmarking, and GPU memory usage validation[cite: 3].
- [ ] Assemble offline demonstration evaluation dataset and prepare pitch demonstration walkthrough[cite: 3].