# Atmospheric Data Store: AI Weather Nowcast

This directory stores observation streams and preprocessed feature tensors across multiple sensor modalities.

## Directory Structure

```
data/
├── raw/
│   ├── radar/         # Volumetric Doppler Weather Radar sweeps (.nc, .h5, .vol, .uf)
│   ├── satellite/     # Geostationary INSAT-3D/3DR / GOES NetCDF/HDF5 files
│   ├── lightning/     # High-resolution lightning detection network CSV/JSON streams
│   └── nwp_models/    # Numerical Weather Prediction GRIB2/NetCDF (WRF / GFS)
├── processed/         # Interpolated spatio-temporal grids (2x2 km) ready for training
└── sample/            # Small mock datasets for local quick development & CI
```

## Recommended Sensor Formats & Projections
- **Coordinate Reference System (CRS)**: WGS84 (`EPSG:4326`)
- **Radar**: Polar coordinates converted to Cartesian CAPPI / Pseudo-CAPPI using Py-ART / Wradlib.
- **Lightning**: Point coordinates with timestamp, peak current ($I_p$ in kA), and stroke classification.
- **Satellite**: Calibrated brightness temperatures for TIR1 (10.8 $\mu\text{m}$) and Water Vapor (6.8 $\mu\text{m}$).
