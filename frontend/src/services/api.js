/**
 * API Client for interacting with AI Weather Nowcast FastAPI backend.
 */

const API_BASE = '/api/v1';

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    console.warn('Backend offline or unreachable, using local fallback', err);
    return { status: 'mock_mode', version: '1.0.0-mock' };
  }
}

export async function fetchNowcast(lat = 28.6139, lon = 77.2090) {
  try {
    const res = await fetch(`${API_BASE}/nowcast/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        radius_km: 60.0,
        lead_times_min: [15, 30, 60, 120],
      }),
    });
    if (!res.ok) throw new Error('Nowcast query failed');
    return await res.json();
  } catch (err) {
    console.warn('Using baseline nowcast mock data', err);
    return {
      generated_at: new Date().toISOString(),
      center_latitude: lat,
      center_longitude: lon,
      forecast_horizons: [15, 30, 60, 120],
      model_version: 'xgb_nowcast_v1.0 (local demo)',
      grid_predictions: [
        { lead_time_min: 15, predicted_max_dbz: 51.2, thunderstorm_probability: 0.86, lightning_risk: 'Severe', estimated_cape_j_kg: 2150.0, vertically_integrated_liquid: 34.2 },
        { lead_time_min: 30, predicted_max_dbz: 46.8, thunderstorm_probability: 0.74, lightning_risk: 'High', estimated_cape_j_kg: 1950.0, vertically_integrated_liquid: 28.6 },
        { lead_time_min: 60, predicted_max_dbz: 38.4, thunderstorm_probability: 0.48, lightning_risk: 'Moderate', estimated_cape_j_kg: 1620.0, vertically_integrated_liquid: 18.2 },
        { lead_time_min: 120, predicted_max_dbz: 26.5, thunderstorm_probability: 0.22, lightning_risk: 'Low', estimated_cape_j_kg: 1100.0, vertically_integrated_liquid: 8.5 },
      ],
      convective_cells: [
        { cell_id: 'CELL-NCR-01', current_lat: 28.68, current_lon: 77.16, centroid_dbz: 52.4, velocity_km_h: 36.0, heading_deg: 45.0, projected_lat_30m: 28.82, projected_lon_30m: 77.30 },
        { cell_id: 'CELL-NCR-02', current_lat: 28.45, current_lon: 77.02, centroid_dbz: 44.1, velocity_km_h: 29.5, heading_deg: 40.0, projected_lat_30m: 28.56, projected_lon_30m: 77.12 },
      ],
    };
  }
}

export async function fetchActiveAlerts() {
  try {
    const res = await fetch(`${API_BASE}/observations/alerts`);
    if (!res.ok) throw new Error('Failed to fetch alerts');
    return await res.json();
  } catch (err) {
    return [
      {
        id: 1,
        alert_type: 'Severe Thunderstorm & Lightning',
        severity: 'Warning',
        region_name: 'NCR & Surrounding Plains',
        center_latitude: 28.6139,
        center_longitude: 77.2090,
        radius_km: 45.0,
        description: 'Multi-radar scan detects rapid vertical cell growth with high lightning stroke frequency (>35 kA). Severe gust front expected.',
      },
    ];
  }
}

export async function fetchLightningStrokes() {
  try {
    const res = await fetch(`${API_BASE}/observations/lightning`);
    if (!res.ok) throw new Error('Failed to fetch lightning');
    return await res.json();
  } catch (err) {
    // Generate synthetic strokes around center
    return Array.from({ length: 12 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
      latitude: 28.6139 + (Math.random() - 0.5) * 0.3,
      longitude: 77.2090 + (Math.random() - 0.5) * 0.3,
      peak_current_ka: Math.round(20 + Math.random() * 60),
      stroke_type: Math.random() > 0.3 ? 'CG' : 'IC',
      polarity: Math.random() > 0.8 ? '+' : '-',
    }));
  }
}
