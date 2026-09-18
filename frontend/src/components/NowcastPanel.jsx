import React from 'react';
import { Zap, Activity, ShieldAlert, Layers, Navigation } from 'lucide-react';

export default function NowcastPanel({
  nowcastData,
  selectedHorizon,
  setSelectedHorizon,
  layers,
  setLayers,
}) {
  const currentPrediction =
    nowcastData?.grid_predictions?.find((p) => p.lead_time_min === selectedHorizon) ||
    nowcastData?.grid_predictions?.[0];

  const getRiskClass = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'severe':
        return 'danger';
      case 'high':
        return 'danger';
      case 'moderate':
        return 'warning';
      default:
        return 'info';
    }
  };

  return (
    <div className="sidebar-panel">
      {/* Forecast Horizons */}
      <div className="section-card">
        <div className="card-title">
          <Activity size={16} color="#38bdf8" />
          <span>Nowcast Lead Time Horizon</span>
        </div>
        <div className="horizons-grid">
          {[15, 30, 60, 120].map((mins) => (
            <button
              key={mins}
              className={`horizon-btn ${selectedHorizon === mins ? 'active' : ''}`}
              onClick={() => setSelectedHorizon(mins)}
            >
              +{mins}m
            </button>
          ))}
        </div>

        {/* Dynamic Metric Display */}
        {currentPrediction && (
          <div className="metrics-row">
            <div className="metric-box">
              <div className="metric-label">Max Reflectivity</div>
              <div className="metric-value warning">
                {currentPrediction.predicted_max_dbz} <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>dBZ</span>
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Storm Probability</div>
              <div className={`metric-value ${getRiskClass(currentPrediction.lightning_risk)}`}>
                {Math.round(currentPrediction.thunderstorm_probability * 100)}%
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Lightning Risk</div>
              <div className={`metric-value ${getRiskClass(currentPrediction.lightning_risk)}`}>
                {currentPrediction.lightning_risk}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">CAPE Index</div>
              <div className="metric-value info">
                {currentPrediction.estimated_cape_j_kg}{' '}
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>J/kg</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Layer Toggles */}
      <div className="section-card">
        <div className="card-title">
          <Layers size={16} color="#38bdf8" />
          <span>Atmospheric Observation Layers</span>
        </div>

        <div className="layer-toggle">
          <div className="layer-info">
            <span>Doppler Radar Composite</span>
          </div>
          <div
            className={`toggle-switch ${layers.radar ? 'on' : ''}`}
            onClick={() => setLayers((prev) => ({ ...prev, radar: !prev.radar }))}
          />
        </div>

        <div className="layer-toggle">
          <div className="layer-info">
            <span>Lightning Strokes (LDN)</span>
          </div>
          <div
            className={`toggle-switch ${layers.lightning ? 'on' : ''}`}
            onClick={() => setLayers((prev) => ({ ...prev, lightning: !prev.lightning }))}
          />
        </div>

        <div className="layer-toggle">
          <div className="layer-info">
            <span>Convective Cell Vectors</span>
          </div>
          <div
            className={`toggle-switch ${layers.cells ? 'on' : ''}`}
            onClick={() => setLayers((prev) => ({ ...prev, cells: !prev.cells }))}
          />
        </div>

        <div className="layer-toggle">
          <div className="layer-info">
            <span>Satellite IR Brightness</span>
          </div>
          <div
            className={`toggle-switch ${layers.satellite ? 'on' : ''}`}
            onClick={() => setLayers((prev) => ({ ...prev, satellite: !prev.satellite }))}
          />
        </div>
      </div>

      {/* Convective Cell Tracking Vectors */}
      {nowcastData?.convective_cells && (
        <div className="section-card">
          <div className="card-title">
            <Navigation size={16} color="#38bdf8" />
            <span>Active Convective Cell Tracks</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {nowcastData.convective_cells.map((cell) => (
              <div
                key={cell.cell_id}
                style={{
                  padding: '8px 10px',
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: '6px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  fontSize: '0.8rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                  <span>{cell.cell_id}</span>
                  <span style={{ color: '#f59e0b' }}>{cell.centroid_dbz} dBZ</span>
                </div>
                <div style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '2px' }}>
                  Velocity: {cell.velocity_km_h} km/h • Heading: {cell.heading_deg}°
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
