import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import WeatherMap from './components/WeatherMap';
import NowcastPanel from './components/NowcastPanel';
import HazardAlertBanner from './components/HazardAlertBanner';
import {
  fetchHealth,
  fetchNowcast,
  fetchActiveAlerts,
  fetchLightningStrokes,
} from './services/api';

export default function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [nowcastData, setNowcastData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [lightningStrokes, setLightningStrokes] = useState([]);
  const [selectedHorizon, setSelectedHorizon] = useState(30);

  const [layers, setLayers] = useState({
    radar: true,
    lightning: true,
    cells: true,
    satellite: false,
  });

  useEffect(() => {
    // 1. Check Backend Health
    fetchHealth()
      .then((data) => {
        setBackendStatus(data.status === 'online' ? 'online' : 'mock_mode');
      })
      .catch(() => setBackendStatus('mock_mode'));

    // 2. Fetch Nowcast Predictions
    fetchNowcast().then((data) => setNowcastData(data));

    // 3. Fetch Alerts
    fetchActiveAlerts().then((data) => setAlerts(data));

    // 4. Fetch Lightning sensor points
    fetchLightningStrokes().then((data) => setLightningStrokes(data));
  }, []);

  return (
    <div className="app-container">
      <Navbar backendStatus={backendStatus} />

      <main className="dashboard-main">
        {/* Map View */}
        <WeatherMap
          layers={layers}
          nowcastData={nowcastData}
          lightningStrokes={lightningStrokes}
        />

        {/* Right Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          {alerts.length > 0 && (
            <div style={{ padding: '1rem 1.25rem 0 1.25rem' }}>
              <HazardAlertBanner alert={alerts[0]} />
            </div>
          )}
          <NowcastPanel
            nowcastData={nowcastData}
            selectedHorizon={selectedHorizon}
            setSelectedHorizon={setSelectedHorizon}
            layers={layers}
            setLayers={setLayers}
          />
        </div>
      </main>
    </div>
  );
}
