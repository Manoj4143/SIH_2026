import React from 'react';
import { CloudLightning, Radio, Activity } from 'lucide-react';

export default function Navbar({ backendStatus }) {
  return (
    <header className="navbar">
      <div className="brand-section">
        <div className="brand-logo">
          <CloudLightning size={22} />
        </div>
        <div>
          <span className="brand-title">AI Weather Nowcast</span>
        </div>
        <span className="brand-badge">AIML v1.0</span>
      </div>

      <div className="nav-actions">
        <div className="status-indicator">
          <span className="status-dot" style={{ backgroundColor: backendStatus === 'online' ? '#10b981' : '#f59e0b' }} />
          <span>Backend: {backendStatus === 'online' ? 'FastAPI Connected' : 'Simulated Feed'}</span>
        </div>
      </div>
    </header>
  );
}
