import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function HazardAlertBanner({ alert }) {
  if (!alert) return null;

  return (
    <div className="alert-card">
      <div className="alert-header">
        <span className="alert-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <AlertTriangle size={16} color="#f43f5e" />
          {alert.region_name}
        </span>
        <span className="alert-tag">{alert.severity}</span>
      </div>
      <p className="alert-desc">{alert.description}</p>
    </div>
  );
}
