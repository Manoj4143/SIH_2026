import React from 'react';
import {
  MapContainer,
  TileLayer,
  Circle,
  CircleMarker,
  Polyline,
  Popup,
  Marker,
} from 'react-leaflet';
import L from 'leaflet';

// Fix default leaflet icon issue in bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

export default function WeatherMap({
  center = [28.6139, 77.2090],
  zoom = 9,
  layers,
  nowcastData,
  lightningStrokes = [],
}) {
  return (
    <div className="map-canvas-container">
      <MapContainer
        center={center}
        zoom={zoom}
        className="leaflet-map"
        scrollWheelZoom={true}
      >
        {/* OpenStreetMap Base Layer */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Doppler Radar Station Origin */}
        <CircleMarker
          center={center}
          radius={7}
          pathOptions={{ color: '#38bdf8', fillColor: '#0284c7', fillOpacity: 0.9 }}
        >
          <Popup>
            <strong>Doppler Weather Radar (DWR-DEL)</strong>
            <br />
            Lat: {center[0]}, Lon: {center[1]}
            <br />
            Band: S-Band Polarimetric
          </Popup>
        </CircleMarker>

        {/* Doppler Radar Simulated Reflectivity Swaths */}
        {layers.radar && (
          <>
            <Circle
              center={[center[0] + 0.08, center[1] - 0.05]}
              radius={14000}
              pathOptions={{
                color: '#ff0000',
                fillColor: '#ff0000',
                fillOpacity: 0.35,
                weight: 1,
              }}
            >
              <Popup>
                <strong>Severe Convective Core</strong>
                <br />
                Reflectivity: 52.4 dBZ
                <br />
                VIL: 34.2 kg/m²
              </Popup>
            </Circle>
            <Circle
              center={[center[0] + 0.08, center[1] - 0.05]}
              radius={24000}
              pathOptions={{
                color: '#ffaa00',
                fillColor: '#ffff00',
                fillOpacity: 0.2,
                weight: 1,
              }}
            />
            <Circle
              center={[center[0] - 0.12, center[1] + 0.02]}
              radius={11000}
              pathOptions={{
                color: '#ff8800',
                fillColor: '#ff8800',
                fillOpacity: 0.3,
                weight: 1,
              }}
            >
              <Popup>
                <strong>Secondary Storm Cell</strong>
                <br />
                Reflectivity: 44.1 dBZ
              </Popup>
            </Circle>
          </>
        )}

        {/* Lightning Detection Network Strikes */}
        {layers.lightning &&
          lightningStrokes.map((stroke, index) => (
            <CircleMarker
              key={index}
              center={[stroke.latitude, stroke.longitude]}
              radius={stroke.peak_current_ka > 40 ? 6 : 4}
              pathOptions={{
                color: '#facc15',
                fillColor: '#fef08a',
                fillOpacity: 0.9,
                weight: 2,
              }}
            >
              <Popup>
                <strong>Lightning Strike ({stroke.stroke_type})</strong>
                <br />
                Peak Current: {stroke.peak_current_ka} kA ({stroke.polarity})
                <br />
                Time: {new Date(stroke.timestamp).toLocaleTimeString()}
              </Popup>
            </CircleMarker>
          ))}

        {/* Convective Cell Trajectory Vectors */}
        {layers.cells &&
          nowcastData?.convective_cells?.map((cell) => (
            <React.Fragment key={cell.cell_id}>
              <Polyline
                positions={[
                  [cell.current_lat, cell.current_lon],
                  [cell.projected_lat_30m, cell.projected_lon_30m],
                ]}
                pathOptions={{
                  color: '#38bdf8',
                  dashArray: '6, 6',
                  weight: 3,
                }}
              />
              <CircleMarker
                center={[cell.projected_lat_30m, cell.projected_lon_30m]}
                radius={5}
                pathOptions={{ color: '#38bdf8', fillColor: '#38bdf8', fillOpacity: 0.8 }}
              >
                <Popup>
                  <strong>{cell.cell_id} (Projected +30m)</strong>
                  <br />
                  Speed: {cell.velocity_km_h} km/h
                  <br />
                  Heading: {cell.heading_deg}°
                </Popup>
              </CircleMarker>
            </React.Fragment>
          ))}
      </MapContainer>

      {/* Radar dBZ Reflectivity Scale */}
      <div className="dbz-legend">
        <div className="legend-title">Radar Reflectivity (dBZ)</div>
        <div className="legend-gradient-bar" />
        <div className="legend-ticks">
          <span>0</span>
          <span>15</span>
          <span>30</span>
          <span>45</span>
          <span>55</span>
          <span>65+</span>
        </div>
      </div>
    </div>
  );
}
