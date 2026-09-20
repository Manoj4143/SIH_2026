"""Blitzortung & IITM Lightning Flash Ingestion Module.

Ingests discrete lightning strike coordinates (latitude, longitude, timestamp),
filters by geographic bounding box and 15-minute temporal window, and formats
point arrays for Gaussian Kernel Density Estimation (KDE).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from data_pipeline.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class LightningStrike:
    """Individual lightning strike record."""
    lat: float
    lon: float
    timestamp: datetime
    amplitude_ka: float = 0.0


@dataclass
class LightningBatch:
    """Collection of lightning strikes within a temporal analysis window."""
    strikes: List[LightningStrike]
    window_start: datetime
    window_end: datetime
    total_count: int
    is_fallback: bool
    metadata: Dict[str, Any]

    @property
    def coordinates(self) -> np.ndarray:
        """Returns [N, 2] array of (lat, lon) coordinates."""
        if not self.strikes:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[s.lat, s.lon] for s in self.strikes], dtype=np.float32)


class LightningIngestor:
    """Ingests raw Blitzortung and IITM lightning strike point feeds."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def ingest_from_csv(
        self,
        file_path: Union[str, Path],
        window_end: Optional[datetime] = None,
        window_minutes: int = 15,
    ) -> LightningBatch:
        """Reads Blitzortung CSV dump and extracts strikes within target time and domain bbox."""
        path = Path(file_path)
        end_time = window_end or datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=window_minutes)

        if not path.exists():
            logger.warning("Lightning CSV not found: %s. Returning empty batch.", path)
            return self.get_empty_batch(start_time, end_time, reason="file_not_found")

        strikes: List[LightningStrike] = []
        bbox = self.config.bbox

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Common CSV keys: lat, lon, time, amplitude
                    lat = float(row.get("lat") or row.get("latitude", 0.0))
                    lon = float(row.get("lon") or row.get("longitude", 0.0))
                    # Spatial filter
                    if not (bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon):
                        continue

                    # Parse timestamp if available
                    ts_raw = row.get("time") or row.get("timestamp")
                    try:
                        ts = datetime.fromisoformat(ts_raw) if ts_raw else end_time
                    except Exception:
                        ts = end_time

                    amp = float(row.get("amplitude") or row.get("peak_current", 0.0))
                    strikes.append(LightningStrike(lat=lat, lon=lon, timestamp=ts, amplitude_ka=amp))

            return LightningBatch(
                strikes=strikes,
                window_start=start_time,
                window_end=end_time,
                total_count=len(strikes),
                is_fallback=False,
                metadata={"source_file": str(path)},
            )
        except Exception as err:
            logger.error("Failed to parse lightning CSV %s: %s", path, err)
            return self.get_empty_batch(start_time, end_time, reason=str(err))

    def ingest_from_json(
        self,
        json_data: Union[str, List[Dict[str, Any]]],
        window_end: Optional[datetime] = None,
        window_minutes: int = 15,
    ) -> LightningBatch:
        """Parses Blitzortung JSON payloads."""
        end_time = window_end or datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=window_minutes)

        try:
            records = json.loads(json_data) if isinstance(json_data, str) else json_data
            strikes: List[LightningStrike] = []
            bbox = self.config.bbox

            for item in records:
                lat = float(item.get("lat", 0.0))
                lon = float(item.get("lon", 0.0))
                if bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon:
                    amp = float(item.get("amp", item.get("amplitude", 0.0)))
                    strikes.append(LightningStrike(lat=lat, lon=lon, timestamp=end_time, amplitude_ka=amp))

            return LightningBatch(
                strikes=strikes,
                window_start=start_time,
                window_end=end_time,
                total_count=len(strikes),
                is_fallback=False,
                metadata={"source": "json_stream"},
            )
        except Exception as err:
            logger.error("Error parsing lightning JSON: %s", err)
            return self.get_empty_batch(start_time, end_time, reason=str(err))

    def generate_synthetic_batch(
        self,
        window_end: Optional[datetime] = None,
        num_strikes: int = 140,
        cluster_center: Optional[Tuple[float, float]] = None,
        cluster_std_deg: float = 0.08,
    ) -> LightningBatch:
        """Generates realistic clustered lightning strikes around a convective core."""
        end_time = window_end or datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=self.config.time_interval_minutes)

        if cluster_center is None:
            c_lat, c_lon = self.config.bbox.center
        else:
            c_lat, c_lon = cluster_center

        rng = np.random.RandomState(42)
        lats = rng.normal(c_lat, cluster_std_deg, size=num_strikes)
        lons = rng.normal(c_lon, cluster_std_deg, size=num_strikes)
        amps = rng.exponential(scale=18.0, size=num_strikes) + 5.0

        bbox = self.config.bbox
        strikes: List[LightningStrike] = []
        for lat, lon, amp in zip(lats, lons, amps):
            if bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon:
                strikes.append(LightningStrike(lat=float(lat), lon=float(lon), timestamp=end_time, amplitude_ka=float(amp)))

        return LightningBatch(
            strikes=strikes,
            window_start=start_time,
            window_end=end_time,
            total_count=len(strikes),
            is_fallback=False,
            metadata={"synthetic": True},
        )

    def get_empty_batch(self, start_time: datetime, end_time: datetime, reason: str = "nominal_zero") -> LightningBatch:
        """Returns clean empty strike batch representing clear weather."""
        return LightningBatch(
            strikes=[],
            window_start=start_time,
            window_end=end_time,
            total_count=0,
            is_fallback=True if reason != "nominal_zero" else False,
            metadata={"reason": reason},
        )
