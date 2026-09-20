"""Unit Tests & Latency Benchmarks for Dynamic XYZ Tile Renderer."""

from __future__ import annotations

import io
import time
import numpy as np
from PIL import Image
import pytest

from backend_api.app.config import settings
from backend_api.app.services.tile_renderer import TileRenderer


class TestTileRenderer:
    """Test suite for TileRenderer coordinate math, PNG rendering, and performance."""

    @pytest.fixture
    def renderer(self) -> TileRenderer:
        return TileRenderer(tile_size=256)

    @pytest.fixture
    def sample_raster(self) -> np.ndarray:
        """Generates a synthetic 128x128 normalized radar raster."""
        h, w = 128, 128
        y, x = np.ogrid[:h, :w]
        # Convective storm centered at (64, 64) with 50 dBZ peak (normalized ~0.75)
        dist_sq = (x - 64) ** 2 + (y - 64) ** 2
        raster = np.exp(-dist_sq / 200.0).astype(np.float32) * 0.75
        return raster

    def test_tile_to_bbox_wgs84_world(self, renderer: TileRenderer) -> None:
        """Verifies full world tile bounding box at zoom 0."""
        min_lon, min_lat, max_lon, max_lat = renderer.tile_to_bbox_wgs84(0, 0, 0)
        assert min_lon == pytest.approx(-180.0, abs=1e-3)
        assert max_lon == pytest.approx(180.0, abs=1e-3)
        assert min_lat < max_lat
        assert -86.0 < min_lat < -85.0
        assert 85.0 < max_lat < 86.0

    def test_domain_intersection(self, renderer: TileRenderer) -> None:
        """Verifies spatial intersection checks for domain tiles vs faraway tiles."""
        domain = settings.domain_bbox  # (79.7, 12.5, 80.85, 13.65)

        # Tile (10, 740, 474) covers Chennai center
        chennai_tile_bbox = renderer.tile_to_bbox_wgs84(10, 740, 474)
        assert renderer.intersects_domain(chennai_tile_bbox, domain) is True

        # Tile (10, 0, 0) in North Atlantic
        far_tile_bbox = renderer.tile_to_bbox_wgs84(10, 0, 0)
        assert renderer.intersects_domain(far_tile_bbox, domain) is False

    def test_out_of_bounds_transparent_tile(self, renderer: TileRenderer, sample_raster: np.ndarray) -> None:
        """Verifies that out-of-bounds tiles return completely transparent PNGs immediately."""
        # Zoom 10, tile (0, 0) is nowhere near India
        png_bytes = renderer.render_tile(
            raster_2d=sample_raster,
            z=10,
            x=0,
            y=0,
            channel=0,
        )

        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (256, 256)
        assert img.mode == "RGBA"

        # Check all pixels have alpha == 0
        arr = np.array(img)
        assert np.all(arr[:, :, 3] == 0)

    def test_intersecting_tile_radar_png(self, renderer: TileRenderer, sample_raster: np.ndarray) -> None:
        """Verifies PNG output structure and colormapping for an intersecting radar tile."""
        png_bytes = renderer.render_tile(
            raster_2d=sample_raster,
            z=10,
            x=740,
            y=474,
            channel=0,
            is_normalized=True,
        )

        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (256, 256)
        assert img.mode == "RGBA"

        arr = np.array(img)
        # Should have visible rendered storm pixels with non-zero alpha
        assert np.any(arr[:, :, 3] > 0)

    def test_lightning_tile_colormap(self, renderer: TileRenderer, sample_raster: np.ndarray) -> None:
        """Verifies lightning density tile rendering."""
        png_bytes = renderer.render_tile(
            raster_2d=sample_raster,
            z=10,
            x=740,
            y=474,
            channel=1,
            is_normalized=True,
        )

        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (256, 256)

    def test_rendering_latency_benchmark(self, renderer: TileRenderer, sample_raster: np.ndarray) -> None:
        """Enforces the strict latency constraint: tile generation must be < 850 ms."""
        # Prime the cache with first call
        renderer.render_tile(sample_raster, z=10, x=740, y=474, channel=0)

        # Benchmark 5 consecutive tile renders
        latencies_ms = []
        for i in range(5):
            # Use different coordinates to bypass exact key cache
            t0 = time.perf_counter()
            _ = renderer.render_tile(
                raster_2d=sample_raster,
                z=10,
                x=740,
                y=474 + i,
                channel=0,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed)

        max_latency = max(latencies_ms)
        mean_latency = sum(latencies_ms) / len(latencies_ms)

        print(f"\n[Tile Benchmark] Mean Latency: {mean_latency:.2f} ms | Max Latency: {max_latency:.2f} ms")
        assert max_latency < settings.TILE_TARGET_LATENCY_MS, (
            f"Tile rendering latency {max_latency:.2f} ms exceeded ceiling of {settings.TILE_TARGET_LATENCY_MS} ms"
        )
