"""Standalone Uvicorn launcher for SIH 26072 FastAPI Backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend_api.app.config import settings


def main() -> None:
    """Parses CLI arguments and launches the Uvicorn web server."""
    parser = argparse.ArgumentParser(description="Launch SIH 26072 Nowcasting Backend API")
    parser.add_argument("--host", type=str, default=settings.HOST, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Starting {settings.APP_NAME} on http://{args.host}:{args.port}")
    print(f"Interactive Swagger Docs: http://{args.host}:{args.port}{settings.API_V1_PREFIX}/docs")
    print("=" * 70)

    uvicorn.run(
        "backend_api.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
