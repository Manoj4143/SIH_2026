"""Database connection and session factory with PostGIS/SQLite dual-mode fallback."""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend_api.app.config import settings
from backend_api.app.db.models import Base

logger = logging.getLogger(__name__)

# Global engine and session maker references
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retrieves or creates the database engine, with automatic SQLite fallback."""
    global _engine, _session_maker
    if _engine is not None:
        return _engine

    target_url = settings.DATABASE_URL
    try:
        # First attempt target database URL (e.g., PostgreSQL with asyncpg)
        engine = create_async_engine(
            target_url,
            echo=settings.DEBUG,
            pool_pre_ping=True,
            connect_args={"timeout": 5} if "postgresql" in target_url else {},
        )
        _engine = engine
        _session_maker = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info(f"Initialized database engine: {target_url.split('@')[-1] if '@' in target_url else target_url}")
        return _engine
    except Exception as exc:
        logger.warning(f"Failed to initialize primary database ({target_url}): {exc}")
        if settings.USE_SQLITE_FALLBACK:
            fallback_url = settings.SQLITE_FALLBACK_URL
            logger.info(f"Falling back to SQLite database: {fallback_url}")
            engine = create_async_engine(
                fallback_url,
                echo=settings.DEBUG,
            )
            _engine = engine
            _session_maker = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            return _engine
        raise


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Retrieves the configured session maker."""
    global _session_maker
    if _session_maker is None:
        get_engine()
    assert _session_maker is not None
    return _session_maker


async def init_db() -> None:
    """Initializes tables and schemas on application startup."""
    global _engine
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            # If PostgreSQL, try creating postgis extension if permitted
            if "postgresql" in str(engine.url):
                try:
                    from sqlalchemy import text
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                except Exception as ext_err:
                    logger.warning(f"Could not enable PostGIS extension (may already exist or insufficient privs): {ext_err}")

            # Create all tables defined in Base
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas initialized successfully.")
    except Exception as err:
        logger.warning(f"Primary database table creation failed ({err}). Attempting SQLite fallback.")
        if settings.USE_SQLITE_FALLBACK and "postgresql" in str(engine.url):
            fallback_url = settings.SQLITE_FALLBACK_URL
            _engine = create_async_engine(fallback_url, echo=settings.DEBUG)
            _session_maker = async_sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("SQLite fallback schemas initialized successfully.")
        else:
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for injecting asynchronous database sessions."""
    maker = get_session_maker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
