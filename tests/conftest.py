"""Pytest configuration and shared fixtures.

Tests marked `db` need the seeded PostgreSQL database (docker compose up -d &&
python seed.py). If the database is unreachable they are skipped, so the unit
tests still run on a machine without Docker.
"""

import sys
from functools import lru_cache
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "db: needs the seeded PostgreSQL database")


@lru_cache(maxsize=1)
def _database_status() -> str | None:
    """None if the seeded database is reachable, else the reason it is not."""
    try:
        from sqlalchemy import text

        from app.db.connection import get_engine

        with get_engine(readonly=True).connect() as conn:
            conn.execute(text("SELECT 1 FROM context.business_definitions LIMIT 1"))
        return None
    except Exception as e:  # connection refused, not seeded, ...
        return f"database not available ({type(e).__name__}); run docker compose up -d && python seed.py"


def pytest_collection_modifyitems(config, items):
    db_items = [item for item in items if "db" in item.keywords]
    if not db_items:
        return
    reason = _database_status()
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in db_items:
            item.add_marker(skip)
