"""Database connection management."""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

# Upper bound on how long agent-generated SQL may run (milliseconds).
READONLY_STATEMENT_TIMEOUT_MS = 10_000


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=2)
def get_engine(readonly: bool = False):
    """Get the (cached) SQLAlchemy engine. Use readonly=True for agent SQL execution.

    Engines own a connection pool and are meant to be created once per process.
    Creating a new engine per query (the previous behaviour) opened a fresh pool
    every time and leaked connections.
    """
    settings = get_settings()
    if readonly:
        return create_engine(
            settings.database_url_readonly,
            echo=False,
            pool_pre_ping=True,
            # Server-side guard against runaway queries (e.g. huge cross joins).
            connect_args={"options": f"-c statement_timeout={READONLY_STATEMENT_TIMEOUT_MS}"},
        )
    return create_engine(settings.database_url, echo=False, pool_pre_ping=True)


@lru_cache(maxsize=2)
def _get_sessionmaker(readonly: bool = False):
    return sessionmaker(bind=get_engine(readonly=readonly))


def get_session(readonly: bool = False):
    """Get a new database session."""
    return _get_sessionmaker(readonly)()


def get_readonly_engine():
    """Get read-only engine for agent SQL execution."""
    return get_engine(readonly=True)
