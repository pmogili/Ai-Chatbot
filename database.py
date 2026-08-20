"""
database.py
------------
SQLAlchemy engine, session factory, and dependency helpers for FastAPI.
"""

from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

logger = logging.getLogger(__name__)

# `check_same_thread` is required for SQLite when used with FastAPI's
# threaded request handling.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def init_db() -> None:
    """Create all database tables if they do not already exist."""
    import models  # noqa: F401  (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", DATABASE_URL)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
