"""
tests/conftest.py
-------------------
Shared pytest fixtures: an isolated in-memory SQLite database and a
TestClient wired to use it instead of the real chatbot.db file.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402
from database import Base, get_db  # noqa: E402

# Use a shared in-memory SQLite DB for the whole test session so all
# connections see the same tables (StaticPool keeps a single connection).
from sqlalchemy.pool import StaticPool

TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables once for the test session."""
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    """A FastAPI TestClient wired to the isolated test database."""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def db_session():
    """A raw SQLAlchemy session against the isolated test database."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
