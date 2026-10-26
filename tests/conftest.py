"""Test fixtures: isolated settings, an HTTP client, and offline model wiring."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> Iterator[None]:
    """Every test runs in mock mode with its own databases and downloads dir."""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("NOTES_DB_PATH", str(tmp_path / "notes.sqlite3"))
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("DOWNLOADS_DIR", str(tmp_path / "downloads"))
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("SESSION_TTL_MINUTES", "60")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(_isolated_env) -> Settings:
    return get_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    from app.main import create_app

    # settings_factory=get_settings so PATCH /api/settings actually hot-reloads,
    # exactly like the production wiring.
    app = create_app(settings, settings_factory=get_settings)
    with TestClient(app) as test_client:
        yield test_client
