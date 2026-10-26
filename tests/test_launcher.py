"""The launcher: host/port come from the config, CLI flags win when given."""

from __future__ import annotations

import pytest

from app import __main__ as launcher
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch) -> None:
    """Isolate config so these tests never read the real .env / settings.json."""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.delenv("APP_PORT", raising=False)
    monkeypatch.delenv("APP_HOST", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestDefaults:
    def test_port_comes_from_settings(self) -> None:
        args = launcher.parse_args([])
        assert args.port == get_settings().app_port
        assert args.host == get_settings().app_host
        assert args.reload is False

    def test_env_port_is_used(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_PORT", "9137")
        monkeypatch.setenv("APP_HOST", "0.0.0.0")
        get_settings.cache_clear()
        args = launcher.parse_args([])
        assert args.port == 9137
        assert args.host == "0.0.0.0"

    def test_settings_json_can_set_the_port(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
        (tmp_path / "settings.json").write_text('{"app_port": 9241}', encoding="utf-8")
        get_settings.cache_clear()
        assert launcher.parse_args([]).port == 9241


class TestOverrides:
    def test_cli_flags_win(self) -> None:
        args = launcher.parse_args(["--port", "9500", "--host", "0.0.0.0", "--reload"])
        assert args.port == 9500
        assert args.host == "0.0.0.0"
        assert args.reload is True

    @pytest.mark.parametrize("port", ["0", "70000", "-1"])
    def test_out_of_range_port_is_rejected(self, port: str) -> None:
        with pytest.raises(SystemExit):
            launcher.parse_args(["--port", port])


class TestMain:
    def test_uvicorn_receives_the_configured_address(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_run(app, **kwargs):
            captured["app"] = app
            captured.update(kwargs)

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)
        monkeypatch.setenv("APP_PORT", "9233")
        get_settings.cache_clear()

        assert launcher.main([]) == 0
        assert captured["app"] == "app.main:app"
        assert captured["port"] == 9233
        assert captured["host"] == get_settings().app_host
        assert captured["reload"] is False
