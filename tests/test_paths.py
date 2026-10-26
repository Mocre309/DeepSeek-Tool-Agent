"""Every path the app persists should stay valid after the folder is moved."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import BASE_DIR, Settings, get_settings
from app.core.settings_store import apply_patch, store_dir


class TestSettingsDefaults:
    def test_relative_values_are_anchored_to_the_project(self) -> None:
        settings = Settings(workspace_dir="some-ws", downloads_dir="data/dl-x")
        assert settings.workspace_dir == (BASE_DIR / "some-ws").resolve()
        assert settings.downloads_dir == (BASE_DIR / "data/dl-x").resolve()

    def test_absolute_values_are_left_alone(self, tmp_path) -> None:
        outside = (tmp_path / "outside").resolve()
        settings = Settings(workspace_dir=str(outside))
        assert settings.workspace_dir == outside


class TestStoreDir:
    def test_project_internal_path_becomes_relative(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("app.core.settings_store.BASE_DIR", tmp_path)
        target = tmp_path / "downloads-x"
        target.mkdir()
        assert store_dir(target.resolve()) == "downloads-x"

    def test_nested_path_becomes_relative(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("app.core.settings_store.BASE_DIR", tmp_path)
        target = tmp_path / "data" / "downloads"
        target.mkdir(parents=True)
        assert store_dir(target.resolve()) == "data/downloads"

    def test_external_path_stays_absolute(self, tmp_path) -> None:
        outside = (tmp_path / "elsewhere").resolve()
        assert Path(store_dir(outside)).is_absolute()


class TestApplyPatch:
    def test_project_internal_dir_is_written_relative(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
        monkeypatch.setattr("app.core.settings_store.BASE_DIR", tmp_path)
        get_settings.cache_clear()

        apply_patch({"downloads_dir": str(tmp_path / "my-downloads")})
        stored = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        assert stored["downloads_dir"] == "my-downloads"
        get_settings.cache_clear()

    def test_broken_json_is_ignored_with_a_warning(self, tmp_path, monkeypatch, caplog) -> None:
        monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
        (tmp_path / "settings.json").write_text('{"deepseek_model": "x" "oops": 1}', encoding="utf-8")
        get_settings.cache_clear()
        with caplog.at_level("ERROR"):
            settings = get_settings()
        assert "ignoring" in caplog.text
        # it still boots, just on defaults
        assert settings.deepseek_model == "deepseek-flash"
        get_settings.cache_clear()


class TestNormalizeWorkspaces:
    def _store(self, tmp_path):
        from app.agent.service import SessionStore

        return SessionStore(
            tmp_path / "conversations.sqlite3", ttl_seconds=0, max_sessions=50
        )

    def test_in_project_absolute_rows_become_relative(self, tmp_path) -> None:
        base = tmp_path / "project"
        base.mkdir()
        store = self._store(tmp_path)

        inside = base / "workspace"
        inside.mkdir()
        store.set_workspace("s-inside", str(inside))
        store.set_workspace("s-relative", "workspace")
        store.set_workspace("s-outside", str((tmp_path / "elsewhere").resolve()))

        rewritten = store.normalize_workspaces(base)
        assert rewritten == 1
        assert store.workspace_of("s-inside") == "workspace"
        assert store.workspace_of("s-relative") == "workspace"
        assert Path(store.workspace_of("s-outside")).is_absolute()

    def test_it_is_idempotent(self, tmp_path) -> None:
        base = tmp_path / "project"
        (base / "ws").mkdir(parents=True)
        store = self._store(tmp_path)
        store.set_workspace("s1", str(base / "ws"))
        assert store.normalize_workspaces(base) == 1
        assert store.normalize_workspaces(base) == 0
