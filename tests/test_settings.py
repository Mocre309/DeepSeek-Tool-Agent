"""Runtime settings API: masking, hot reload, tool toggles, path validation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.tools import TOOLS, resolve_tools
from app.core.config import get_settings
from app.core.settings_store import mask_key


class TestMasking:
    def test_mask_never_reveals_the_middle(self) -> None:
        assert mask_key("sk-1234567890abcdef") == "sk-…cdef"
        assert mask_key(None) == ""
        assert mask_key("short") == "…"

    def test_read_never_returns_the_raw_key(self, client: TestClient) -> None:
        view = client.get("/api/settings").json()
        assert "api_key" not in view
        assert set(view) >= {"model", "api_key_set", "enabled_tools", "downloads_dir", "tools"}
        assert view["api_key_hint"] == "" or "…" in view["api_key_hint"]


class TestPatch:
    def test_model_change_is_persisted_and_reloaded(self, client: TestClient) -> None:
        assert client.get("/api/settings").json()["model"] == "deepseek-flash"
        view = client.patch("/api/settings", json={"model": "deepseek-v4-pro"}).json()
        assert view["model"] == "deepseek-v4-pro"
        # persisted to data/settings.json and visible to a fresh get_settings()
        assert get_settings().deepseek_model == "deepseek-v4-pro"
        # the reloaded graph reports the new model through the settings view too
        assert client.get("/api/settings").json()["model"] == "deepseek-v4-pro"

    def test_api_key_is_stored_but_masked(self, client: TestClient) -> None:
        view = client.patch("/api/settings", json={"api_key": "sk-testkey0000111122223333"}).json()
        assert view["api_key_set"] is True
        assert view["api_key_hint"] == "sk-…3333"
        assert "sk-testkey" not in str(view)

    def test_blank_values_are_ignored(self, client: TestClient) -> None:
        before = client.get("/api/settings").json()
        after = client.patch("/api/settings", json={"api_key": "   ", "model": ""}).json()
        assert after["model"] == before["model"]
        assert after["api_key_hint"] == before["api_key_hint"]

    def test_empty_patch_is_a_no_op(self, client: TestClient) -> None:
        assert client.patch("/api/settings", json={}).status_code == 200


class TestToolToggles:
    def test_disabling_a_tool_updates_registry_and_graph(self, client: TestClient) -> None:
        enabled = [tool["name"] for tool in client.get("/api/tools").json() if tool["name"] != "web_search"]
        view = client.patch("/api/settings", json={"enabled_tools": enabled}).json()

        assert "web_search" in view["disabled_tools"]
        assert "web_search" not in view["enabled_tools"]

        tools = {tool["name"]: tool["enabled"] for tool in client.get("/api/tools").json()}
        assert tools["web_search"] is False
        assert tools["calculator"] is True

        # the resolver the graph uses agrees
        assert "web_search" not in {tool.name for tool in resolve_tools(get_settings().disabled_tools)}

    def test_unknown_tool_is_rejected(self, client: TestClient) -> None:
        response = client.patch("/api/settings", json={"enabled_tools": ["calculator", "rm_rf"]})
        assert response.status_code == 400
        assert "rm_rf" in response.json()["detail"]

    def test_all_tools_can_be_disabled(self, client: TestClient) -> None:
        view = client.patch("/api/settings", json={"enabled_tools": []}).json()
        assert view["enabled_tools"] == []
        assert len(view["disabled_tools"]) == len(TOOLS)


class TestDownloadDir:
    def test_valid_directory_is_accepted_and_created(self, client: TestClient, tmp_path) -> None:
        target = tmp_path / "my downloads"
        view = client.patch("/api/settings", json={"downloads_dir": str(target)}).json()
        assert view["downloads_dir"] == str(target.resolve())
        assert target.is_dir()

    def test_unwritable_path_is_rejected(self, client: TestClient, tmp_path) -> None:
        blocker = tmp_path / "a-file"
        blocker.write_text("x", encoding="utf-8")
        response = client.patch("/api/settings", json={"downloads_dir": str(blocker / "sub")})
        assert response.status_code == 400
        assert "不可写" in response.json()["detail"]


class TestWorkspaceDir:
    def test_workspace_is_reported_and_changeable(self, client: TestClient, tmp_path) -> None:
        assert "workspace_dir" in client.get("/api/settings").json()

        target = tmp_path / "my area"
        patched = client.patch("/api/settings", json={"workspace_dir": str(target)}).json()
        assert patched["workspace_dir"] == str(target.resolve())
        assert target.is_dir()

    def test_file_tools_follow_the_new_workspace(self, client: TestClient, tmp_path) -> None:
        target = tmp_path / "shifted"
        client.patch("/api/settings", json={"workspace_dir": str(target)})

        from app.agent.tools.files import workspace_root

        assert workspace_root() == target.resolve()

    def test_unwritable_workspace_is_rejected(self, client: TestClient, tmp_path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        response = client.patch("/api/settings", json={"workspace_dir": str(blocker / "ws")})
        assert response.status_code == 400


class TestReload:
    def test_conversations_survive_a_reload(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "重启后还在吗", "session_id": "settings-keep"})
        client.patch("/api/settings", json={"model": "deepseek-flash"})

        sessions = client.get("/api/sessions").json()
        assert any(item["session_id"] == "settings-keep" for item in sessions)
        history = client.get("/api/sessions/settings-keep/messages").json()["messages"]
        assert [message["role"] for message in history] == ["user", "assistant"]
