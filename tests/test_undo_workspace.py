"""Undo / rewrite of the last turn, and per-conversation workspaces."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.core.config import get_settings
from tests.support import ScriptedChatModel, parse_sse, tool_call_message


def stream_frames(client: TestClient, url: str, payload: dict) -> list[dict]:
    with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    return parse_sse(raw)


class TestUndo:
    def test_undo_removes_the_last_exchange(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "第一句", "session_id": "u1"})
        client.post("/api/chat", json={"message": "第二句", "session_id": "u1"})
        assert len(client.get("/api/sessions/u1/messages").json()["messages"]) == 4

        payload = client.post("/api/sessions/u1/undo").json()
        assert payload["removed"] >= 2
        roles = [message["role"] for message in payload["messages"]]
        assert roles == ["user", "assistant"]
        assert payload["messages"][0]["content"] == "第一句"

    def test_undo_on_empty_session_is_a_noop(self, client: TestClient) -> None:
        payload = client.post("/api/sessions/never-there/undo").json()
        assert payload["removed"] == 0
        assert payload["messages"] == []

    def test_undo_twice_empties_the_thread(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "只有一句", "session_id": "u2"})
        assert client.post("/api/sessions/u2/undo").json()["messages"] == []

    def test_turn_count_drops_with_the_undo(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "第一句", "session_id": "u3"})
        client.post("/api/chat", json={"message": "第二句", "session_id": "u3"})
        before = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "u3"
        )["turns"]
        client.post("/api/sessions/u3/undo")
        after = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "u3"
        )["turns"]
        assert after == before - 1

    def test_undo_is_blocked_while_an_approval_is_pending(self, gated_client: TestClient) -> None:
        stream_frames(
            gated_client, "/api/chat/stream", {"message": "写个文件", "session_id": "u4"}
        )
        response = gated_client.post("/api/sessions/u4/undo")
        assert response.status_code == 409
        assert "批准" in response.json()["detail"]


class TestRewrite:
    def test_rewrite_replaces_instead_of_appending(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "原来的问题", "session_id": "r1"})
        frames = stream_frames(
            client,
            "/api/sessions/r1/rewrite",
            {"message": "改成这个问题"},
        )
        kinds = [frame["event"] for frame in frames]
        assert kinds[0] == "truncated"
        assert "token" in kinds
        assert frames[-1]["event"] == "done"

        messages = client.get("/api/sessions/r1/messages").json()["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "改成这个问题"

    def test_rewrite_keeps_earlier_turns(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "第一轮", "session_id": "r2"})
        client.post("/api/chat", json={"message": "第二轮", "session_id": "r2"})
        stream_frames(client, "/api/sessions/r2/rewrite", {"message": "第二轮改"})

        contents = [
            message["content"]
            for message in client.get("/api/sessions/r2/messages").json()["messages"]
        ]
        assert contents[0] == "第一轮"
        assert contents[2] == "第二轮改"
        assert "第二轮" not in contents

    def test_rewrite_rejects_empty_message(self, client: TestClient) -> None:
        response = client.post("/api/sessions/r3/rewrite", json={"message": "   "})
        assert response.status_code == 422


@pytest.fixture
def gated_client(settings, monkeypatch, tmp_path):
    """A client whose model wants to write a file (so the turn pauses)."""
    model = ScriptedChatModel(
        script=[
            tool_call_message(
                "write_file", {"path": "a.md", "content": "x"}, call_id="call_w"
            ),
            AIMessage(content="写完了。"),
        ]
    )
    monkeypatch.setattr("app.agent.service.build_model", lambda _settings: model)
    from app.main import create_app

    with TestClient(create_app(settings, settings_factory=get_settings)) as test_client:
        yield test_client


class TestSessionWorkspace:
    def test_listing_reports_the_default(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "建个会话", "session_id": "ws1"})
        summary = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "ws1"
        )
        assert summary["workspace_dir"] in {"", None}

    def test_patch_pins_the_conversation(self, client: TestClient, tmp_path) -> None:
        target = tmp_path / "conv-ws"
        response = client.patch("/api/sessions/ws2", json={"workspace_dir": str(target)})
        assert response.status_code == 200
        assert response.json()["workspace_dir"] == str(target.resolve())
        assert target.is_dir()

        listed = next(
            item for item in client.get("/api/sessions").json() if item["session_id"] == "ws2"
        )
        assert listed["workspace_dir"] == str(target.resolve())

    def test_empty_string_restores_the_default(self, client: TestClient, settings) -> None:
        client.patch("/api/sessions/ws3", json={"workspace_dir": str(settings.workspace_dir)})
        response = client.patch("/api/sessions/ws3", json={"workspace_dir": ""})
        assert response.status_code == 200
        # the stored override is cleared, so the conversation follows the default
        assert response.json()["workspace_dir"] == ""

    def test_bad_path_is_rejected(self, client: TestClient, tmp_path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        response = client.patch(
            "/api/sessions/ws4", json={"workspace_dir": str(blocker / "inside")}
        )
        assert response.status_code == 400

    def test_writes_land_in_the_conversation_workspace(
        self, gated_client: TestClient, settings, tmp_path
    ) -> None:
        target = tmp_path / "pinned"
        gated_client.patch("/api/sessions/ws5", json={"workspace_dir": str(target)})

        stream_frames(gated_client, "/api/chat/stream", {"message": "写文件", "session_id": "ws5"})
        stream_frames(gated_client, "/api/sessions/ws5/resume", {"approved": True})

        assert (target / "a.md").is_file()
        assert not (settings.workspace_dir / "a.md").exists()
