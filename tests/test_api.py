"""HTTP contract tests against the real ASGI app (mock model, no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import get_settings
from tests.support import parse_sse


class TestMeta:
    def test_health_reports_ready_in_mock_mode(self, client: TestClient) -> None:
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert payload["ready"] is True
        assert payload["mock"] is True
        assert payload["tool_count"] == 10

    def test_tools_registry_is_listed(self, client: TestClient) -> None:
        tools = client.get("/api/tools").json()
        assert {tool["name"] for tool in tools} == {
            "calculator",
            "current_time",
            "web_search",
            "fetch_webpage",
            "download_file",
            "list_files",
            "read_file",
            "write_file",
            "save_note",
            "search_notes",
        }
        assert all(tool["label"] and tool["icon"] for tool in tools)

    def test_index_page_is_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "DeepSeek Tool Agent" in response.text

    def test_openapi_is_available(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "/api/chat/stream" in schema["paths"]


class TestChat:
    def test_chat_returns_answer_and_usage(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"message": "你好，介绍一下自己"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["error"] is None
        assert "你好，介绍一下自己" in payload["answer"]
        assert payload["usage"]["total_tokens"] >= 0

    def test_session_id_is_echoed(self, client: TestClient) -> None:
        payload = client.post(
            "/api/chat", json={"message": "hi", "session_id": "my-session_1"}
        ).json()
        assert payload["session_id"] == "my-session_1"

    def test_generated_session_id_when_omitted(self, client: TestClient) -> None:
        payload = client.post("/api/chat", json={"message": "hi"}).json()
        assert len(payload["session_id"]) == 32

    def test_empty_message_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"message": "   "}).status_code == 422

    def test_illegal_session_id_rejected(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"message": "hi", "session_id": "bad id!"})
        assert response.status_code == 422

    def test_oversized_message_rejected(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"message": "x" * 4001})
        assert response.status_code == 422


class TestStream:
    def test_sse_frame_sequence(self, client: TestClient) -> None:
        with client.stream(
            "POST", "/api/chat/stream", json={"message": "流式测试", "session_id": "sse-1"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["x-accel-buffering"] == "no"
            raw = "".join(response.iter_text())

        frames = parse_sse(raw)
        kinds = [frame["event"] for frame in frames]
        assert kinds[0] == "start"
        assert kinds[-1] == "done"
        assert "usage" in kinds

        tokens = "".join(f["data"]["text"] for f in frames if f["event"] == "token")
        assert "流式测试" in tokens

        usage = next(frame for frame in frames if frame["event"] == "usage")
        assert usage["data"]["tool_calls"] == 0
        assert usage["data"]["elapsed_ms"] >= 0

    def test_request_id_header(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert len(response.headers["x-request-id"]) == 12


class TestSessions:
    def test_history_and_reset_round_trip(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "记住我", "session_id": "hist-1"})

        history = client.get("/api/sessions/hist-1/messages").json()
        assert [message["role"] for message in history["messages"]] == ["user", "assistant"]

        reset = client.delete("/api/sessions/hist-1")
        assert reset.status_code == 204

        cleared = client.get("/api/sessions/hist-1/messages").json()
        assert cleared["messages"] == []

    def test_unknown_session_is_empty_not_404(self, client: TestClient) -> None:
        payload = client.get("/api/sessions/never-used/messages").json()
        assert payload["messages"] == []


class TestSessionList:
    def test_starts_empty(self, client: TestClient) -> None:
        assert client.get("/api/sessions").json() == []

    def test_lists_conversations_with_titles_and_turns(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "第一条消息当标题", "session_id": "list-a"})
        client.post("/api/chat", json={"message": "第二条不该改标题", "session_id": "list-a"})
        client.post("/api/chat", json={"message": "另一个会话", "session_id": "list-b"})

        sessions = client.get("/api/sessions").json()
        by_id = {session["session_id"]: session for session in sessions}

        assert set(by_id) == {"list-a", "list-b"}
        assert by_id["list-a"]["title"] == "第一条消息当标题"
        assert by_id["list-a"]["turns"] == 2
        assert by_id["list-b"]["turns"] == 1
        assert by_id["list-a"]["last_active"] >= by_id["list-a"]["created_at"]

    def test_delete_removes_it_from_the_list(self, client: TestClient) -> None:
        client.post("/api/chat", json={"message": "等一下就被删", "session_id": "list-del"})
        assert client.delete("/api/sessions/list-del").status_code == 204
        assert client.get("/api/sessions").json() == []


TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7q9JAAAAAElFTkSuQmCC"
)


class TestImages:
    def test_image_turn_round_trip(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat",
            json={"message": "这张图是什么？", "session_id": "img-1", "images": [TINY_PNG]},
        )
        assert response.status_code == 200

        history = client.get("/api/sessions/img-1/messages").json()
        assert history["messages"][0]["images"] == [TINY_PNG]

    def test_image_only_message_is_allowed(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"message": "", "session_id": "img-2", "images": [TINY_PNG]}
        )
        assert response.status_code == 200

    def test_empty_message_and_no_images_rejected(self, client: TestClient) -> None:
        assert client.post("/api/chat", json={"message": "  "}).status_code == 422

    def test_svg_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat",
            json={"message": "x", "images": ["data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="]},
        )
        assert response.status_code == 422

    def test_non_data_url_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"message": "x", "images": ["ftp://example.com/a.png"]}
        )
        assert response.status_code == 422

    def test_too_many_images_rejected(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"message": "x", "images": [TINY_PNG] * 5})
        assert response.status_code == 422

    def test_oversized_image_rejected(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("MAX_IMAGE_BYTES", "64")
        get_settings.cache_clear()
        big = "data:image/png;base64," + "A" * 400
        response = client.post("/api/chat", json={"message": "x", "images": [big]})
        assert response.status_code == 422

    def test_http_url_is_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat",
            json={
                "message": "远程图片",
                "session_id": "img-url",
                "images": ["https://example.com/a.png"],
            },
        )
        assert response.status_code == 200

    def test_health_reports_image_support(self, client: TestClient) -> None:
        assert client.get("/api/health").json()["supports_images"] is True

