"""Human-in-the-loop: the download tool must not run without approval."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent.tools import download
from app.core.config import get_settings
from tests.support import ScriptedChatModel, parse_sse, tool_call_message

URL = "https://example.com/report.pdf"
SESSION = "approval-1"


def fake_download(calls: list[str]):
    def _fake(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 fake")
        return 13, "application/pdf"

    return _fake


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def client(settings, monkeypatch, calls):
    """App whose model always tries to download the same file."""
    model = ScriptedChatModel(
        script=[
            tool_call_message("download_file", {"url": URL}, call_id="call_dl"),
            AIMessage(content="文件已经下载好了，链接在工具结果里。"),
            tool_call_message("download_file", {"url": URL}, call_id="call_dl2"),
            AIMessage(content="好的，不下载了。"),
        ]
    )
    monkeypatch.setattr("app.agent.service.build_model", lambda _settings: model)
    monkeypatch.setattr(download, "_download_to", fake_download(calls))

    from app.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client


def stream_frames(client: TestClient, url: str, payload: dict) -> list[dict]:
    with client.stream("POST", url, json=payload) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())
    return parse_sse(raw)


class TestApprovalGate:
    def test_download_pauses_instead_of_running(self, client: TestClient, calls) -> None:
        frames = stream_frames(client, "/api/chat/stream", {"message": "帮我下载报告", "session_id": SESSION})
        kinds = [frame["event"] for frame in frames]

        assert "approval_required" in kinds
        assert "tool_result" not in kinds  # the tool never ran

        request = next(frame for frame in frames if frame["event"] == "approval_required")
        assert request["data"]["tool"] == "download_file"
        assert request["data"]["args"] == {"url": URL}
        assert request["data"]["label"] == "下载文件"

        done = frames[-1]
        assert done["event"] == "done"
        assert done["data"]["awaiting_approval"] is True
        assert calls == []

    def test_pending_survives_a_page_refresh(self, client: TestClient) -> None:
        stream_frames(client, "/api/chat/stream", {"message": "帮我下载报告", "session_id": SESSION})
        pending = client.get(f"/api/sessions/{SESSION}/pending").json()["pending"]
        assert pending is not None
        assert pending["tool"] == "download_file"
        assert pending["args"] == {"url": URL}

    def test_approve_runs_the_tool_and_continues(
        self, client: TestClient, calls, settings
    ) -> None:
        stream_frames(client, "/api/chat/stream", {"message": "帮我下载报告", "session_id": SESSION})

        frames = stream_frames(
            client, f"/api/sessions/{SESSION}/resume", {"approved": True}
        )
        kinds = [frame["event"] for frame in frames]

        assert calls == [URL]  # ran only after approval
        assert "tool_result" in kinds
        result = next(frame for frame in frames if frame["event"] == "tool_result")
        assert result["data"]["status"] == "ok"
        assert "下载链接: /api/files/" in result["data"]["content"]

        assert frames[-1]["data"]["awaiting_approval"] is False
        assert client.get(f"/api/sessions/{SESSION}/pending").json()["pending"] is None

        name = result["data"]["content"].split("/api/files/")[1].split()[0]
        assert (settings.downloads_dir / name).is_file()

        # and the user can actually fetch it back
        served = client.get(f"/api/files/{name}")
        assert served.status_code == 200
        assert served.content == b"%PDF-1.4 fake"

    def test_reject_skips_the_tool(self, client: TestClient, calls) -> None:
        stream_frames(client, "/api/chat/stream", {"message": "帮我下载报告", "session_id": SESSION})

        frames = stream_frames(
            client,
            f"/api/sessions/{SESSION}/resume",
            {"approved": False, "reason": "这个文件不需要"},
        )
        assert calls == []  # the download never happened
        results = [frame for frame in frames if frame["event"] == "tool_result"]
        assert results and results[0]["data"]["status"] == "rejected"
        assert frames[-1]["data"]["awaiting_approval"] is False

        tool_messages = client.get(
            f"/api/sessions/{SESSION}/messages", params={"include_tools": True}
        ).json()["messages"]
        assert any(message["role"] == "tool" for message in tool_messages)

    def test_resume_without_a_pending_call_is_harmless(self, client: TestClient) -> None:
        frames = stream_frames(
            client, "/api/sessions/never-approved/resume", {"approved": True}
        )
        assert frames[-1]["event"] == "done"


class TestFilesEndpoint:
    def test_rejects_path_traversal(self, client: TestClient) -> None:
        assert client.get("/api/files/..%2F..%2Fetc%2Fpasswd").status_code in {400, 404}
        assert client.get("/api/files/notes.sqlite3").status_code == 404

    def test_missing_file_is_404(self, client: TestClient) -> None:
        assert client.get("/api/files/nope.pdf").status_code == 404


class TestWriteApproval:
    """The workspace write tool sits behind the same gate as the downloader."""

    @pytest.fixture
    def write_client(self, settings, monkeypatch):
        model = ScriptedChatModel(
            script=[
                tool_call_message(
                    "write_file",
                    {"path": "notes/a.md", "content": "# 标题\n正文"},
                    call_id="call_write",
                ),
                AIMessage(content="已经写好了。"),
            ]
        )
        monkeypatch.setattr("app.agent.service.build_model", lambda _settings: model)
        from app.main import create_app

        with TestClient(create_app(settings, settings_factory=get_settings)) as test_client:
            yield test_client

    def test_write_is_paused_until_approved(self, write_client: TestClient, settings) -> None:
        frames = stream_frames(
            write_client, "/api/chat/stream", {"message": "写个 md 文件", "session_id": "w1"}
        )
        assert "approval_required" in [frame["event"] for frame in frames]
        request = next(frame for frame in frames if frame["event"] == "approval_required")
        assert request["data"]["tool"] == "write_file"
        assert not (settings.workspace_dir / "notes" / "a.md").exists()

        frames = stream_frames(write_client, "/api/sessions/w1/resume", {"approved": True})
        target = settings.workspace_dir / "notes" / "a.md"
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == "# 标题\n正文"

        result = next(frame for frame in frames if frame["event"] == "tool_result")
        assert "/api/workspace/notes/a.md" in result["data"]["content"]

        served = write_client.get("/api/workspace/notes/a.md")
        assert served.status_code == 200

    def test_rejected_write_creates_nothing(self, write_client: TestClient, settings) -> None:
        stream_frames(
            write_client, "/api/chat/stream", {"message": "写个 md 文件", "session_id": "w2"}
        )
        stream_frames(write_client, "/api/sessions/w2/resume", {"approved": False})
        assert not (settings.workspace_dir / "notes" / "a.md").exists()
