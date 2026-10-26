"""Agent-loop tests: tool events, memory, isolation, failure handling."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from app.agent.service import (
    AgentEvent,
    AgentService,
    SessionStore,
    build_human_message,
    images_of,
)
from tests.support import ScriptedChatModel, tool_call_message


async def collect(service: AgentService, session_id: str, message: str) -> list[AgentEvent]:
    return [event async for event in service.stream(session_id=session_id, message=message)]


async def collect_with_images(
    service: AgentService, session_id: str, message: str, images: list[str]
) -> list[AgentEvent]:
    return [
        event
        async for event in service.stream(session_id=session_id, message=message, images=images)
    ]


@pytest.fixture
def service(settings, monkeypatch):
    model = ScriptedChatModel(
        script=[
            tool_call_message("calculator", {"expression": "2+3"}, call_id="call_1"),
            AIMessage(content="2+3 的结果是 5。"),
        ]
    )
    monkeypatch.setattr("app.agent.service.build_model", lambda _settings: model)
    return AgentService(settings)


class TestToolLoop:
    async def test_full_event_sequence(self, service: AgentService) -> None:
        events = await collect(service, "s1", "算一下 2+3")
        types = [event.type for event in events]

        assert types[0] == "start"
        assert types[-2:] == ["usage", "done"]
        assert types.index("tool_call") < types.index("tool_result")

        call = next(event for event in events if event.type == "tool_call")
        assert call.data["name"] == "calculator"
        assert call.data["label"] == "计算器"
        assert call.data["args"] == {"expression": "2+3"}

        result = next(event for event in events if event.type == "tool_result")
        assert result.data["content"].strip() == "5"
        assert result.data["status"] == "ok"
        assert isinstance(result.data["elapsed_ms"], int)

        answer = "".join(event.data["text"] for event in events if event.type == "token")
        assert "5" in answer

        usage = next(event for event in events if event.type == "usage")
        assert usage.data["tool_calls"] == 1
        assert usage.data["elapsed_ms"] >= 0

    async def test_run_aggregates_the_same_events(self, service: AgentService) -> None:
        payload = await service.run(session_id="s2", message="算一下 2+3")
        assert payload["error"] is None
        assert "5" in payload["answer"]
        assert any(event["type"] == "tool_call" for event in payload["events"])
        assert payload["usage"]["tool_calls"] == 1


class TestMemory:
    async def test_second_turn_sees_first_turn(self, service: AgentService) -> None:
        await collect(service, "m1", "第一句")
        first = await service.history("m1")
        assert [message["role"] for message in first] == ["user", "assistant"]

        await collect(service, "m1", "第二句")
        second = await service.history("m1")
        assert len(second) == 4
        assert second[2]["content"] == "第二句"

    async def test_sessions_do_not_leak(self, service: AgentService) -> None:
        await collect(service, "a", "只属于 a 的消息")
        assert await service.history("b") == []

    async def test_reset_clears_context(self, service: AgentService) -> None:
        await collect(service, "r1", "会被清掉")
        assert len(await service.history("r1")) == 2
        await service.reset("r1")
        assert await service.history("r1") == []


class TestFailureHandling:
    async def test_model_failure_becomes_an_error_event(self, settings, monkeypatch) -> None:
        class ExplodingModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "exploding"

            def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
                return self

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                raise RuntimeError("upstream exploded")

        monkeypatch.setattr(
            "app.agent.service.build_model", lambda _settings: ExplodingModel()
        )
        service = AgentService(settings)
        events = await collect(service, "boom", "hi")
        errors = [event for event in events if event.type == "error"]
        assert errors
        assert "RuntimeError" in errors[0].data["message"]
        # A failure must still terminate the stream cleanly.
        assert events[-1].type == "done"


class TestSessionStore:
    """Metadata lives in SQLite, so these also cover the persistence layer."""

    @staticmethod
    def make(tmp_path, ttl_seconds: int = 3600, max_sessions: int = 10) -> SessionStore:
        return SessionStore(
            tmp_path / "sessions.sqlite3",
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
        )

    def test_expires_after_ttl(self, tmp_path) -> None:
        """A positive TTL still prunes idle conversations (opt-in cleanup)."""
        store = self.make(tmp_path, ttl_seconds=60)
        store.touch("s1")
        con = sqlite3.connect(tmp_path / "sessions.sqlite3")
        con.execute("UPDATE sessions SET last_active = last_active - 3600")
        con.commit()
        con.close()
        assert store.pop_expired() == ["s1"]
        assert len(store) == 0

    def test_zero_ttl_never_expires(self, tmp_path) -> None:
        """TTL 0 is the default: history must survive idle time."""
        store = self.make(tmp_path, ttl_seconds=0)
        store.touch("old-session")
        # pretend it has been idle for a year
        con = sqlite3.connect(tmp_path / "sessions.sqlite3")
        con.execute("UPDATE sessions SET last_active = last_active - 31536000")
        con.commit()
        con.close()
        assert store.pop_expired() == []
        assert [record.session_id for record in store.summaries()] == ["old-session"]

    def test_evicts_oldest_beyond_capacity(self, tmp_path) -> None:
        store = self.make(tmp_path, max_sessions=2)
        for name in ("a", "b", "c"):
            store.touch(name)
            time.sleep(0.02)
        expired = store.pop_expired()
        assert len(store) == 2
        assert "a" in expired

    def test_title_comes_from_the_first_message_only(self, tmp_path) -> None:
        store = self.make(tmp_path)
        store.touch("s1", title_hint="  多  空格\n换行 ")
        store.touch("s1", title_hint="第二次不应覆盖标题")
        store.touch("s2")

        records = {record.session_id: record for record in store.summaries()}
        assert records["s1"].title == "多 空格 换行"
        assert records["s1"].turns == 2
        assert records["s2"].title == "新对话"
        assert records["s2"].turns == 1

    def test_long_title_is_truncated(self, tmp_path) -> None:
        store = self.make(tmp_path)
        store.touch("s1", title_hint="一" * 40)
        title = store.summaries()[0].title
        assert title.endswith("…")
        assert len(title) == 29

    def test_most_recent_first(self, tmp_path) -> None:
        store = self.make(tmp_path)
        store.touch("old")
        time.sleep(0.05)  # Windows wall-clock granularity is ~15ms
        store.touch("new")
        assert [record.session_id for record in store.summaries()] == ["new", "old"]

    def test_forget_drops_the_conversation(self, tmp_path) -> None:
        store = self.make(tmp_path)
        store.touch("s1")
        store.forget("s1")
        assert store.summaries() == []

    def test_survives_a_new_instance(self, tmp_path) -> None:
        first = self.make(tmp_path)
        first.touch("kept", title_hint="标题要留下来")
        second = self.make(tmp_path)
        records = second.summaries()
        assert [record.session_id for record in records] == ["kept"]
        assert records[0].title == "标题要留下来"
        assert records[0].turns == 1


class TestMultimodal:
    IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+7q9JAAAAAElFTkSuQmCC"

    async def test_image_turn_is_recorded_with_images(
        self, service: AgentService
    ) -> None:
        events = await collect_with_images(service, "img-1", "这图上是什么？", [self.IMAGE])
        start = next(event for event in events if event.type == "start")
        assert start.data["images"] == 1

        history = await service.history("img-1")
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "这图上是什么？"
        assert history[0]["images"] == [self.IMAGE]

    async def test_image_only_turn_gets_a_placeholder_title(
        self, service: AgentService
    ) -> None:
        await collect_with_images(service, "img-2", "", [self.IMAGE])
        titles = {record.session_id: record.title for record in service.sessions()}
        assert titles["img-2"] == "图片对话"

    def test_text_turns_carry_no_images(self) -> None:
        message = build_human_message("只有文字")
        assert message.content == "只有文字"
        assert images_of(message.content) == []

    def test_blocks_are_built_in_order(self) -> None:
        message = build_human_message("看图", [TestMultimodal.IMAGE])
        assert isinstance(message.content, list)
        assert message.content[0] == {"type": "text", "text": "看图"}
        assert message.content[1]["type"] == "image_url"
        assert message.content[1]["image_url"]["url"] == self.IMAGE


class TestEventWireFormat:
    def test_sse_frame_is_parseable(self) -> None:
        frame = AgentEvent("token", {"text": "你好"}).as_sse()
        assert frame.startswith("event: token\ndata: ")
        assert frame.endswith("\n\n")
        assert "你好" in frame and "\\u" not in frame  # ensure_ascii=False
