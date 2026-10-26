"""Shared test helpers: a scripted chat model so the agent loop runs offline."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed script of ``AIMessage``s, streaming like a real model.

    Used to exercise the tool-calling loop (``model -> tools -> model``) without
    spending tokens or depending on the network.
    """

    script: list[AIMessage] = Field(default_factory=list)
    index: int = 0
    chunk_size: int = 4

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self

    def _next(self) -> AIMessage:
        if not self.script:
            return AIMessage(content="")
        message = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        message = self._next()
        if message.tool_calls:
            for index, call in enumerate(message.tool_calls):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"],
                                "args": json.dumps(call["args"]),
                                "id": call["id"],
                                "index": index,
                            }
                        ],
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=chunk)
                yield chunk
            return

        text = message.content if isinstance(message.content, str) else ""
        size = max(1, self.chunk_size)
        for start in range(0, len(text), size):
            piece = text[start : start + size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call_test") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def parse_sse(raw: str) -> list[dict[str, Any]]:
    """Turn a raw SSE body into ``[{event, data}]`` for assertions."""
    frames: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        event = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            frames.append({"event": event, "data": json.loads("\n".join(data_lines))})
    return frames
