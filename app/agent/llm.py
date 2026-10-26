"""Model construction: real DeepSeek, or a deterministic mock for tests/offline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised at startup when no key is configured and mock mode is off."""


#: Models on the DeepSeek API whose ``input_modalities`` include ``image``.
#: Verified against ``GET https://api.deepseek.com/models``.
IMAGE_CAPABLE_MODELS = frozenset({"deepseek-flash", "deepseek-v4.1-flash"})


def model_supports_images(model: str) -> bool:
    """Best-effort capability check, used to warn the UI before an upload."""
    name = (model or "").strip().lower()
    return name in IMAGE_CAPABLE_MODELS or "vision" in name or "-vl" in name


class EchoChatModel(BaseChatModel):
    """Streaming stub used by tests and by ``MOCK_LLM=1``.

    It never calls the network, which lets the whole HTTP + SSE path be tested
    end-to-end in CI without a DeepSeek key.
    """

    script: list[str] = Field(default_factory=list)
    index: int = 0
    chunk_size: int = 3

    @property
    def _llm_type(self) -> str:
        return "echo-mock"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self

    def _next_text(self, messages: list[BaseMessage]) -> str:
        if self.script:
            reply = self.script[min(self.index, len(self.script) - 1)]
            self.index += 1
            return reply
        last_user = ""
        for message in reversed(messages):
            if message.type == "human":
                last_user = str(message.content)
                break
        return f"（mock 模式，未调用大模型）收到你的消息：{last_user}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._next_text(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        text = self._next_text(messages)
        size = max(1, self.chunk_size)
        for start in range(0, len(text), size):
            piece = text[start : start + size]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def build_model(settings: Settings) -> BaseChatModel:
    """Return the chat model the agent should run on."""
    if settings.mock_llm:
        logger.warning("MOCK_LLM is on — the agent will not call DeepSeek")
        return EchoChatModel()

    api_key = settings.resolved_api_key
    if not api_key:
        raise MissingAPIKeyError(
            "未找到 DEEPSEEK_API_KEY。请设置环境变量或写入 .env，"
            "或设置 MOCK_LLM=1 以离线模式启动。"
        )

    from langchain_deepseek import ChatDeepSeek

    logger.info("using DeepSeek model %s", settings.deepseek_model)
    return ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=api_key,
        api_base=settings.deepseek_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        # Streams token usage on the final chunk, so the UI can show cost.
        stream_usage=True,
    )
