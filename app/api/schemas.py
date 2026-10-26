"""Request/response models — the API contract, validated at the edge."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

#: Only formats both DeepSeek's vision endpoint and browsers agree on.
ALLOWED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


def validate_image_ref(raw: str) -> str:
    """Accept a ``data:image/...;base64,...`` payload or an http(s) URL.

    Size is checked from the base64 length before decoding, so an oversized
    upload is rejected without allocating the bytes.
    """
    settings = get_settings()
    text = raw.strip()
    if not text:
        raise ValueError("图片内容为空")
    if text.startswith(("http://", "https://")):
        return text
    if not text.startswith("data:image/"):
        raise ValueError("图片必须是 data:image/...;base64 或 http(s) 链接")

    header, _, payload = text.partition(",")
    if not payload:
        raise ValueError("data URL 缺少 base64 内容")
    if ";base64" not in header:
        raise ValueError("data URL 必须使用 base64 编码")
    mime = header[len("data:") :].split(";")[0].strip().lower()
    if mime not in ALLOWED_IMAGE_MIMES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_MIMES))
        raise ValueError(f"不支持的图片格式 {mime}（支持 {allowed}）")

    approx_bytes = len(payload) * 3 // 4
    limit = settings.max_image_bytes
    if approx_bytes > limit:
        raise ValueError(f"单张图片不能超过 {limit // 1_000_000} MB")
    return text


class ChatRequest(BaseModel):
    """One user turn — text, images, or both."""

    message: str = Field(default="", max_length=4000, description="用户输入")
    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        pattern=SESSION_ID_PATTERN,
        description="会话标识；相同 id 的多次请求共享上下文",
    )
    images: list[str] = Field(
        default_factory=list,
        description="图片列表：data URL（base64）或 http(s) 链接；上限见 MAX_IMAGES_PER_MESSAGE",
    )

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("images")
    @classmethod
    def _check_images(cls, value: list[str]) -> list[str]:
        limit = get_settings().max_images_per_message
        if len(value) > limit:
            raise ValueError(f"单条消息最多 {limit} 张图片")
        return [validate_image_ref(item) for item in value]

    @model_validator(mode="after")
    def _not_empty(self) -> ChatRequest:
        if not self.message and not self.images:
            raise ValueError("message 和 images 不能同时为空")
        return self


class ChatResponse(BaseModel):
    """Non-streaming reply, aggregated from the same event stream."""

    session_id: str
    answer: str
    error: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    #: Present on user turns that carried images, so the UI can replay them.
    images: list[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class SessionSummary(BaseModel):
    """One row in the conversation list."""

    session_id: str
    title: str
    turns: int
    created_at: float = Field(description="Unix 时间戳")
    last_active: float = Field(description="Unix 时间戳")
    workspace_dir: str = Field(default="", description="该会话的独立工作区域（空=跟随默认）")
    group_name: str = Field(default="", description="所在分组（空=未分组）")


class SessionPatch(BaseModel):
    """Per-conversation fields; ``None`` means "leave unchanged"."""

    title: str | None = Field(default=None, max_length=60)
    group_name: str | None = Field(default=None, max_length=40)
    workspace_dir: str | None = Field(default=None, max_length=500)


class GroupInfo(BaseModel):
    name: str
    count: int = 0


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=40)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("分组名不能为空")
        return cleaned


class GroupRename(GroupCreate):
    """Same validation as creating: a rename is just a new name."""


class ArchivedGroupInfo(BaseModel):
    """A deleted group that can still be restored."""

    name: str
    count: int = 0
    deleted_at: float = 0.0


class GroupRestore(BaseModel):
    name: str = Field(min_length=1, max_length=40)


class UndoResponse(BaseModel):
    session_id: str
    removed: int
    messages: list[HistoryMessage] = Field(default_factory=list)


class RewriteRequest(BaseModel):
    """Replace the last turn with this content."""

    message: str = Field(min_length=1, max_length=4000)
    images: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message 不能为空")
        return cleaned

    @field_validator("images")
    @classmethod
    def _check_images(cls, value: list[str]) -> list[str]:
        limit = get_settings().max_images_per_message
        if len(value) > limit:
            raise ValueError(f"单条消息最多 {limit} 张图片")
        return [validate_image_ref(item) for item in value]


class ApprovalRequest(BaseModel):
    """The user's answer to a paused tool call."""

    approved: bool
    reason: str = Field(default="", max_length=500, description="拒绝时给模型的原因")


class PendingApproval(BaseModel):
    """A tool call the graph is holding until the user decides."""

    id: str
    tool: str
    label: str
    icon: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class PendingResponse(BaseModel):
    session_id: str
    pending: PendingApproval | None = None


class SettingsView(BaseModel):
    """What the settings panel renders (never includes the raw API key)."""

    model: str
    api_key_set: bool
    api_key_hint: str
    base_url: str
    downloads_dir: str
    workspace_dir: str
    enabled_tools: list[str]
    disabled_tools: list[str]
    supports_images: bool
    mock: bool
    available_models: list[str]
    tools: list[ToolInfo] = Field(default_factory=list)


class SettingsPatch(BaseModel):
    """Only the fields the panel may write; ``None`` means "leave unchanged"."""

    api_key: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=100)
    enabled_tools: list[str] | None = None
    downloads_dir: str | None = Field(default=None, max_length=500)
    workspace_dir: str | None = Field(default=None, max_length=500)

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @field_validator("enabled_tools")
    @classmethod
    def _dedupe(cls, value: list[str] | None) -> list[str] | None:
        return sorted(set(value)) if value is not None else value


class ToolInfo(BaseModel):
    name: str
    description: str
    label: str
    icon: str
    hint: str
    enabled: bool = True
    requires_approval: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    version: str
    model: str
    mock: bool
    tool_count: int
    active_sessions: int
    supports_images: bool = Field(description="当前模型是否接受图片输入")
    detail: str | None = None
