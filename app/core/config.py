"""Application settings — single source of truth, loaded from env / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

#: Path settings that must never depend on the process working directory.
_PATH_FIELDS = ("conversation_db_path", "notes_db_path", "downloads_dir", "workspace_dir")


class Settings(BaseSettings):
    """All runtime knobs live here so nothing is hard-coded in business code."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- app -------------------------------------------------------------
    app_name: str = "DeepSeek Tool Agent"
    app_version: str = "0.1.0"
    environment: str = "dev"
    log_level: str = "INFO"
    #: Bind address / port used by ``python -m app`` (see app/__main__.py).
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)

    # ---- LLM -------------------------------------------------------------
    deepseek_api_key: SecretStr | None = None
    #: ``deepseek-flash`` is the native multimodal model (text + image input).
    #: Text-only alternatives: ``deepseek-v4-pro``, ``deepseek-chat``,
    #: ``deepseek-reasoner`` (the latter streams a reasoning trace).
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int | None = None
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    #: Serve a deterministic fake model instead of calling DeepSeek (tests / offline demo).
    mock_llm: bool = False

    # ---- agent loop guards ----------------------------------------------
    #: Hard ceiling on tool executions per single user turn.
    max_tool_calls_per_run: int = 8
    #: Hard ceiling on LLM calls per single user turn (tool loop can otherwise run away).
    max_model_calls_per_run: int = 12
    #: LangGraph super-step ceiling; defence in depth behind the two limits above.
    recursion_limit: int = 40

    # ---- tools -----------------------------------------------------------
    tool_timeout_seconds: float = 20.0
    web_max_results: int = 5
    #: ``auto`` tries duckduckgo then bing; or pin one: duckduckgo | bing.
    search_provider: str = "auto"
    fetch_max_chars: int = 4000
    fetch_max_bytes: int = 2_000_000
    notes_db_path: Path = BASE_DIR / "data" / "notes.sqlite3"
    #: Conversations live here: LangGraph checkpoints (messages + images) plus the
    #: session metadata that drives the sidebar.
    conversation_db_path: Path = BASE_DIR / "data" / "conversations.sqlite3"
    #: Files the agent downloads for the user, served back via /api/files/{name}.
    downloads_dir: Path = BASE_DIR / "data" / "downloads"
    download_max_bytes: int = 25_000_000
    download_timeout_seconds: float = 30.0
    #: The only directory the file tools may read or write. Configure in the UI.
    workspace_dir: Path = BASE_DIR / "workspace"
    workspace_max_read_bytes: int = 1_000_000
    workspace_max_write_bytes: int = 2_000_000

    # ---- sessions --------------------------------------------------------
    #: Minutes of inactivity before a conversation is *deleted* (metadata +
    #: messages). 0 disables the cleanup entirely — the default, because losing
    #: history two hours after writing it is never what a local agent should do.
    session_ttl_minutes: int = 0
    #: Upper bound on stored conversations; the oldest are evicted beyond this.
    max_sessions: int = 500

    # ---- multimodal ------------------------------------------------------
    max_images_per_message: int = 4
    max_image_bytes: int = 4_000_000

    # ---- tools -----------------------------------------------------------
    #: Tools the user switched off in the settings panel (graph is rebuilt).
    disabled_tools: list[str] = Field(default_factory=list)

    # ---- http ------------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _anchor_paths(self) -> Settings:
        """Resolve relative paths against the project root, not the CWD.

        ``data/settings.json`` stores project-relative paths so the whole folder
        can be copied to another machine or directory and still work.
        """
        for name in _PATH_FIELDS:
            value = getattr(self, name, None)
            if isinstance(value, Path) and not value.is_absolute():
                object.__setattr__(self, name, (BASE_DIR / value).resolve())
        return self

    @property
    def resolved_api_key(self) -> str | None:
        return self.deepseek_api_key.get_secret_value() if self.deepseek_api_key else None

    @property
    def has_api_key(self) -> bool:
        return bool(self.resolved_api_key)

    @property
    def can_use_llm(self) -> bool:
        """True when a real model call is possible."""
        return self.mock_llm or self.has_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Current settings: runtime overrides (data/settings.json) win over env/.env."""
    import json
    import logging
    import os

    overrides: dict[str, object] = {}
    configured = os.environ.get("SETTINGS_FILE")
    path = Path(configured) if configured else BASE_DIR / "data" / "settings.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                overrides = data
            else:
                logging.getLogger(__name__).warning(
                    "%s must contain a JSON object; ignoring it", path
                )
        except (OSError, ValueError) as exc:
            # Hand-edited files break easily (a missing comma is enough). Say so
            # instead of silently running on defaults.
            logging.getLogger(__name__).error(
                "ignoring %s: %s — fix the JSON, or change settings in the UI", path, exc
            )
            overrides = {}
    return Settings(**overrides)
