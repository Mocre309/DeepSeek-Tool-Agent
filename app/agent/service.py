"""Agent service — graph assembly, session lifecycle, and the streaming adapter.

The service is the only place that knows about LangChain. The HTTP layer only
sees ``AgentEvent`` objects, which keeps the transport (SSE, JSON, CLI, tests)
swappable and made the whole thing unit-testable with a fake model.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.agent.llm import build_model
from app.agent.prompts import build_system_prompt
from app.agent.tools import TOOL_LABELS, TOOLS_REQUIRING_APPROVAL, resolve_tools
from app.core import settings_store
from app.core.config import BASE_DIR, Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PendingApprovalError(RuntimeError):
    """A turn is paused waiting for the user; resolve it before mutating history."""

#: Tool output is trimmed for the UI; the model always sees the full result.
_TOOL_PREVIEW_CHARS = 600


@dataclass(slots=True)
class AgentEvent:
    """One normalized step of an agent turn."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_sse(self) -> str:
        import json

        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.type}\ndata: {payload}\n\n"


@dataclass(slots=True)
class _TurnStats:
    """Mutable bookkeeping shared between the event pump and the settlement."""

    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    awaiting: list[dict[str, Any]] = field(default_factory=list)


def _approval_payload(interrupt_item: Any, action: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``HumanInTheLoopMiddleware`` action request for the UI."""
    name = str(action.get("name") or "")
    meta = TOOL_LABELS.get(name, {})
    return {
        "id": str(getattr(interrupt_item, "id", "")),
        "tool": name,
        "label": meta.get("label", name),
        "icon": meta.get("icon", "🔧"),
        "args": action.get("args") or {},
        "description": action.get("description") or "",
    }


def _tool_status(message: ToolMessage) -> str:
    """``ok`` / ``error`` / ``rejected`` for the tool card in the UI.

    A rejected call still produces a ToolMessage (the model has to learn that it
    was refused), and the middleware marks it as ``status="error"`` — so the
    refusal text is checked first, before the generic error branch.
    """
    text = _text_of(message.content) or ""
    if text.startswith("User rejected the tool call"):
        return "rejected"
    if getattr(message, "status", "success") == "error":
        return "error"
    return "ok"


def _text_of(content: Any) -> str:
    """Flatten message content, which may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _messages_of(update: Any) -> list[BaseMessage]:
    if not isinstance(update, dict):
        return []
    raw = update.get("messages")
    if isinstance(raw, BaseMessage):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, BaseMessage)]
    return []


def images_of(content: Any) -> list[str]:
    """Pull image URLs back out of a multimodal message content list."""
    if not isinstance(content, list):
        return []
    urls: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in {"image_url", "image"}:
            continue
        value = block.get("image_url", block.get("image"))
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value:
            urls.append(value)
    return urls


def build_human_message(message: str, images: Sequence[str] = ()) -> HumanMessage:
    """Compose a turn as plain text, or as multimodal content blocks."""
    if not images:
        return HumanMessage(content=message)
    blocks: list[dict[str, Any]] = []
    if message:
        blocks.append({"type": "text", "text": message})
    blocks.extend(
        {"type": "image_url", "image_url": {"url": url}} for url in images
    )
    return HumanMessage(content=blocks)


def _preview(content: Any) -> str:
    text = _text_of(content) or ("" if content is None else str(content))
    text = text.strip()
    if len(text) <= _TOOL_PREVIEW_CHARS:
        return text
    return text[:_TOOL_PREVIEW_CHARS] + "…"


def _safe_error_message(exc: Exception, settings: Settings) -> str:
    """Map an exception onto something a user (and a demo audience) can act on."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "authentication" in lowered or "invalid api key" in lowered:
        return "DeepSeek 鉴权失败：请检查 DEEPSEEK_API_KEY 是否正确。"
    if "429" in text or "rate limit" in lowered:
        return "DeepSeek 触发了限流（429），请稍后重试。"
    if "image" in lowered and ("modality" in lowered or "support" in lowered):
        return (
            f"当前模型（{settings.deepseek_model}）不支持图片输入，"
            "请把 DEEPSEEK_MODEL 换成 deepseek-flash。"
        )
    if "timeout" in lowered:
        return f"模型调用超时（> {settings.llm_timeout_seconds:.0f}s），请重试或简化问题。"
    if "insufficient balance" in lowered:
        return "DeepSeek 账户余额不足。"
    logger.error("unhandled agent error [%s]: %s", name, text)
    return f"处理失败（{name}）。详情见服务端日志。"


@dataclass(slots=True)
class SessionRecord:
    """Metadata the conversation list in the UI needs (messages live elsewhere)."""

    session_id: str
    title: str
    turns: int
    created_at: float  # epoch seconds, wall clock — for display
    last_active: float
    workspace_dir: str = ""
    group_name: str = ""


@dataclass(slots=True)
class GroupRecord:
    """A user-created folder in the sidebar (groups may be empty)."""

    name: str
    count: int = 0


@dataclass(slots=True)
class ArchivedGroup:
    """A deleted group kept around so an accidental delete can be undone."""

    name: str
    count: int = 0
    deleted_at: float = 0.0


def _make_title(text: str, limit: int = 28) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"


def _clean_group(raw: str) -> str:
    return " ".join((raw or "").split())[:40]


_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    turns       INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    last_active REAL NOT NULL,
    workspace_dir TEXT NOT NULL DEFAULT '',
    group_name  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active DESC);
CREATE TABLE IF NOT EXISTS groups (
    name       TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS groups_archive (
    name        TEXT PRIMARY KEY,
    session_ids TEXT NOT NULL,
    deleted_at  REAL NOT NULL
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release, in place."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "workspace_dir" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN workspace_dir TEXT NOT NULL DEFAULT ''")
    if "group_name" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN group_name TEXT NOT NULL DEFAULT ''")


class SessionStore:
    """Conversation metadata in SQLite, so the sidebar survives a restart.

    The checkpointer owns the messages; this table owns just enough to render the
    conversation list (title, turns, recency) and to know which threads are
    candidates for pruning. Writes are single-row statements against a local
    file, so they stay synchronous, like the notes tool.
    """

    def __init__(self, db_path: Path, ttl_seconds: float, max_sessions: int) -> None:
        self._path = db_path
        self._ttl = ttl_seconds
        self._max = max(1, max_sessions)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SESSION_SCHEMA)
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def touch(self, session_id: str, *, title_hint: str | None = None) -> None:
        """Record activity; ``title_hint`` (the first user message) names the chat."""
        now = time.time()
        title = _make_title(title_hint) if title_hint else ""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, turns, created_at, last_active)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active = excluded.last_active,
                    turns = sessions.turns + 1,
                    title = CASE WHEN sessions.title = '' THEN excluded.title
                                 ELSE sessions.title END
                """,
                (session_id, title, now, now),
            )

    def forget(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def summaries(self) -> list[SessionRecord]:
        """Most recently active first — the order the sidebar renders."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, title, turns, created_at, last_active, "
                "workspace_dir, group_name FROM sessions ORDER BY last_active DESC"
            ).fetchall()
        return [
            SessionRecord(
                session_id=row["session_id"],
                title=row["title"] or "新对话",
                turns=row["turns"],
                created_at=row["created_at"],
                last_active=row["last_active"],
                workspace_dir=row["workspace_dir"] or "",
                group_name=row["group_name"] or "",
            )
            for row in rows
        ]

    def rename(self, session_id: str, title: str) -> bool:
        """Set a human title (creates the row for a not-yet-used conversation)."""
        cleaned = " ".join((title or "").split())[:60]
        if not cleaned:
            return False
        with self._lock, self._connect() as conn:
            updated = conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?", (cleaned, session_id)
            ).rowcount
            if not updated:
                now = time.time()
                conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, title, turns, created_at, last_active) VALUES (?, ?, 0, ?, ?)",
                    (session_id, cleaned, now, now),
                )
        return True

    def set_group(self, session_id: str, group: str) -> bool:
        """Move a conversation into a group ("" = ungrouped)."""
        name = _clean_group(group)
        with self._lock, self._connect() as conn:
            if name:
                conn.execute(
                    "INSERT OR IGNORE INTO groups (name, created_at) VALUES (?, ?)",
                    (name, time.time()),
                )
            updated = conn.execute(
                "UPDATE sessions SET group_name = ? WHERE session_id = ?", (name, session_id)
            ).rowcount
            if not updated:
                now = time.time()
                conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, title, turns, created_at, last_active, group_name) "
                    "VALUES (?, '', 0, ?, ?, ?)",
                    (session_id, now, now, name),
                )
        return True

    def group_of(self, session_id: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT group_name FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return (row["group_name"] or "") if row else ""

    def list_groups(self) -> list[GroupRecord]:
        """Groups with how many conversations sit in each, alphabetical."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT g.name AS name, COUNT(s.session_id) AS n "
                "FROM groups g LEFT JOIN sessions s ON s.group_name = g.name "
                "GROUP BY g.name ORDER BY g.name COLLATE NOCASE"
            ).fetchall()
        return [GroupRecord(name=row["name"], count=int(row["n"])) for row in rows]

    def create_group(self, name: str) -> bool:
        """Create an empty group; False when the name is taken or blank."""
        cleaned = _clean_group(name)
        if not cleaned:
            return False
        with self._lock, self._connect() as conn:
            inserted = conn.execute(
                "INSERT OR IGNORE INTO groups (name, created_at) VALUES (?, ?)",
                (cleaned, time.time()),
            ).rowcount
        return bool(inserted)

    def delete_group(self, name: str) -> None:
        """Archive the group, then free its conversations.

        Deleting used to be instant and irreversible, which made an accidental
        click (or a careless script) lose the whole grouping. The membership is
        snapshotted into ``groups_archive`` first, so ``restore_group`` can put
        everything back.
        """
        cleaned = _clean_group(name)
        if not cleaned:
            return
        with self._lock, self._connect() as conn:
            members = [
                row["session_id"]
                for row in conn.execute(
                    "SELECT session_id FROM sessions WHERE group_name = ?", (cleaned,)
                ).fetchall()
            ]
            previous = conn.execute(
                "SELECT session_ids FROM groups_archive WHERE name = ?", (cleaned,)
            ).fetchone()
            if previous is not None:
                try:
                    merged = json.loads(previous["session_ids"])
                except (TypeError, ValueError):
                    merged = []
                members = list(dict.fromkeys([*merged, *members]))
            conn.execute(
                "INSERT OR REPLACE INTO groups_archive (name, session_ids, deleted_at) "
                "VALUES (?, ?, ?)",
                (cleaned, json.dumps(members), time.time()),
            )
            conn.execute("DELETE FROM groups WHERE name = ?", (cleaned,))
            conn.execute(
                "UPDATE sessions SET group_name = '' WHERE group_name = ?", (cleaned,)
            )

    def list_archived(self) -> list[ArchivedGroup]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT name, session_ids, deleted_at FROM groups_archive "
                "ORDER BY deleted_at DESC"
            ).fetchall()
        archived: list[ArchivedGroup] = []
        for row in rows:
            try:
                count = len(json.loads(row["session_ids"]))
            except (TypeError, ValueError):
                count = 0
            archived.append(
                ArchivedGroup(name=row["name"], count=count, deleted_at=row["deleted_at"])
            )
        return archived

    def restore_group(self, name: str) -> int:
        """Recreate an archived group and reattach its surviving conversations."""
        cleaned = _clean_group(name)
        if not cleaned:
            raise ValueError("分组名不能为空")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT session_ids FROM groups_archive WHERE name = ?", (cleaned,)
            ).fetchone()
            if row is None:
                raise LookupError(f"没有可恢复的分组：{cleaned}")
            try:
                members = json.loads(row["session_ids"])
            except (TypeError, ValueError):
                members = []
            if conn.execute("SELECT 1 FROM groups WHERE name = ?", (cleaned,)).fetchone():
                raise ValueError(f"分组已存在：{cleaned}")

            conn.execute(
                "INSERT INTO groups (name, created_at) VALUES (?, ?)", (cleaned, time.time())
            )
            restored = 0
            for session_id in members:
                updated = conn.execute(
                    "UPDATE sessions SET group_name = ? WHERE session_id = ?",
                    (cleaned, session_id),
                ).rowcount
                restored += int(updated)
            conn.execute("DELETE FROM groups_archive WHERE name = ?", (cleaned,))
        return restored

    def rename_group(self, old: str, new: str) -> bool:
        """Rename a group and move its conversations with it.

        Returns False when ``old`` does not exist; raises ValueError when the new
        name is blank or already taken (merging groups silently is surprising).
        """
        source = _clean_group(old)
        target = _clean_group(new)
        if not source or not target:
            raise ValueError("分组名不能为空")
        with self._lock, self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM groups WHERE name = ?", (source,)
            ).fetchone()
            if not exists:
                return False
            if source == target:
                return True
            taken = conn.execute(
                "SELECT 1 FROM groups WHERE name = ?", (target,)
            ).fetchone()
            if taken:
                raise ValueError(f"分组已存在：{target}")
            conn.execute("UPDATE groups SET name = ? WHERE name = ?", (target, source))
            conn.execute(
                "UPDATE sessions SET group_name = ? WHERE group_name = ?", (target, source)
            )
        return True

    def workspace_of(self, session_id: str) -> str:
        """Per-conversation workspace override ("" means "use the default")."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_dir FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return (row["workspace_dir"] or "") if row else ""

    def set_workspace(self, session_id: str, workspace_dir: str) -> None:
        """Store (or clear, with "") this conversation's workspace."""
        with self._lock, self._connect() as conn:
            updated = conn.execute(
                "UPDATE sessions SET workspace_dir = ? WHERE session_id = ?",
                (workspace_dir, session_id),
            ).rowcount
            if not updated:
                now = time.time()
                conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, title, turns, created_at, last_active, workspace_dir) "
                    "VALUES (?, '', 0, ?, ?, ?)",
                    (session_id, now, now, workspace_dir),
                )

    def normalize_workspaces(self, base: Path) -> int:
        """Rewrite absolute in-project workspace overrides as relative paths.

        Keeps databases written by older versions movable together with the code.
        """
        root = base.resolve()
        changed = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, workspace_dir FROM sessions WHERE workspace_dir != ''"
            ).fetchall()
            for row in rows:
                path = Path(row["workspace_dir"])
                if not path.is_absolute():
                    continue
                try:
                    relative = path.resolve().relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                conn.execute(
                    "UPDATE sessions SET workspace_dir = ? WHERE session_id = ?",
                    (relative, row["session_id"]),
                )
                changed += 1
        if changed:
            logger.info("rewrote %d conversation workspace path(s) as project-relative", changed)
        return changed

    def decrement_turn(self, session_id: str) -> None:
        """Called after an undo, so the sidebar's turn count stays honest."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET turns = MAX(turns - 1, 0) WHERE session_id = ?",
                (session_id,),
            )

    def pop_expired(self) -> list[str]:
        """Evict conversations that aged out or overflowed; return their ids.

        A TTL of 0 means "never expire" — the default for a local tool. Only the
        capacity guard (``max_sessions``) still runs in that case.
        """
        cutoff = time.time() - self._ttl if self._ttl > 0 else None
        with self._lock, self._connect() as conn:
            stale: list[str] = []
            if cutoff is not None:
                stale = [
                    row["session_id"]
                    for row in conn.execute(
                        "SELECT session_id FROM sessions WHERE last_active < ?", (cutoff,)
                    ).fetchall()
                ]
            overflow = [
                row["session_id"]
                for row in conn.execute(
                    "SELECT session_id FROM sessions ORDER BY last_active DESC LIMIT -1 OFFSET ?",
                    (self._max,),
                ).fetchall()
            ]
            victims = list(dict.fromkeys(stale + overflow))
            if victims:
                conn.executemany(
                    "DELETE FROM sessions WHERE session_id = ?", [(item,) for item in victims]
                )
        return victims

    def __len__(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return int(row["n"])


class AgentService:
    """Owns the compiled graph and translates its stream into ``AgentEvent``s."""

    def __init__(self, settings: Settings | None = None, *, factory: Any = None) -> None:
        # ``factory`` is what makes the settings panel work: reload() re-reads it.
        if factory is not None:
            self._factory = factory
        elif settings is not None:
            self._factory = lambda: settings
        else:
            self._factory = get_settings
        self._settings = self._factory()
        self._sessions = SessionStore(
            self._settings.conversation_db_path,
            ttl_seconds=self._settings.session_ttl_minutes * 60,
            max_sessions=self._settings.max_sessions,
        )
        # Older databases stored absolute workspace paths; make them portable.
        self._sessions.normalize_workspaces(BASE_DIR)
        self._graph: Any = None
        self._checkpointer: Any = None
        self._conn: aiosqlite.Connection | None = None
        self._ready = asyncio.Lock()
        #: Turns currently streaming; reload() waits for them so a settings change
        #: cannot yank the SQLite connection out from under an in-flight reply.
        self._in_flight = 0
        self._drained = asyncio.Event()
        self._drained.set()

    async def reload(self) -> None:
        """Re-read settings and recompile (model / tool set / API key changed)."""
        await self._await_idle()
        async with self._ready:
            await self._close()
            self._settings = self._factory()
            self._sessions = SessionStore(
                self._settings.conversation_db_path,
                ttl_seconds=self._settings.session_ttl_minutes * 60,
                max_sessions=self._settings.max_sessions,
            )
        await self.start()
        logger.info(
            "agent reloaded: model=%s tools=%d",
            self._settings.deepseek_model,
            len(resolve_tools(self._settings.disabled_tools)),
        )

    async def _await_idle(self, timeout: float = 30.0) -> None:
        """Give running turns a chance to finish before the graph is swapped."""
        if self._in_flight <= 0:
            return
        logger.info("waiting for %d in-flight turn(s) before reload", self._in_flight)
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning("reloading with %d turn(s) still running", self._in_flight)

    async def _close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception as exc:
                logger.debug("closing checkpoint db failed: %s", exc)
        self._graph = None
        self._checkpointer = None
        self._conn = None

    async def start(self) -> None:
        """Open the checkpoint database and compile the graph.

        Idempotent, and deliberately async: ``AsyncSqliteSaver`` needs a live
        connection, so the graph cannot be built in ``__init__``.
        """
        async with self._ready:
            if self._graph is not None:
                return
            # Build the model first: a missing API key must not leak a connection.
            model = build_model(self._settings)
            db_path = self._settings.conversation_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(db_path)
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()
            self._conn = conn
            self._checkpointer = checkpointer
            self._graph = self._build_graph(model, checkpointer)

    async def _ensure_ready(self) -> Any:
        await self.start()
        return self._graph

    @property
    def ready(self) -> bool:
        return self._graph is not None

    # -- lifecycle ---------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    def sessions(self) -> list[SessionRecord]:
        """Conversation list for the sidebar, most recently active first."""
        return self._sessions.summaries()

    @property
    def graph(self) -> Any:
        """Compiled graph; raises if ``start()`` has not run yet."""
        if self._graph is None:
            raise RuntimeError("AgentService.start() has not completed")
        return self._graph

    def _build_graph(self, model: Any, checkpointer: Any) -> Any:
        tools = resolve_tools(self._settings.disabled_tools)
        graph = create_agent(
            model,
            tools=tools,
            system_prompt=build_system_prompt(self._settings.disabled_tools),
            middleware=[
                # Anything the user must sign off on pauses the graph here.
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        name: True
                        for name in TOOLS_REQUIRING_APPROVAL
                        if name not in set(self._settings.disabled_tools)
                    },
                    description_prefix="以下操作需要你确认",
                ),
                # Two independent runaway guards: tool executions, and LLM calls.
                ToolCallLimitMiddleware(
                    run_limit=self._settings.max_tool_calls_per_run,
                    exit_behavior="end",
                ),
                ModelCallLimitMiddleware(
                    run_limit=self._settings.max_model_calls_per_run,
                    exit_behavior="end",
                ),
            ],
            checkpointer=checkpointer,
            name="tool_agent",
        )
        logger.info(
            "agent graph ready: tools=%d mock=%s store=%s guards(tool=%d, model=%d)",
            len(tools),
            self._settings.mock_llm,
            self._settings.conversation_db_path.name,
            self._settings.max_tool_calls_per_run,
            self._settings.max_model_calls_per_run,
        )
        return graph

    def session_workspace(self, session_id: str) -> str:
        """The effective workspace for a conversation (override or global default).

        Overrides are stored project-relative when they live inside the project,
        so moving the folder does not break them.
        """
        override = self._sessions.workspace_of(session_id)
        if not override:
            return str(self._settings.workspace_dir)
        path = Path(override)
        return str(path if path.is_absolute() else (BASE_DIR / path).resolve())

    def set_session_workspace(self, session_id: str, raw: str | None) -> str:
        """Pin this conversation to its own workspace; ``None``/"" clears it.

        Returns the *stored override* ("" when the conversation follows the
        global default), which is what the settings panel needs to display.
        """
        text = (raw or "").strip()
        if not text:
            self._sessions.set_workspace(session_id, "")
            return ""
        resolved = settings_store.resolve_dir(text)
        stored = settings_store.store_dir(resolved)
        self._sessions.set_workspace(session_id, stored)
        return stored

    def rename_session(self, session_id: str, title: str) -> str:
        """Give a conversation a human title."""
        cleaned = " ".join((title or "").split())[:60]
        if not cleaned:
            raise ValueError("标题不能为空")
        self._sessions.rename(session_id, cleaned)
        return cleaned

    def set_session_group(self, session_id: str, group: str | None) -> str:
        """Move a conversation into a group ("" = ungrouped)."""
        name = " ".join((group or "").split())[:40]
        self._sessions.set_group(session_id, name)
        return name

    def groups(self) -> list[GroupRecord]:
        return self._sessions.list_groups()

    def create_group(self, name: str) -> GroupRecord:
        cleaned = " ".join((name or "").split())[:40]
        if not cleaned:
            raise ValueError("分组名不能为空")
        if not self._sessions.create_group(cleaned):
            raise FileExistsError(f"分组已存在：{cleaned}")
        return GroupRecord(name=cleaned, count=0)

    def delete_group(self, name: str) -> None:
        self._sessions.delete_group(name)

    def archived_groups(self) -> list[ArchivedGroup]:
        """Deleted groups that can still be restored."""
        return self._sessions.list_archived()

    def restore_group(self, name: str) -> GroupRecord:
        restored = self._sessions.restore_group(name)
        logger.info("restored group %s with %d conversation(s)", name, restored)
        return GroupRecord(name=" ".join((name or "").split())[:40], count=restored)

    def rename_group(self, old: str, new: str) -> GroupRecord:
        target = " ".join((new or "").split())[:40]
        if not target:
            raise ValueError("分组名不能为空")
        if not self._sessions.rename_group(old, target):
            raise LookupError(f"分组不存在：{old}")
        record = next(
            (group for group in self._sessions.list_groups() if group.name == target), None
        )
        return record or GroupRecord(name=target)

    def _config(self, session_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": session_id,
                # File tools read this instead of the global default, which is how
                # one conversation can be pinned to its own workspace.
                "workspace": self.session_workspace(session_id),
            },
            "recursion_limit": self._settings.recursion_limit,
        }

    async def _prune_expired(self) -> None:
        expired = self._sessions.pop_expired()
        if not expired:
            return
        deleter = getattr(self._checkpointer, "adelete_thread", None) if self._checkpointer else None
        for session_id in expired:
            try:
                if deleter is not None:
                    await deleter(session_id)
                else:  # pragma: no cover - depends on checkpointer implementation
                    self._checkpointer.delete_thread(session_id)
            except Exception as exc:
                logger.debug("failed to prune session %s: %s", session_id, exc)
        logger.info("pruned %d expired session(s)", len(expired))

    # -- streaming ---------------------------------------------------------
    async def stream(
        self,
        *,
        session_id: str,
        message: str,
        images: Sequence[str] = (),
    ) -> AsyncIterator[AgentEvent]:
        """Run one user turn, yielding normalized events as they happen."""
        await self._prune_expired()
        # A conversation is named after its first message; image-only turns get a
        # placeholder title instead of an empty row in the sidebar.
        title_hint = message or ("图片对话" if images else "")
        self._sessions.touch(session_id, title_hint=title_hint or None)

        started = time.perf_counter()
        stats = _TurnStats()

        yield AgentEvent(
            "start",
            {
                "session_id": session_id,
                "model": "mock" if self._settings.mock_llm else self._settings.deepseek_model,
                "mock": self._settings.mock_llm,
                "images": len(images),
            },
        )

        payload = {"messages": [build_human_message(message, images)]}
        async for event in self._run_turn(session_id, payload, stats):
            yield event
        async for event in self._finish(session_id, stats, started):
            yield event

    async def resume(
        self, *, session_id: str, approved: bool, reason: str = ""
    ) -> AsyncIterator[AgentEvent]:
        """Answer a paused approval request and continue the same turn."""
        self._sessions.touch(session_id)
        started = time.perf_counter()
        stats = _TurnStats()

        yield AgentEvent(
            "start",
            {
                "session_id": session_id,
                "model": "mock" if self._settings.mock_llm else self._settings.deepseek_model,
                "mock": self._settings.mock_llm,
                "resume": True,
                "approved": approved,
            },
        )

        decision = (
            {"type": "approve"}
            if approved
            else {"type": "reject", "message": reason or "用户拒绝了这次操作"}
        )
        payload = Command(resume={"decisions": [decision]})
        async for event in self._run_turn(session_id, payload, stats):
            yield event
        async for event in self._finish(session_id, stats, started):
            yield event

    async def pending_approval(self, session_id: str) -> dict[str, Any] | None:
        """Return the approval request a paused turn is waiting on, if any.

        Read from the checkpoint, so the prompt survives a page refresh.
        """
        graph = await self._ensure_ready()
        state = await graph.aget_state(self._config(session_id))
        for task in getattr(state, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                actions = (getattr(item, "value", None) or {}).get("action_requests", [])
                if not actions:
                    continue
                return _approval_payload(item, actions[0])
        return None

    # -- event pump --------------------------------------------------------
    async def _run_turn(
        self, session_id: str, payload: Any, stats: _TurnStats
    ) -> AsyncIterator[AgentEvent]:
        """Drive one graph invocation and normalize its stream into events."""
        pending: dict[str, dict[str, Any]] = {}
        self._in_flight += 1
        self._drained.clear()
        try:
            # start() failures become a friendly SSE error instead of a 500.
            graph = await self._ensure_ready()
            async for mode, chunk in graph.astream(
                payload,
                config=self._config(session_id),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message, metadata = chunk
                    if (metadata or {}).get("langgraph_node") != "model":
                        continue
                    text = _text_of(getattr(message, "content", ""))
                    if text:
                        yield AgentEvent("token", {"text": text})
                    reasoning = (getattr(message, "additional_kwargs", None) or {}).get(
                        "reasoning_content"
                    )
                    if reasoning:
                        yield AgentEvent("reasoning", {"text": str(reasoning)})
                    chunk_usage = getattr(message, "usage_metadata", None)
                    if chunk_usage:
                        stats.input_tokens = max(
                            stats.input_tokens, int(chunk_usage.get("input_tokens") or 0)
                        )
                        stats.output_tokens += int(chunk_usage.get("output_tokens") or 0)
                    continue

                # mode == "updates": nodes report their finished state.
                for node, update in (chunk or {}).items():
                    if node == "__interrupt__":
                        # HumanInTheLoopMiddleware paused the graph for approval.
                        for item in update or ():
                            requests = (getattr(item, "value", None) or {}).get(
                                "action_requests", []
                            )
                            for action in requests:
                                request = _approval_payload(item, action)
                                stats.awaiting.append(request)
                                yield AgentEvent("approval_required", request)
                        continue

                    for message in _messages_of(update):
                        if isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                call_id = str(call.get("id") or f"call_{stats.tool_calls}")
                                tool_name = str(call.get("name") or "unknown")
                                pending[call_id] = {
                                    "name": tool_name,
                                    "start": time.perf_counter(),
                                }
                                stats.tool_calls += 1
                                meta = TOOL_LABELS.get(tool_name, {})
                                yield AgentEvent(
                                    "tool_call",
                                    {
                                        "id": call_id,
                                        "name": tool_name,
                                        "label": meta.get("label", tool_name),
                                        "icon": meta.get("icon", "🔧"),
                                        "args": call.get("args") or {},
                                    },
                                )
                        elif isinstance(message, ToolMessage):
                            call_id = str(message.tool_call_id or "")
                            info = pending.pop(call_id, None)
                            elapsed_ms = (
                                round((time.perf_counter() - info["start"]) * 1000)
                                if info
                                else None
                            )
                            raw = _text_of(message.content) or str(message.content or "")
                            yield AgentEvent(
                                "tool_result",
                                {
                                    "id": call_id,
                                    "name": message.name or (info or {}).get("name") or "unknown",
                                    "status": _tool_status(message),
                                    "content": _preview(raw),
                                    "chars": len(raw),
                                    "elapsed_ms": elapsed_ms,
                                },
                            )
        except asyncio.CancelledError:
            logger.info("client disconnected, turn cancelled (session=%s)", session_id)
            raise
        except GraphRecursionError:
            logger.warning("recursion limit hit (session=%s)", session_id)
            yield AgentEvent(
                "error",
                {"message": "本轮循环次数超出上限，已中止。把问题拆成小步骤会更容易完成。"},
            )
        except Exception as exc:
            # Keep the traceback: an in-stream failure otherwise leaves no evidence.
            logger.exception("turn failed (session=%s)", session_id)
            yield AgentEvent("error", {"message": _safe_error_message(exc, self._settings)})
        finally:
            self._in_flight -= 1
            if self._in_flight <= 0:
                self._in_flight = 0
                self._drained.set()

    async def _finish(
        self, session_id: str, stats: _TurnStats, started: float
    ) -> AsyncIterator[AgentEvent]:
        """Emit the per-turn settlement (usage) and the terminal event."""
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        total = stats.input_tokens + stats.output_tokens
        awaiting = bool(stats.awaiting)
        yield AgentEvent(
            "usage",
            {
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "total_tokens": total,
                "tool_calls": stats.tool_calls,
                "elapsed_ms": elapsed_ms,
            },
        )
        yield AgentEvent(
            "done",
            {
                "session_id": session_id,
                "elapsed_ms": elapsed_ms,
                "awaiting_approval": awaiting,
            },
        )
        logger.info(
            "turn done session=%s tools=%d tokens=%d elapsed_ms=%d awaiting=%s",
            session_id,
            stats.tool_calls,
            total,
            elapsed_ms,
            awaiting,
        )

    # -- request/response --------------------------------------------------
    async def run(
        self, *, session_id: str, message: str, images: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Aggregate ``stream`` into a single response for the non-SSE endpoint."""
        chunks: list[str] = []
        events: list[dict[str, Any]] = []
        error: str | None = None
        usage: dict[str, Any] = {}

        async for event in self.stream(session_id=session_id, message=message, images=images):
            if event.type == "token":
                chunks.append(str(event.data.get("text", "")))
            elif event.type == "error":
                error = str(event.data.get("message", ""))
                events.append({"type": event.type, **event.data})
            elif event.type in {"tool_call", "tool_result", "reasoning"}:
                events.append({"type": event.type, **event.data})
            elif event.type == "usage":
                usage = event.data

        return {
            "session_id": session_id,
            "answer": "".join(chunks).strip(),
            "events": events,
            "usage": usage,
            "error": error,
        }

    # -- undo / rewrite ----------------------------------------------------
    async def undo(self, session_id: str) -> int:
        """Drop the last user turn (and everything after it); return how many went."""
        if await self.pending_approval(session_id) is not None:
            raise PendingApprovalError("这个会话还有待批准的操作，先批准或拒绝再撤回")
        graph = await self._ensure_ready()
        config = self._config(session_id)
        state = await graph.aget_state(config)
        messages: Sequence[BaseMessage] = (state.values or {}).get("messages", []) if state else []

        last_human = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if isinstance(messages[index], HumanMessage)),
            None,
        )
        if last_human is None:
            return 0
        removals = [
            RemoveMessage(id=message.id)
            for message in messages[last_human:]
            if getattr(message, "id", None)
        ]
        if not removals:
            return 0
        await graph.aupdate_state(config, {"messages": removals})
        self._sessions.decrement_turn(session_id)
        logger.info("undo session=%s removed=%d message(s)", session_id, len(removals))
        return len(removals)

    async def rewrite(
        self, *, session_id: str, message: str, images: Sequence[str] = ()
    ) -> AsyncIterator[AgentEvent]:
        """Replace the last user turn: undo it, then answer ``message`` instead."""
        try:
            removed = await self.undo(session_id)
        except PendingApprovalError as exc:
            yield AgentEvent("error", {"message": str(exc)})
            return
        yield AgentEvent("truncated", {"session_id": session_id, "removed": removed})
        async for event in self.stream(session_id=session_id, message=message, images=images):
            yield event

    # -- session management ------------------------------------------------
    async def history(self, session_id: str, *, include_tools: bool = False) -> list[dict[str, Any]]:
        """Replay a thread as chat messages.

        Tool messages are internal plumbing, so they are hidden unless a caller
        explicitly asks for them (useful when debugging a turn).
        """
        graph = await self._ensure_ready()
        state = await graph.aget_state(self._config(session_id))
        messages: Sequence[BaseMessage] = (state.values or {}).get("messages", []) if state else []
        result: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, ToolMessage):
                role = "tool"
            else:
                continue
            if role == "tool" and not include_tools:
                continue
            if role == "assistant" and not _text_of(message.content).strip():
                continue  # pure tool-call turns look empty in a chat UI
            result.append(
                {
                    "role": role,
                    "content": _text_of(message.content),
                    "images": images_of(message.content),
                }
            )
        return result

    async def reset(self, session_id: str) -> None:
        """Delete a conversation: checkpointed messages + the registry entry."""
        self._sessions.forget(session_id)
        if self._checkpointer is None:
            logger.info("session removed before it materialised: %s", session_id)
            return
        deleter = getattr(self._checkpointer, "adelete_thread", None)
        try:
            if deleter is not None:
                await deleter(session_id)
            else:  # pragma: no cover
                self._checkpointer.delete_thread(session_id)
        except Exception as exc:
            logger.debug("reset failed for %s: %s", session_id, exc)
        logger.info("session reset: %s", session_id)

    async def shutdown(self) -> None:
        """Release the graph and the SQLite connection."""
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception as exc:
                logger.debug("closing checkpoint db failed: %s", exc)
        self._graph = None
        self._checkpointer = None
        self._conn = None
