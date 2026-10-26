"""Long-term memory tools backed by SQLite.

Conversation history (LangGraph checkpointer) is per-thread and short-lived;
these notes survive restarts and are shared across sessions, which is what
makes "记住我的偏好" actually work. A fresh connection per call keeps the tools
thread-safe under FastAPI's threadpool without a connection pool.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at DESC);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@tool
def save_note(content: str, tags: str = "") -> str:
    """Persist one durable fact about the user into long-term memory.

    Use this for stable preferences, profile details, decisions or anything the
    user asks you to remember across conversations. Do not save transient
    chatter, and never save secrets, tokens or passwords.

    Args:
        content: The single fact to remember, written as a self-contained
            sentence (e.g. ``"用户偏好在回答里先给结论，再给细节"``).
        tags: Optional comma-separated tags, e.g. ``"preference,writing"``.
    """
    text = (content or "").strip()
    if not text:
        return "错误: 内容为空"
    if len(text) > 2000:
        return "错误: 内容超过 2000 字符"
    tag_text = ",".join(t.strip() for t in (tags or "").split(",") if t.strip())

    path = get_settings().notes_db_path
    try:
        with _connect(path) as conn:
            cursor = conn.execute(
                "INSERT INTO notes (content, tags, created_at) VALUES (?, ?, ?)",
                (text, tag_text, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            note_id = cursor.lastrowid
    except sqlite3.Error as exc:
        logger.error("save_note failed: %s", exc)
        return f"错误: 写入记忆失败（{type(exc).__name__}）"

    return f"已保存到长期记忆（id={note_id}，tags={tag_text or '-'}）。"


@tool
def search_notes(query: str = "", limit: int = 5) -> str:
    """Recall previously saved long-term memories.

    Call this before answering questions about the user, their preferences or
    earlier decisions. An empty query returns the most recent notes.

    Args:
        query: Keyword(s) to match against note content and tags. Empty string
            means "give me the latest notes".
        limit: Maximum number of notes to return, 1-20.
    """
    path = get_settings().notes_db_path
    try:
        count = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        count = 5
    keyword = (query or "").strip()

    try:
        with _connect(path) as conn:
            if keyword:
                pattern = f"%{keyword}%"
                rows = conn.execute(
                    "SELECT id, content, tags, created_at FROM notes "
                    "WHERE content LIKE ? OR tags LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (pattern, pattern, count),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, content, tags, created_at FROM notes ORDER BY id DESC LIMIT ?",
                    (count,),
                ).fetchall()
    except sqlite3.Error as exc:
        logger.error("search_notes failed: %s", exc)
        return f"错误: 读取记忆失败（{type(exc).__name__}）"

    if not rows:
        scope = f"匹配 {keyword!r} 的" if keyword else ""
        return f"长期记忆里没有{scope}记录。"

    payload = [
        {
            "id": row["id"],
            "content": row["content"],
            "tags": row["tags"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
