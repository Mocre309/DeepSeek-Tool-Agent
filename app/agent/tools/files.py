"""Workspace file tools — the only place the agent touches the local disk.

Two hard rules, enforced here rather than in the prompt:

1. **One root.** Every path is resolved against the configured workspace and
   refused if it escapes it — via ``..``, an absolute path, or a symlink that
   points outside. ``Path.resolve()`` normalises all three before the check.
2. **Writes need a human.** ``write_file`` is registered with
   ``HumanInTheLoopMiddleware``, so the graph pauses for approval; reads are
   cheap and stay unobstructed.

Sizes are capped on both directions so a prompt cannot exhaust memory or disk.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_LIST_ENTRIES = 200
_DEFAULT_READ_CHARS = 8000
_MAX_READ_CHARS = 60_000


class _WorkspaceError(ValueError):
    """A path or IO problem worth showing to the model verbatim."""


def workspace_from_config(config: RunnableConfig | None) -> Path:
    """The workspace for *this* run: a conversation override, else the default.

    The service puts the resolved path into ``configurable.workspace`` for every
    turn, which is how one conversation can be pinned to its own directory.
    """
    value = ((config or {}).get("configurable") or {}).get("workspace")
    raw = value if isinstance(value, str) and value.strip() else None
    return Path(raw) if raw else get_settings().workspace_dir


def workspace_root(config: RunnableConfig | None = None) -> Path:
    """The resolved sandbox root (created on demand)."""
    root = workspace_from_config(config)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def is_inside_workspace(root: Path, target: Path) -> bool:
    """Case-insensitive containment check that also covers Windows paths."""
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def resolve_in_workspace(raw: str, config: RunnableConfig | None = None) -> Path:
    """Map a user-supplied relative path into the workspace, or refuse."""
    root = workspace_root(config)
    text = (raw or ".").strip().replace("\\", "/")
    if not text:
        text = "."
    candidate = Path(text)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not is_inside_workspace(root, target):
        raise _WorkspaceError(
            f"路径越界：只允许访问工作区域内的文件（{root}）"
        )
    return target


def relative_label(path: Path, config: RunnableConfig | None = None) -> str:
    """Render a path relative to the workspace for messages and links."""
    root = workspace_root(config)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - resolve_in_workspace prevents this
        return path.name


def _human_size(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    if num < 1024**2:
        return f"{num / 1024:.1f} KB"
    return f"{num / 1024**2:.2f} MB"


def _looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


@tool
def list_files(config: RunnableConfig, path: str = ".") -> str:
    """List files and folders inside the workspace, with sizes and modified time.

    Use this first to see what is available before reading anything.

    Args:
        path: Directory relative to the workspace root; ``"."`` is the root.
    """
    try:
        target = resolve_in_workspace(path, config)
    except _WorkspaceError as exc:
        return f"错误: {exc}"
    if not target.exists():
        return f"错误: 目录不存在: {relative_label(target, config) or path}"
    if not target.is_dir():
        return f"错误: 这不是目录: {relative_label(target, config)}"

    entries = sorted(
        target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
    )
    if not entries:
        return f"工作区域 {relative_label(target, config) or '.'} 是空的。"

    lines = [f"目录: {relative_label(target, config) or '.'}", f"条目数: {len(entries)}", ""]
    for item in entries[:_MAX_LIST_ENTRIES]:
        try:
            stat = item.stat()
            stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size = "<dir>" if item.is_dir() else _human_size(stat.st_size)
        except OSError:
            stamp, size = "?", "?"
        suffix = "/" if item.is_dir() else ""
        lines.append(f"{item.name}{suffix}\t{size}\t{stamp}")
    if len(entries) > _MAX_LIST_ENTRIES:
        lines.append(f"…（只显示前 {_MAX_LIST_ENTRIES} 项）")
    return "\n".join(lines)


@tool
def read_file(config: RunnableConfig, path: str, max_chars: int = _DEFAULT_READ_CHARS) -> str:
    """Read a UTF-8 text file from the workspace.

    Handles source code, notes, CSV, JSON, logs and similar text. Refuses binary
    files and anything outside the workspace.

    Args:
        path: File path relative to the workspace root.
        max_chars: Maximum characters to return (default 8000, capped at 60000).
    """
    try:
        target = resolve_in_workspace(path, config)
    except _WorkspaceError as exc:
        return f"错误: {exc}"
    if not target.exists():
        return f"错误: 文件不存在: {relative_label(target, config)}"
    if target.is_dir():
        return f"错误: 这是一个目录，请用 list_files: {relative_label(target, config)}"

    settings = get_settings()
    try:
        size = target.stat().st_size
        if size > settings.workspace_max_read_bytes:
            return (
                f"错误: 文件过大（{_human_size(size)}），"
                f"上限 {_human_size(settings.workspace_max_read_bytes)}"
            )
        with target.open("rb") as handle:
            raw = handle.read()
    except OSError as exc:
        logger.warning("read_file failed: %s", exc)
        return "错误: 读取失败"

    if _looks_binary(raw[:4096]):
        return f"错误: 这不是文本文件: {relative_label(target, config)}"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"错误: 文件不是 UTF-8 编码，无法作为文本读取: {relative_label(target, config)}"

    limit = max(1, min(int(max_chars or _DEFAULT_READ_CHARS), _MAX_READ_CHARS))
    truncated = len(text) > limit
    body = text[:limit]
    header = [
        f"文件: {relative_label(target, config)}",
        f"大小: {_human_size(size)} · 行数: {text.count(chr(10)) + 1}",
        "内容:" + ("（已截断）" if truncated else ""),
    ]
    return "\n".join(header) + "\n" + body


@tool
def write_file(config: RunnableConfig, path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file inside the workspace.

    The user must approve every write, so explain what you are about to save
    before calling this. Parent folders are created automatically.

    Args:
        path: File path relative to the workspace root, e.g. ``"notes/todo.md"``.
        content: Full file content to write (overwrites existing content).
    """
    settings = get_settings()
    payload = content if isinstance(content, str) else str(content)
    size = len(payload.encode("utf-8"))
    if size > settings.workspace_max_write_bytes:
        return (
            f"错误: 内容过大（{_human_size(size)}），"
            f"上限 {_human_size(settings.workspace_max_write_bytes)}"
        )

    try:
        target = resolve_in_workspace(path, config)
    except _WorkspaceError as exc:
        return f"错误: {exc}"
    if target.is_dir():
        return f"错误: 目标是一个目录: {relative_label(target, config)}"
    if target == workspace_root(config):
        return "错误: 需要一个文件路径，而不是工作区域根目录"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.is_file()
        target.write_text(payload, encoding="utf-8")
    except OSError as exc:
        logger.warning("write_file failed: %s", exc)
        return "错误: 写入失败（检查工作区域是否可写）"

    label = relative_label(target, config)
    logger.info("workspace write: %s (%d bytes, existed=%s)", label, size, existed)
    return "\n".join(
        [
            f"{'已更新' if existed else '已创建'}: {label}",
            f"大小: {_human_size(size)} · {payload.count(chr(10)) + 1} 行",
            f"查看/下载: /api/workspace/{label}",
        ]
    )
