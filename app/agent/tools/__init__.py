"""Tool registry — the single place that decides what the agent can do."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool

from app.agent.tools.calculator import calculator
from app.agent.tools.clock import current_time
from app.agent.tools.download import download_file
from app.agent.tools.files import list_files, read_file, write_file
from app.agent.tools.notes import save_note, search_notes
from app.agent.tools.web import fetch_webpage, web_search

#: Ordered list handed to ``create_agent``.
TOOLS: list[BaseTool] = [
    calculator,
    current_time,
    web_search,
    fetch_webpage,
    download_file,
    list_files,
    read_file,
    write_file,
    save_note,
    search_notes,
]

#: Presentation metadata so the UI never has to hard-code tool names.
TOOL_LABELS: dict[str, dict[str, str]] = {
    "calculator": {"label": "计算器", "icon": "🧮", "hint": "精确算术"},
    "current_time": {"label": "时钟", "icon": "🕐", "hint": "当前日期与时区"},
    "web_search": {"label": "网页搜索", "icon": "🔍", "hint": "DuckDuckGo 免密钥"},
    "fetch_webpage": {"label": "网页正文", "icon": "📄", "hint": "抓取并提取正文"},
    "download_file": {"label": "下载文件", "icon": "⬇️", "hint": "需用户批准"},
    "list_files": {"label": "列出文件", "icon": "📂", "hint": "工作区域目录"},
    "read_file": {"label": "读文件", "icon": "📖", "hint": "工作区域内的文本"},
    "write_file": {"label": "写文件", "icon": "✍️", "hint": "需用户批准"},
    "save_note": {"label": "记入记忆", "icon": "📝", "hint": "SQLite 长期记忆"},
    "search_notes": {"label": "检索记忆", "icon": "🗂️", "hint": "跨会话回忆"},
}

#: Tools that pause the graph for explicit user approval before they run.
TOOLS_REQUIRING_APPROVAL: tuple[str, ...] = ("download_file", "write_file")

TOOL_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}


def resolve_tools(disabled: Sequence[str] = ()) -> list[BaseTool]:
    """The tools actually handed to the graph, minus whatever the user switched off."""
    blocked = set(disabled)
    return [tool for tool in TOOLS if tool.name not in blocked]


def describe_tools(disabled: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Serialize the registry for ``GET /api/tools`` (flagged, not filtered).

    The settings panel needs disabled tools too, so every entry carries
    ``enabled`` and the caller decides what to render.
    """
    blocked = set(disabled)
    described: list[dict[str, Any]] = []
    for tool in TOOLS:
        meta = TOOL_LABELS.get(tool.name, {})
        described.append(
            {
                "name": tool.name,
                "description": (tool.description or "").strip().splitlines()[0],
                "label": meta.get("label", tool.name),
                "icon": meta.get("icon", "🔧"),
                "hint": meta.get("hint", ""),
                "enabled": tool.name not in blocked,
                "requires_approval": tool.name in TOOLS_REQUIRING_APPROVAL,
            }
        )
    return described
