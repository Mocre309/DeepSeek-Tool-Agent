"""Runtime settings that the UI can change without a restart.

Precedence: ``data/settings.json`` (written by the settings panel) > environment
> ``.env``. The file is the only thing this module owns; everything else keeps
going through :class:`app.core.config.Settings`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Fields the settings panel is allowed to write, mapped to their config names.
EDITABLE_FIELDS: dict[str, str] = {
    "api_key": "deepseek_api_key",
    "model": "deepseek_model",
    "enabled_tools": "_enabled_tools",
    "downloads_dir": "downloads_dir",
    "workspace_dir": "workspace_dir",
}


def settings_file() -> Path:
    """Where overrides live; overridable so tests never touch the real file."""
    configured = os.environ.get("SETTINGS_FILE")
    return Path(configured) if configured else BASE_DIR / "data" / "settings.json"


def read_overrides() -> dict[str, Any]:
    path = settings_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_overrides(overrides: dict[str, Any]) -> None:
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def mask_key(raw: str | None) -> str:
    """``sk-abcdef...wxyz`` → ``sk-…wxyz``; never return the real key."""
    if not raw:
        return ""
    if len(raw) <= 8:
        return "…"
    return f"{raw[:3]}…{raw[-4:]}"


def apply_patch(patch: dict[str, Any]) -> Settings:
    """Persist a validated patch and return freshly loaded settings."""
    overrides = read_overrides()

    enabled = patch.get("enabled_tools")
    if enabled is not None:
        # Stored as the disabled list: a newly added tool is on by default.
        from app.agent.tools import TOOLS

        known = {tool.name for tool in TOOLS}
        unknown = sorted(set(enabled) - known)
        if unknown:
            raise ValueError(f"未知工具: {', '.join(unknown)}")
        overrides["disabled_tools"] = sorted(known - set(enabled))

    for field, config_name in (("api_key", "deepseek_api_key"), ("model", "deepseek_model")):
        value = patch.get(field)
        if isinstance(value, str) and value.strip():
            overrides[config_name] = value.strip()

    downloads = patch.get("downloads_dir")
    if isinstance(downloads, str) and downloads.strip():
        overrides["downloads_dir"] = store_dir(resolve_dir(downloads))

    workspace = patch.get("workspace_dir")
    if isinstance(workspace, str) and workspace.strip():
        overrides["workspace_dir"] = store_dir(resolve_dir(workspace))

    _write_overrides(overrides)
    get_settings.cache_clear()
    logger.info("runtime settings updated: %s", sorted(overrides))
    return get_settings()


def resolve_dir(raw: str) -> Path:
    """Accept an absolute path or one relative to the project root; create it."""
    candidate = Path(raw).expanduser()
    path = candidate if candidate.is_absolute() else BASE_DIR / candidate
    path = path.resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"目录不可写: {path}（{exc.strerror or exc}）") from exc
    return path


def store_dir(path: Path) -> str:
    """Persist project-internal paths as relative so the folder stays movable."""
    try:
        return path.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


# Backwards-compatible private alias (older call sites/tests used the underscore).
_store_dir = store_dir


def compact_paths() -> None:
    """One-time migration: rewrite project-internal absolute overrides as relative.

    Older versions wrote absolute paths (e.g. ``E:\\work\\agent\\workspace``),
    which break as soon as the folder is copied somewhere else.
    """
    overrides = read_overrides()
    changed = False
    for key in ("downloads_dir", "workspace_dir"):
        value = overrides.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            continue
        compacted = store_dir(resolved.resolve())
        if compacted != value:
            overrides[key] = compacted
            changed = True
    if changed:
        _write_overrides(overrides)
        get_settings.cache_clear()
        logger.info("rewrote path overrides as project-relative: %s", sorted(overrides))


def describe(settings: Settings) -> dict[str, Any]:
    """The settings panel's view of the current configuration."""
    from app.agent.llm import model_supports_images
    from app.agent.tools import TOOLS

    enabled = [tool.name for tool in TOOLS if tool.name not in set(settings.disabled_tools)]
    return {
        "model": settings.deepseek_model,
        "api_key_set": settings.has_api_key,
        "api_key_hint": mask_key(settings.resolved_api_key),
        "base_url": settings.deepseek_base_url,
        "downloads_dir": str(settings.downloads_dir),
        "workspace_dir": str(settings.workspace_dir),
        "enabled_tools": enabled,
        "disabled_tools": list(settings.disabled_tools),
        "supports_images": settings.mock_llm or model_supports_images(settings.deepseek_model),
        "mock": settings.mock_llm,
        "available_models": [
            "deepseek-flash",
            "deepseek-v4-pro",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    }
