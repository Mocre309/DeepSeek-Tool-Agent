"""Download tool — fetches a remote file to local disk and returns a link.

Two layers of protection, deliberately separate:

1. **The user decides.** This tool is registered with
   ``HumanInTheLoopMiddleware``, so the graph pauses before it runs and the UI
   has to approve the call. The tool never widens that decision.
2. **The tool still refuses dangerous targets.** Even an approved call cannot
   point at localhost or a private network, so approval cannot be farmed out as
   an SSRF primitive. Size and time are bounded for the same reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from langchain_core.tools import tool

from app.agent.tools.web import _ACCEPT_LANGUAGE, _USER_AGENT, blocked_target
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_GENERIC_NAME = "download.bin"


class _DownloadFailed(Exception):
    """A failure worth showing to the model verbatim."""


def _safe_filename(raw: str) -> str:
    """Reduce arbitrary input to one conservative, ASCII-only path component.

    ASCII-only is deliberate: the same name travels back out through
    ``GET /api/files/{name}``, whose pattern only allows ``[A-Za-z0-9._-]``.
    """
    candidate = raw.strip().replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(
        ch for ch in candidate if ch.isascii() and (ch.isalnum() or ch in "._- ")
    ).strip(" -")
    if not cleaned.strip(". "):
        return _GENERIC_NAME
    stem, dot, ext = cleaned.rpartition(".")
    if dot and 0 < len(ext) <= 8 and ext.isalnum():
        base = stem.strip(" .-")
        return f"{(base or 'download')[:60]}.{ext.lower()}"
    return cleaned.strip(". ")[:64] or _GENERIC_NAME


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    tail = unquote(path.rsplit("/", 1)[-1]) if path else ""
    return _safe_filename(tail or _GENERIC_NAME)


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.2f} MB"


def _download_to(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
    """Stream the response body to ``dest``; return ``(bytes_written, content_type)``."""
    written = 0
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT, "Accept-Language": _ACCEPT_LANGUAGE},
    ) as client, client.stream("GET", url) as response:
        if response.status_code >= 400:
            raise _DownloadFailed(f"服务器返回 HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise _DownloadFailed(f"文件超过 {max_bytes // 1_000_000} MB 上限")

        with dest.open("wb") as handle:
            for chunk in response.iter_bytes():
                written += len(chunk)
                if written > max_bytes:
                    raise _DownloadFailed(f"文件超过 {max_bytes // 1_000_000} MB 上限")
                handle.write(chunk)
    return written, content_type


@tool
def download_file(url: str, filename: str = "") -> str:
    """Download a file from the public web into the server's downloads folder.

    Use this when the user asks you to download / save / fetch a file (PDF,
    image, archive, dataset...). The call is paused for user approval, so say
    what you are about to fetch. Afterwards, give the user the download link the
    tool returns so they can grab the file.

    Args:
        url: Absolute http(s) URL of the file.
        filename: Optional name to save as; defaults to the last URL segment.
    """
    target = (url or "").strip()
    if not target:
        return "错误: URL 为空"
    if not target.startswith(("http://", "https://")):
        return "错误: 只支持 http/https 链接"

    refusal = blocked_target(target)
    if refusal:
        return f"错误: {refusal}"

    settings = get_settings()
    base_name = _safe_filename(filename) if filename else _filename_from_url(target)
    # Content-addressed suffix: same URL overwrites itself, different URLs never
    # collide, and the name stays a single safe path component.
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:8]
    stem, dot, ext = base_name.rpartition(".")
    unique = f"{stem}-{digest}.{ext}" if dot and ext else f"{base_name}-{digest}"
    dest = settings.downloads_dir / unique

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        written, content_type = _download_to(
            target, dest, settings.download_max_bytes, settings.download_timeout_seconds
        )
    except _DownloadFailed as exc:
        dest.unlink(missing_ok=True)
        return f"错误: {exc}"
    except httpx.TimeoutException:
        dest.unlink(missing_ok=True)
        return f"错误: 下载超时（> {settings.download_timeout_seconds:.0f}s）"
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        logger.warning("download_file failed: %s: %s", type(exc).__name__, exc)
        return f"错误: 下载失败（{type(exc).__name__}）"
    except OSError as exc:
        dest.unlink(missing_ok=True)
        logger.warning("download_file could not write: %s", exc)
        return "错误: 写入磁盘失败"

    if written == 0:
        dest.unlink(missing_ok=True)
        return "错误: 服务器返回了空文件"

    return "\n".join(
        [
            f"已保存: {unique}",
            f"大小: {_format_size(written)}",
            f"类型: {content_type or '未知'}",
            f"来源: {target}",
            f"下载链接: /api/files/{unique}",
        ]
    )
