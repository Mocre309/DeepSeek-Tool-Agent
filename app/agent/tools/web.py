"""Web tools: keyless search (pluggable providers) + readable page extraction.

Both tools are deliberately defensive — network failures return a short,
LLM-readable error string instead of raising, so the agent can retry with a
different query rather than dying mid-loop.

Search goes through a small provider chain: DuckDuckGo (via ``ddgs``) first,
Bing HTML as the fallback, because a single upstream is never reliable and this
keeps the whole stack key-free. ``fetch_webpage`` also refuses private /
loopback targets (SSRF guard): a tool that fetches arbitrary URLs on behalf of
a prompt is otherwise an internal-network proxy.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ToolAgent/0.1"
)
_ACCEPT_LANGUAGE = "zh-CN,zh;q=0.9,en;q=0.8"
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localhost", ".home.arpa")
_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
    "button",
)
_TEXTUAL_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def _search_duckduckgo(query: str, limit: int) -> list[SearchHit]:
    from ddgs import DDGS

    rows = DDGS(timeout=6).text(query, max_results=limit)
    return [
        SearchHit(
            title=(row.get("title") or "").strip(),
            url=(row.get("href") or row.get("url") or "").strip(),
            snippet=" ".join((row.get("body") or row.get("snippet") or "").split()),
        )
        for row in rows
    ]


def _search_bing(query: str, limit: int) -> list[SearchHit]:
    response = httpx.get(
        "https://www.bing.com/search",
        params={"q": query, "count": str(limit)},
        timeout=15,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT, "Accept-Language": _ACCEPT_LANGUAGE},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    hits: list[SearchHit] = []
    for item in soup.select("li.b_algo"):
        link = item.select_one("h2 a")
        if link is None or not link.get("href"):
            continue
        snippet_node = item.select_one("p") or item.select_one(".b_caption")
        hits.append(
            SearchHit(
                title=link.get_text(" ", strip=True),
                url=str(link["href"]),
                snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
            )
        )
        if len(hits) >= limit:
            break
    return hits


_PROVIDERS = {
    "duckduckgo": lambda query, limit: _search_duckduckgo(query, limit),
    "bing": lambda query, limit: _search_bing(query, limit),
}
#: Tried in order when ``SEARCH_PROVIDER=auto``. Bing HTML answers in ~1-3s and
#: is reachable from networks where DuckDuckGo is filtered; ddgs stays as backup.
_AUTO_ORDER = ("bing", "duckduckgo")

#: A provider that just failed is skipped for a while, so one flaky upstream
#: does not add its timeout to every single search.
_COOLDOWN_SECONDS = 120.0
_provider_cooldown: dict[str, float] = {}


def _on_cooldown(name: str) -> bool:
    until = _provider_cooldown.get(name)
    return until is not None and time.monotonic() < until


def _run_search(query: str, limit: int, provider: str) -> tuple[list[SearchHit], str | None, list[str]]:
    """Return ``(hits, provider_used, failures)``; never raises."""
    order = _AUTO_ORDER if provider == "auto" else (provider,)
    failures: list[str] = []
    for name in order:
        searcher = _PROVIDERS.get(name)
        if searcher is None:
            failures.append(f"{name}: 未知 provider")
            continue
        if provider == "auto" and _on_cooldown(name):
            failures.append(f"{name}: 冷却中")
            continue
        try:
            hits = searcher(query, limit)
        except Exception as exc:
            logger.warning("search provider %s failed: %s: %s", name, type(exc).__name__, exc)
            _provider_cooldown[name] = time.monotonic() + _COOLDOWN_SECONDS
            failures.append(f"{name}: {type(exc).__name__}")
            continue
        if hits:
            _provider_cooldown.pop(name, None)
            return hits, name, failures
        failures.append(f"{name}: 无结果")
    return [], None, failures


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the public web (no API key required).

    Use this when the answer depends on facts you may not know, may be
    outdated, or when the user asks for sources. Follow up with
    ``fetch_webpage`` when a snippet is not enough.

    Args:
        query: Natural-language search query. Prefer specific keywords.
    """
    text = (query or "").strip()
    if not text:
        return "错误: 搜索词为空"

    settings = get_settings()
    hits, used, failures = _run_search(text, settings.web_max_results, settings.search_provider)
    if not hits:
        detail = "；".join(failures) if failures else "所有 provider 均无结果"
        return f"错误: 搜索失败（{detail}）。可以换个查询词，或直接基于已有知识回答并说明未联网。"

    lines = [f"搜索词: {text}", f"来源: {used}", f"结果数: {len(hits)}", ""]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. {hit.title}")
        if hit.url:
            lines.append(f"   链接: {hit.url}")
        if hit.snippet:
            lines.append(f"   摘要: {hit.snippet[:300]}")
    return "\n".join(lines)


def blocked_target(url: str) -> str | None:
    """Return a human-readable reason when ``url`` must not be fetched."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "只支持 http/https 链接"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "链接缺少主机名"
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return "拒绝访问本机/内网地址"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if literal.is_private or literal.is_loopback or literal.is_link_local or literal.is_reserved:
            return "拒绝访问本机/内网地址"
        return None
    # Hostname: resolve first so DNS-based SSRF (nip.io style) is caught too.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return None  # let httpx produce the real DNS error
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return "拒绝访问本机/内网地址"
    return None


def _extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    root = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(id="content")
        or soup.body
        or soup
    )
    lines: list[str] = []
    for line in root.get_text("\n").splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            lines.append(cleaned)
    return title, "\n".join(lines)


@tool
def fetch_webpage(url: str) -> str:
    """Download a page and return its readable text (scripts/nav stripped).

    Use after ``web_search`` when you need details a snippet cannot give.

    Args:
        url: Absolute http/https URL.
    """
    target = (url or "").strip()
    if not target:
        return "错误: URL 为空"
    if not target.startswith(("http://", "https://")):
        target = "https://" + target

    refusal = blocked_target(target)
    if refusal:
        return f"错误: {refusal}"

    settings = get_settings()
    try:
        with httpx.Client(
            timeout=settings.tool_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": _ACCEPT_LANGUAGE},
        ) as client, client.stream("GET", target) as response:
            if response.status_code >= 400:
                return f"错误: HTTP {response.status_code}（{target}）"
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith(_TEXTUAL_TYPES):
                return f"错误: 不支持的内容类型 {content_type}（只解析 HTML/纯文本）"
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                total += len(chunk)
                if total >= settings.fetch_max_bytes:
                    break
            encoding = response.encoding or "utf-8"
    except httpx.TimeoutException:
        return f"错误: 抓取 {target} 超时"
    except httpx.HTTPError as exc:
        logger.warning("fetch_webpage failed: %s: %s", type(exc).__name__, exc)
        return f"错误: 抓取失败（{type(exc).__name__}）"

    body = b"".join(chunks).decode(encoding, errors="replace")
    title, text = _extract_text(body)
    if not text:
        return f"页面 {target} 没有可读文本。"

    truncated = len(text) > settings.fetch_max_chars
    if truncated:
        text = text[: settings.fetch_max_chars]

    header = [f"URL: {target}"]
    if title:
        header.append(f"标题: {title}")
    header.append(f"正文{'（已截断）' if truncated else ''}:")
    return "\n".join(header) + "\n" + text
