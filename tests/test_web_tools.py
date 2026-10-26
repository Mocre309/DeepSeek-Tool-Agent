"""Web tool tests — pure functions and provider chain, no sockets."""

from __future__ import annotations

import pytest

from app.agent.tools import web
from app.agent.tools.web import SearchHit, _extract_text, _run_search, blocked_target


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost/admin",
            "http://127.0.0.1:8000/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/",
            "http://[::1]/",
            "http://metadata.google.internal/",
        ],
    )
    def test_blocks_internal_targets(self, url: str) -> None:
        assert blocked_target(url) is not None

    @pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd"])
    def test_blocks_non_http_schemes(self, url: str) -> None:
        assert blocked_target(url) is not None

    def test_allows_public_hosts(self) -> None:
        assert blocked_target("https://example.com/page") is None


class TestExtraction:
    HTML = """
    <html><head><title>测试页</title></head>
    <body>
      <nav>导航栏不要</nav>
      <script>console.log('不要')</script>
      <article>
        <h1>标题</h1>
        <p>第一段正文。</p>
        <p>第二段正文。</p>
      </article>
      <footer>页脚不要</footer>
    </body></html>
    """

    def test_prefers_article_and_drops_noise(self) -> None:
        title, text = _extract_text(self.HTML)
        assert title == "测试页"
        assert "第一段正文。" in text
        assert "第二段正文。" in text
        assert "导航栏不要" not in text
        assert "console.log" not in text
        assert "页脚不要" not in text


class TestProviderChain:
    @pytest.fixture(autouse=True)
    def _clear_cooldown(self):
        web._provider_cooldown.clear()
        yield
        web._provider_cooldown.clear()

    def test_falls_back_to_next_provider(self, monkeypatch) -> None:
        def boom(query: str, limit: int):
            raise TimeoutError("upstream down")

        def works(query: str, limit: int):
            return [SearchHit(title="hit", url="https://example.com", snippet="snip")]

        monkeypatch.setitem(web._PROVIDERS, "bing", boom)
        monkeypatch.setitem(web._PROVIDERS, "duckduckgo", works)

        hits, used, failures = _run_search("q", 5, "auto")
        assert used == "duckduckgo"
        assert hits[0].title == "hit"
        assert any("bing" in failure for failure in failures)

    def test_failed_provider_cools_down(self, monkeypatch) -> None:
        calls: list[str] = []

        def boom(query: str, limit: int):
            calls.append("bing")
            raise TimeoutError("down")

        def works(query: str, limit: int):
            return [SearchHit(title="t", url="u", snippet="s")]

        monkeypatch.setitem(web._PROVIDERS, "bing", boom)
        monkeypatch.setitem(web._PROVIDERS, "duckduckgo", works)

        _run_search("q", 5, "auto")
        _run_search("q", 5, "auto")
        assert calls == ["bing"], "a provider on cooldown must not be called again"

    def test_all_providers_failing_is_reported(self, monkeypatch) -> None:
        def boom(query: str, limit: int):
            raise TimeoutError("down")

        monkeypatch.setitem(web._PROVIDERS, "bing", boom)
        monkeypatch.setitem(web._PROVIDERS, "duckduckgo", boom)

        hits, used, failures = _run_search("q", 5, "auto")
        assert hits == [] and used is None
        assert len(failures) == 2

    def test_unknown_provider_is_handled(self) -> None:
        hits, used, failures = _run_search("q", 5, "not-a-provider")
        assert hits == [] and used is None
        assert "未知 provider" in failures[0]
