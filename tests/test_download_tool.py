"""Download tool: filename safety, SSRF refusal, size cap and output shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.tools import download
from app.agent.tools.download import (
    _DownloadFailed,
    _filename_from_url,
    _safe_filename,
    download_file,
)


class TestFilenames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report.pdf", "report.pdf"),
            ("report.PDF", "report.pdf"),
            ("../../etc/passwd", "passwd"),
            ("/tmp/a/b/notes.txt", "notes.txt"),
            ("..\\..\\windows\\system32\\x.exe", "x.exe"),
            ("   ", "download.bin"),
            ("报告.pdf", "download.pdf"),  # non-ASCII stem is dropped, extension kept
            ("no-extension", "no-extension"),
        ],
    )
    def test_sanitizes(self, raw: str, expected: str) -> None:
        assert _safe_filename(raw) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.com/a/report.pdf", "report.pdf"),
            ("https://example.com/a/report.pdf?token=abc", "report.pdf"),
            ("https://example.com/dir/", "download.bin"),
            ("https://example.com", "download.bin"),
            ("https://example.com/%E6%8A%A5%E5%91%8A.pdf", "download.pdf"),
        ],
    )
    def test_derives_from_url(self, url: str, expected: str) -> None:
        assert _filename_from_url(url) == expected


class TestGuardrails:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/admin",
            "http://localhost/secret",
            "http://169.254.169.254/latest/meta-data/",
            "ftp://example.com/x.zip",
            "file:///etc/passwd",
        ],
    )
    def test_refuses_dangerous_targets(self, url: str) -> None:
        result = download_file.invoke({"url": url})
        assert result.startswith("错误:")

    def test_refuses_empty_url(self) -> None:
        assert download_file.invoke({"url": "  "}).startswith("错误:")

    def test_size_cap_is_reported(self, monkeypatch) -> None:
        def explode(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
            raise _DownloadFailed("文件超过 25 MB 上限")

        monkeypatch.setattr(download, "_download_to", explode)
        result = download_file.invoke({"url": "https://example.com/big.iso"})
        assert "超过" in result


class TestSuccess:
    def test_writes_file_and_returns_link(self, monkeypatch, settings) -> None:
        payload = b"%PDF-1.4 fake"

        def fake(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return len(payload), "application/pdf"

        monkeypatch.setattr(download, "_download_to", fake)
        result = download_file.invoke({"url": "https://example.com/report.pdf"})

        assert "已保存: report-" in result
        assert "下载链接: /api/files/" in result
        assert "application/pdf" in result

        link_name = result.split("下载链接: /api/files/")[1].splitlines()[0]
        saved = settings.downloads_dir / link_name
        assert saved.is_file()
        assert saved.read_bytes() == payload

    def test_empty_file_is_an_error(self, monkeypatch) -> None:
        def fake(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"")
            return 0, ""

        monkeypatch.setattr(download, "_download_to", fake)
        result = download_file.invoke({"url": "https://example.com/empty.txt"})
        assert "空文件" in result

    def test_same_url_reuses_the_same_name(self, monkeypatch) -> None:
        def fake(url: str, dest: Path, max_bytes: int, timeout: float) -> tuple[int, str]:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            return 1, "text/plain"

        monkeypatch.setattr(download, "_download_to", fake)
        first = download_file.invoke({"url": "https://example.com/a.txt"})
        second = download_file.invoke({"url": "https://example.com/a.txt"})
        assert first.splitlines()[0] == second.splitlines()[0]

        other = download_file.invoke({"url": "https://example.com/b.txt"})
        assert other.splitlines()[0] != first.splitlines()[0]
