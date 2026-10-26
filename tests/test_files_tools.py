"""Workspace file tools: one sandbox root, reads free, writes gated."""

from __future__ import annotations

import pytest

from app.agent.tools import TOOLS_REQUIRING_APPROVAL, resolve_tools
from app.agent.tools.files import (
    list_files,
    read_file,
    resolve_in_workspace,
    workspace_root,
    write_file,
)


def run(tool, **kwargs):
    return tool.invoke(kwargs)


class TestSandbox:
    @pytest.fixture(autouse=True)
    def _workspace(self, settings):
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        (settings.workspace_dir / "notes").mkdir(exist_ok=True)
        (settings.workspace_dir / "notes" / "todo.md").write_text("买牛奶", encoding="utf-8")
        yield

    def test_root_is_the_configured_dir(self, settings) -> None:
        assert workspace_root() == settings.workspace_dir.resolve()

    @pytest.mark.parametrize(
        "raw",
        ["../outside.txt", "../../etc/passwd", "notes/../../escape.txt", "C:/Windows/win.ini"],
    )
    def test_escapes_are_refused(self, raw: str) -> None:
        with pytest.raises(ValueError, match="越界"):
            resolve_in_workspace(raw)

    def test_inside_paths_resolve(self, settings) -> None:
        target = resolve_in_workspace("notes/todo.md")
        assert target == (settings.workspace_dir / "notes" / "todo.md").resolve()

    def test_traversal_is_also_blocked_in_tools(self) -> None:
        assert run(read_file, path="../../etc/passwd").startswith("错误")
        assert run(write_file, path="../evil.txt", content="x").startswith("错误")

    def test_symlink_escape_is_blocked(self, settings, tmp_path) -> None:
        outside = tmp_path / "secret.txt"
        outside.write_text("top secret", encoding="utf-8")
        link = settings.workspace_dir / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        assert run(read_file, path="link.txt").startswith("错误")


class TestReadWrite:
    @pytest.fixture(autouse=True)
    def _workspace(self, settings):
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        yield

    def test_write_then_read_round_trip(self, settings) -> None:
        saved = run(write_file, path="notes/deep.md", content="第一行\n第二行")
        assert saved.startswith("已创建: notes/deep.md")
        assert "/api/workspace/notes/deep.md" in saved
        assert (settings.workspace_dir / "notes" / "deep.md").read_text(encoding="utf-8") == (
            "第一行\n第二行"
        )

        body = run(read_file, path="notes/deep.md")
        assert "第一行" in body and "第二行" in body
        assert "行数: 2" in body

    def test_second_write_reports_update(self) -> None:
        run(write_file, path="a.txt", content="v1")
        assert run(write_file, path="a.txt", content="v2").startswith("已更新: a.txt")

    def test_list_files_shows_entries_and_dirs(self) -> None:
        run(write_file, path="notes/b.md", content="b")
        listing = run(list_files, path="notes")
        assert "b.md" in listing
        root_listing = run(list_files)
        assert "notes/" in root_listing

    def test_missing_file_is_reported(self) -> None:
        assert "不存在" in run(read_file, path="nope.txt")

    def test_directory_is_not_readable_as_text(self) -> None:
        run(write_file, path="sub/x.txt", content="x")
        assert "目录" in run(read_file, path="sub")

    def test_binary_file_is_refused(self, settings) -> None:
        (settings.workspace_dir / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        assert "不是文本文件" in run(read_file, path="blob.bin")

    def test_oversized_write_is_refused(self, settings, monkeypatch) -> None:
        monkeypatch.setattr(settings, "workspace_max_write_bytes", 10)
        assert "过大" in run(write_file, path="big.txt", content="x" * 100)

    def test_oversized_read_is_refused(self, settings, monkeypatch) -> None:
        (settings.workspace_dir / "big.txt").write_text("x" * 100, encoding="utf-8")
        monkeypatch.setattr(settings, "workspace_max_read_bytes", 10)
        assert "过大" in run(read_file, path="big.txt")

    def test_truncation_flag(self) -> None:
        run(write_file, path="long.txt", content="0123456789" * 10)
        body = run(read_file, path="long.txt", max_chars=20)
        assert "已截断" in body

    def test_write_into_directory_is_refused(self) -> None:
        run(write_file, path="dir/x.txt", content="x")
        assert "目录" in run(write_file, path="dir", content="x")


class TestRegistration:
    def test_writes_need_approval(self) -> None:
        assert "write_file" in TOOLS_REQUIRING_APPROVAL
        assert "read_file" not in TOOLS_REQUIRING_APPROVAL

    def test_tools_can_be_disabled(self) -> None:
        names = {tool.name for tool in resolve_tools(["write_file", "read_file"])}
        assert "write_file" not in names and "read_file" not in names
        assert "list_files" in names
