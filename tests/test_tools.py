"""Tool-level tests. No network, no LLM."""

from __future__ import annotations

import json

import pytest

from app.agent.tools import TOOLS, describe_tools
from app.agent.tools.calculator import calculator
from app.agent.tools.clock import current_time
from app.agent.tools.notes import save_note, search_notes


def run(tool, **kwargs):
    return tool.invoke(kwargs)


class TestCalculator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1+2", "3"),
            ("(1280*0.85)**2/7", "169106.2857142857"),
            ("sqrt(144)", "12"),
            ("2**10", "1024"),
            ("round(pi, 4)", "3.1416"),
            ("-3 % 5", "2"),
        ],
    )
    def test_evaluates(self, expression: str, expected: str) -> None:
        assert run(calculator, expression=expression) == expected

    @pytest.mark.parametrize(
        "expression",
        ["1/0", "sqrt(-1)", "2**1000", "__import__('os').system('ls')", "open('/etc/passwd')"],
    )
    def test_rejects_or_reports_errors(self, expression: str) -> None:
        result = run(calculator, expression=expression)
        assert result.startswith("错误"), result

    def test_rejects_empty(self) -> None:
        assert run(calculator, expression="   ").startswith("错误")

    def test_rejects_overlong_input(self) -> None:
        assert run(calculator, expression="1+" * 200).startswith("错误")


class TestClock:
    def test_known_timezone(self) -> None:
        result = run(current_time, timezone="Asia/Shanghai")
        assert "日期:" in result
        assert "+08" in result

    def test_default_timezone(self) -> None:
        assert "UTC 偏移:" in run(current_time)

    def test_unknown_timezone_is_a_hint_not_a_crash(self) -> None:
        result = run(current_time, timezone="Mars/Olympus")
        assert result.startswith("错误")
        assert "IANA" in result


class TestNotes:
    def test_save_then_recall(self) -> None:
        saved = run(save_note, content="用户是 Python 后端工程师", tags="profile")
        assert saved.startswith("已保存")

        found = json.loads(run(search_notes, query="Python"))
        assert len(found) == 1
        assert found[0]["content"] == "用户是 Python 后端工程师"
        assert found[0]["tags"] == "profile"

    def test_latest_notes_when_query_empty(self) -> None:
        run(save_note, content="第一条")
        run(save_note, content="第二条")
        found = json.loads(run(search_notes, query=""))
        assert [item["content"] for item in found] == ["第二条", "第一条"]

    def test_miss_is_explained(self) -> None:
        assert "没有" in run(search_notes, query="不存在的东西")

    def test_limit_is_clamped(self) -> None:
        for index in range(5):
            run(save_note, content=f"note-{index}")
        assert len(json.loads(run(search_notes, query="note", limit=999))) == 5
        assert len(json.loads(run(search_notes, query="note", limit=2))) == 2

    def test_empty_content_rejected(self) -> None:
        assert run(save_note, content="  ").startswith("错误")


class TestRegistry:
    def test_every_tool_is_described(self) -> None:
        described = describe_tools()
        assert len(described) == len(TOOLS) == 10
        for item in described:
            assert item["description"], item["name"]
            assert item["label"] and item["icon"]

    def test_names_are_unique_and_snake_case(self) -> None:
        names = [tool.name for tool in TOOLS]
        assert len(names) == len(set(names))
        assert all(name.islower() for name in names)

    def test_prompt_documents_the_options_protocol(self) -> None:
        """The UI renders agent options from a fenced block, so the prompt must say so."""
        from app.agent.prompts import build_system_prompt

        prompt = build_system_prompt()
        assert "```options" in prompt
        assert "让用户做选择" in prompt

    def test_prompt_only_lists_enabled_tools(self) -> None:
        from app.agent.prompts import build_system_prompt

        prompt = build_system_prompt(["write_file", "download_file"])
        assert "`write_file`" not in prompt
        assert "`download_file`" not in prompt
        assert "`calculator`" in prompt
