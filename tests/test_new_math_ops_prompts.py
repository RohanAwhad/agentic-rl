"""Tests for New Math Ops prompt formatting."""

from __future__ import annotations

from src.new_math_ops.prompts import PROMPT_VERSION, build_messages


def test_prompt_version_is_pinned():
    assert PROMPT_VERSION == "v2"


def test_build_messages_contains_expected_structure():
    messages = build_messages("9 / 2 ## 3")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "<final_answer>integer</final_answer>" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "Expression: 9 / 2 ## 3" in messages[1]["content"]
    assert "<final_answer>integer</final_answer>" in messages[1]["content"]
