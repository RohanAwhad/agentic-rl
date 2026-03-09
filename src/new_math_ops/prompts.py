"""Prompt contract for New Math Ops evaluation/training."""

from __future__ import annotations

from typing import Literal, TypedDict

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are evaluating expressions in a normal and synthetic arithmetic system.
You may think through the problem step by step before responding.

Return your final answer as exactly one XML tag:
<final_answer>integer</final_answer>
"""

USER_TEMPLATE = """Expression: {expression}
Return <final_answer>integer</final_answer>."""


ChatRole = Literal["system", "user"]


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


def build_messages(expression: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(expression=expression)},
    ]
