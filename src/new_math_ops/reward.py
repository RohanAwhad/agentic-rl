"""Response parsing and binary reward for New Math Ops."""

from __future__ import annotations

import re
from typing import Any

FINAL_ANSWER_PATTERN = re.compile(
    r"<\s*final_answer\s*>\s*(-?\d+)\s*<\s*/\s*final_answer\s*>",
    re.IGNORECASE,
)


def parse_final_answer(text: str) -> int | None:
    """Parse integer from <final_answer>...</final_answer>."""
    match = FINAL_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def extract_choice_text(choice: Any) -> str:
    """Extract assistant text from OpenAI choice-like objects."""
    if hasattr(choice, "message"):
        message = choice.message
    elif isinstance(choice, dict) and "message" in choice:
        message = choice["message"]
    else:
        message = choice

    content = _extract_message_content(message)
    return content.strip()


def score_prediction(raw_response: str, expected_output: int) -> tuple[float, int | None, bool]:
    """Binary reward from raw response text."""
    predicted_output = parse_final_answer(raw_response)
    format_error = predicted_output is None
    reward = 1.0 if predicted_output == expected_output else 0.0
    return reward, predicted_output, format_error


def reward_from_choice(choice: Any, expected_output: int) -> tuple[float, int | None, bool, str]:
    """Extract text, parse answer, and compute binary reward."""
    raw_response = extract_choice_text(choice)
    reward, predicted_output, format_error = score_prediction(raw_response, expected_output)
    return reward, predicted_output, format_error, raw_response


def _extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
            else:
                text = getattr(part, "text", "")
            parts.append(str(text))
        return "".join(parts)

    return str(content)
