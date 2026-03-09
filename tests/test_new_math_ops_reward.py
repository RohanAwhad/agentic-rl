"""Tests for New Math Ops answer parsing and reward."""

from __future__ import annotations

from src.new_math_ops.reward import (
    extract_choice_text,
    parse_final_answer,
    reward_from_choice,
    score_prediction,
)


def test_parse_final_answer_accepts_integer_tag():
    assert parse_final_answer("<final_answer>-17</final_answer>") == -17


def test_parse_final_answer_returns_none_for_missing_tag():
    assert parse_final_answer("Answer: -17") is None


def test_score_prediction_binary_reward_exact_match():
    reward, predicted_output, format_error = score_prediction(
        "<final_answer>42</final_answer>",
        expected_output=42,
    )
    assert reward == 1.0
    assert predicted_output == 42
    assert format_error is False


def test_score_prediction_binary_reward_wrong_value():
    reward, predicted_output, format_error = score_prediction(
        "<final_answer>4</final_answer>",
        expected_output=5,
    )
    assert reward == 0.0
    assert predicted_output == 4
    assert format_error is False


def test_score_prediction_binary_reward_format_error():
    reward, predicted_output, format_error = score_prediction("not xml", expected_output=5)
    assert reward == 0.0
    assert predicted_output is None
    assert format_error is True


def test_extract_choice_text_supports_list_content():
    choice = {
        "message": {
            "content": [
                {"text": "step"},
                {"text": "<final_answer>3</final_answer>"},
            ]
        }
    }
    assert extract_choice_text(choice) == "step<final_answer>3</final_answer>"


def test_reward_from_choice_reads_dict_like_choice():
    choice = {"message": {"content": "<final_answer>9</final_answer>"}}
    reward, predicted_output, format_error, raw_response = reward_from_choice(choice, 9)

    assert reward == 1.0
    assert predicted_output == 9
    assert format_error is False
    assert raw_response == "<final_answer>9</final_answer>"
