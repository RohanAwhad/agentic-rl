"""Tests for New Math Ops evaluation helpers."""

from __future__ import annotations

import json

import pytest

from src.new_math_ops.eval import evaluate_dataset_rows, write_run_artifacts


class _FakeClient:
    async def complete(self, messages):
        user_content = messages[1]["content"]
        if "8 ## 3" in user_content:
            return "<final_answer>5</final_answer>"
        if "1 @@ 9" in user_content:
            return "not-valid"
        return "<final_answer>0</final_answer>"


@pytest.mark.asyncio
async def test_evaluate_dataset_rows_computes_bucket_metrics():
    rows = [
        {
            "input": "8 ## 3",
            "expected_output": 5,
            "metadata": {
                "id": "s1",
                "arithmetic_family": "new",
                "difficulty_level": "L1",
                "n_ops": 1,
                "op_seq": ["##"],
            },
        },
        {
            "input": "1 @@ 9",
            "expected_output": 9,
            "metadata": {
                "id": "s2",
                "arithmetic_family": "new",
                "difficulty_level": "L2",
                "n_ops": 1,
                "op_seq": ["@@"],
            },
        },
    ]

    predictions, metrics = await evaluate_dataset_rows(
        rows=rows,
        client=_FakeClient(),
        concurrency=2,
    )

    assert len(predictions) == 2
    assert metrics["total"] == 2
    assert metrics["correct"] == 1
    assert metrics["format_errors"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["format_error_rate"] == 0.5
    assert "new" in metrics["by_family"]
    assert "L1" in metrics["by_difficulty"]
    assert 1 in metrics["by_n_ops"]


def test_write_run_artifacts_writes_files(tmp_path):
    predictions = [
        {
            "id": "s1",
            "input": "8 ## 3",
            "expected_output": 5,
            "predicted_output": 5,
            "is_correct": True,
            "format_error": False,
            "raw_response": "<final_answer>5</final_answer>",
            "arithmetic_family": "new",
            "difficulty_level": "L1",
            "n_ops": 1,
            "latency_seconds": 0.01,
        }
    ]
    metrics = {
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
        "format_errors": 0,
        "format_error_rate": 0.0,
        "by_family": {},
        "by_difficulty": {},
        "by_n_ops": {},
    }
    run_config = {
        "model_name": "Qwen/Qwen3-4B",
        "prompt_version": "v2",
    }

    paths = write_run_artifacts(
        output_dir=tmp_path,
        label="unit",
        predictions=predictions,
        metrics=metrics,
        run_config=run_config,
    )

    assert paths["predictions"].exists()
    assert paths["metrics"].exists()
    assert paths["config"].exists()

    row = json.loads(paths["predictions"].read_text(encoding="utf-8").strip())
    assert row["id"] == "s1"
