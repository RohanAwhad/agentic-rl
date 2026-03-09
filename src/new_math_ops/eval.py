"""Evaluation utilities for New Math Ops."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict

from src.new_math_ops.dataset import NewMathOpsRow
from src.new_math_ops.prompts import ChatMessage, build_messages
from src.new_math_ops.reward import parse_final_answer


class PredictionRow(TypedDict):
    id: str
    input: str
    expected_output: int
    predicted_output: int | None
    is_correct: bool
    format_error: bool
    raw_response: str
    arithmetic_family: str
    difficulty_level: str
    n_ops: int
    latency_seconds: float


class BucketMetrics(TypedDict):
    total: int
    correct: int
    format_errors: int
    accuracy: float
    format_error_rate: float


class Metrics(TypedDict):
    total: int
    correct: int
    accuracy: float
    format_errors: int
    format_error_rate: float
    by_family: dict[str, BucketMetrics]
    by_difficulty: dict[str, BucketMetrics]
    by_n_ops: dict[int, BucketMetrics]


class CompletionClient(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...


def compute_metrics(predictions: list[PredictionRow]) -> Metrics:
    total = len(predictions)
    correct = sum(1 for row in predictions if row["is_correct"])
    format_errors = sum(1 for row in predictions if row["format_error"])

    by_family_counts: defaultdict[str, dict[str, int]] = defaultdict(_init_bucket)
    by_difficulty_counts: defaultdict[str, dict[str, int]] = defaultdict(_init_bucket)
    by_n_ops_counts: defaultdict[int, dict[str, int]] = defaultdict(_init_bucket)

    for row in predictions:
        family_bucket = by_family_counts[row["arithmetic_family"]]
        difficulty_bucket = by_difficulty_counts[row["difficulty_level"]]
        n_ops_bucket = by_n_ops_counts[row["n_ops"]]

        for bucket in (family_bucket, difficulty_bucket, n_ops_bucket):
            bucket["total"] += 1
            bucket["correct"] += int(row["is_correct"])
            bucket["format_errors"] += int(row["format_error"])

    by_family = {key: _finalize_bucket(bucket) for key, bucket in by_family_counts.items()}
    by_difficulty = {key: _finalize_bucket(bucket) for key, bucket in by_difficulty_counts.items()}
    by_n_ops = {key: _finalize_bucket(bucket) for key, bucket in by_n_ops_counts.items()}

    accuracy = (correct / total) if total else 0.0
    format_error_rate = (format_errors / total) if total else 0.0

    metrics: Metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_errors": format_errors,
        "format_error_rate": format_error_rate,
        "by_family": by_family,
        "by_difficulty": by_difficulty,
        "by_n_ops": by_n_ops,
    }
    return metrics


async def evaluate_dataset_rows(
    *,
    rows: list[NewMathOpsRow],
    client: CompletionClient,
    concurrency: int,
) -> tuple[list[PredictionRow], Metrics]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_row(row: NewMathOpsRow) -> PredictionRow:
        async with semaphore:
            expression = row["input"]
            expected_output = row["expected_output"]
            metadata = row["metadata"]

            start = time.perf_counter()
            raw_response = await client.complete(build_messages(expression))
            latency = time.perf_counter() - start

            predicted_output = parse_final_answer(raw_response)
            format_error = predicted_output is None
            is_correct = predicted_output == expected_output

            return {
                "id": metadata["id"],
                "input": expression,
                "expected_output": expected_output,
                "predicted_output": predicted_output,
                "is_correct": is_correct,
                "format_error": format_error,
                "raw_response": raw_response,
                "arithmetic_family": metadata["arithmetic_family"],
                "difficulty_level": metadata["difficulty_level"],
                "n_ops": metadata["n_ops"],
                "latency_seconds": round(latency, 6),
            }

    predictions: list[PredictionRow] = []
    pending = [evaluate_row(row) for row in rows]

    for completed in asyncio.as_completed(pending):
        predictions.append(await completed)

    metrics = compute_metrics(predictions)
    return predictions, metrics


def write_run_artifacts(
    *,
    output_dir: Path,
    label: str,
    predictions: list[PredictionRow],
    metrics: Metrics,
    run_config: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"{label}_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    metrics_path = output_dir / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    config_path = output_dir / f"{label}_run_config.json"
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "config": config_path,
    }


def _init_bucket() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "format_errors": 0,
    }


def _finalize_bucket(bucket: dict[str, int]) -> BucketMetrics:
    total = bucket["total"]
    if total == 0:
        accuracy = 0.0
        format_error_rate = 0.0
    else:
        accuracy = bucket["correct"] / total
        format_error_rate = bucket["format_errors"] / total

    return {
        "total": bucket["total"],
        "correct": bucket["correct"],
        "format_errors": bucket["format_errors"],
        "accuracy": accuracy,
        "format_error_rate": format_error_rate,
    }
