"""Dataset loading and deterministic split helpers for New Math Ops."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict


class NewMathOpsMetadata(TypedDict):
    id: str
    arithmetic_family: str
    difficulty_level: str
    n_ops: int
    op_seq: list[str]


class NewMathOpsRow(TypedDict):
    input: str
    expected_output: int
    metadata: NewMathOpsMetadata


def default_dataset_path() -> Path:
    """Default local dataset path inside this monorepo."""
    return (
        Path(__file__).resolve().parents[4]
        / "benchmarks"
        / "new_math_ops"
        / "data"
        / "new_math_ops_v7_10000"
        / "dataset.jsonl"
    )


def load_dataset_rows(dataset_path: str | Path, limit: int | None = None) -> list[NewMathOpsRow]:
    """Load New Math Ops JSONL rows with lightweight schema validation."""
    rows: list[NewMathOpsRow] = []
    path = Path(dataset_path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed: object = json.loads(line)
            rows.append(_validate_row(parsed))
            if limit is not None and len(rows) >= limit:
                break

    return rows


def split_train_val(
    rows: list[NewMathOpsRow],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[NewMathOpsRow], list[NewMathOpsRow]]:
    """Deterministically split rows into train/val."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    if not rows:
        return [], []

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    split_index = int(len(shuffled) * train_ratio)
    split_index = min(max(split_index, 1), len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]


def load_train_val_rows(
    dataset_path: str | Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[list[NewMathOpsRow], list[NewMathOpsRow]]:
    """Load a dataset then apply deterministic train/val split."""
    rows = load_dataset_rows(dataset_path, limit=limit)
    return split_train_val(rows, train_ratio=train_ratio, seed=seed)


def _validate_row(raw_row: object) -> NewMathOpsRow:
    if not isinstance(raw_row, dict):
        raise ValueError("dataset row must be a dict")

    expression = raw_row.get("input")
    expected_output = raw_row.get("expected_output")
    metadata_obj = raw_row.get("metadata")

    if not isinstance(expression, str):
        raise ValueError("row['input'] must be a string")
    if not isinstance(expected_output, int):
        raise ValueError("row['expected_output'] must be an int")
    if not isinstance(metadata_obj, dict):
        raise ValueError("row['metadata'] must be a dict")

    sample_id = metadata_obj.get("id")
    arithmetic_family = metadata_obj.get("arithmetic_family")
    difficulty_level = metadata_obj.get("difficulty_level")
    n_ops = metadata_obj.get("n_ops")
    op_seq = metadata_obj.get("op_seq")

    if not isinstance(sample_id, str):
        raise ValueError("metadata['id'] must be a string")
    if not isinstance(arithmetic_family, str):
        raise ValueError("metadata['arithmetic_family'] must be a string")
    if not isinstance(difficulty_level, str):
        raise ValueError("metadata['difficulty_level'] must be a string")
    if not isinstance(n_ops, int):
        raise ValueError("metadata['n_ops'] must be an int")
    if not isinstance(op_seq, list) or not all(isinstance(op, str) for op in op_seq):
        raise ValueError("metadata['op_seq'] must be a list[str]")

    metadata: NewMathOpsMetadata = {
        "id": sample_id,
        "arithmetic_family": arithmetic_family,
        "difficulty_level": difficulty_level,
        "n_ops": n_ops,
        "op_seq": op_seq,
    }

    row: NewMathOpsRow = {
        "input": expression,
        "expected_output": expected_output,
        "metadata": metadata,
    }
    return row
