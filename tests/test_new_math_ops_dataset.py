"""Tests for New Math Ops dataset loading/splitting."""

from __future__ import annotations

import json

import pytest

from src.new_math_ops.dataset import load_dataset_rows, split_train_val


def _row(sample_id: str, expected_output: int) -> dict:
    return {
        "input": f"{expected_output} + 0",
        "expected_output": expected_output,
        "metadata": {
            "id": sample_id,
            "arithmetic_family": "normal",
            "difficulty_level": "L1",
            "n_ops": 1,
            "op_seq": ["+"],
        },
    }


def test_load_dataset_rows_validates_schema(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [_row("s1", 1), _row("s2", 2)]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_dataset_rows(dataset_path)

    assert len(loaded) == 2
    assert loaded[0]["metadata"]["id"] == "s1"
    assert loaded[1]["expected_output"] == 2


def test_load_dataset_rows_rejects_invalid_schema(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    bad_row = {
        "input": "1 + 2",
        "expected_output": 3,
    }
    dataset_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset_rows(dataset_path)


def test_split_train_val_is_deterministic():
    rows = [_row(f"sample_{index:03d}", index) for index in range(40)]

    train_a, val_a = split_train_val(rows, train_ratio=0.75, seed=11)
    train_b, val_b = split_train_val(rows, train_ratio=0.75, seed=11)
    train_c, val_c = split_train_val(rows, train_ratio=0.75, seed=12)

    ids_train_a = [row["metadata"]["id"] for row in train_a]
    ids_train_b = [row["metadata"]["id"] for row in train_b]
    ids_train_c = [row["metadata"]["id"] for row in train_c]

    assert ids_train_a == ids_train_b
    assert [row["metadata"]["id"] for row in val_a] == [row["metadata"]["id"] for row in val_b]
    assert ids_train_a != ids_train_c
    assert len(train_a) == 30
    assert len(val_a) == 10
