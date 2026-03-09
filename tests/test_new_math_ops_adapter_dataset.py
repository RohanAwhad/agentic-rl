"""Tests for New Math Ops adapter dataset split helper."""

from __future__ import annotations

import pytest

from src.new_math_ops_adapter.dataset import split_train_val


def test_split_train_val_is_deterministic():
    rows = [{"metadata": {"id": f"sample_{index:03d}"}} for index in range(40)]

    train_a, val_a = split_train_val(rows, train_ratio=0.75, seed=11)
    train_b, val_b = split_train_val(rows, train_ratio=0.75, seed=11)
    train_c, _ = split_train_val(rows, train_ratio=0.75, seed=12)

    ids_train_a = [row["metadata"]["id"] for row in train_a]
    ids_train_b = [row["metadata"]["id"] for row in train_b]
    ids_train_c = [row["metadata"]["id"] for row in train_c]

    assert ids_train_a == ids_train_b
    assert [row["metadata"]["id"] for row in val_a] == [row["metadata"]["id"] for row in val_b]
    assert ids_train_a != ids_train_c
    assert len(train_a) == 30
    assert len(val_a) == 10


def test_split_train_val_validates_ratio():
    with pytest.raises(ValueError):
        split_train_val([{"metadata": {"id": "x"}}], train_ratio=1.0, seed=1)
