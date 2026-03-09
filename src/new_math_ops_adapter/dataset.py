"""Local dataset helpers for splitting benchmark rows."""

from __future__ import annotations

import random
from typing import Any


def split_train_val(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
