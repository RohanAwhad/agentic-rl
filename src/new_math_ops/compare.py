"""Metrics comparison helpers for New Math Ops eval runs."""

from __future__ import annotations

from typing import Any, Mapping


def compare_metrics(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute deltas between baseline and candidate metric payloads."""
    baseline_family = _coerce_bucket_group(baseline.get("by_family"))
    candidate_family = _coerce_bucket_group(candidate.get("by_family"))

    baseline_difficulty = _coerce_bucket_group(baseline.get("by_difficulty"))
    candidate_difficulty = _coerce_bucket_group(candidate.get("by_difficulty"))

    baseline_n_ops = _coerce_bucket_group(baseline.get("by_n_ops"))
    candidate_n_ops = _coerce_bucket_group(candidate.get("by_n_ops"))

    family_keys = sorted(set(baseline_family) | set(candidate_family))
    difficulty_keys = sorted(set(baseline_difficulty) | set(candidate_difficulty))
    n_ops_keys = _sort_n_ops_keys(set(baseline_n_ops) | set(candidate_n_ops))

    return {
        "overall": _compare_bucket(baseline, candidate),
        "by_family": {
            key: _compare_bucket(baseline_family.get(key, {}), candidate_family.get(key, {}))
            for key in family_keys
        },
        "by_difficulty": {
            key: _compare_bucket(
                baseline_difficulty.get(key, {}),
                candidate_difficulty.get(key, {}),
            )
            for key in difficulty_keys
        },
        "by_n_ops": {
            key: _compare_bucket(baseline_n_ops.get(key, {}), candidate_n_ops.get(key, {}))
            for key in n_ops_keys
        },
    }


def _coerce_bucket_group(raw_group: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_group, dict):
        return {}

    buckets: dict[str, dict[str, Any]] = {}
    for key, value in raw_group.items():
        if isinstance(value, dict):
            buckets[str(key)] = value
    return buckets


def _compare_bucket(
    baseline_bucket: Mapping[str, Any],
    candidate_bucket: Mapping[str, Any],
) -> dict[str, float | int]:
    baseline_total = _get_int(baseline_bucket, "total")
    candidate_total = _get_int(candidate_bucket, "total")

    baseline_correct = _get_int(baseline_bucket, "correct")
    candidate_correct = _get_int(candidate_bucket, "correct")

    baseline_format_errors = _get_int(baseline_bucket, "format_errors")
    candidate_format_errors = _get_int(candidate_bucket, "format_errors")

    baseline_accuracy = _get_float(baseline_bucket, "accuracy")
    candidate_accuracy = _get_float(candidate_bucket, "accuracy")

    baseline_format_error_rate = _get_float(baseline_bucket, "format_error_rate")
    candidate_format_error_rate = _get_float(candidate_bucket, "format_error_rate")

    return {
        "baseline_total": baseline_total,
        "candidate_total": candidate_total,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_format_errors": baseline_format_errors,
        "candidate_format_errors": candidate_format_errors,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "baseline_format_error_rate": baseline_format_error_rate,
        "candidate_format_error_rate": candidate_format_error_rate,
        "format_error_rate_delta": candidate_format_error_rate - baseline_format_error_rate,
    }


def _get_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _get_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _sort_n_ops_keys(keys: set[str]) -> list[str]:
    def sort_key(key: str) -> tuple[int, int | str]:
        trimmed = key.lstrip("-")
        if trimmed.isdigit():
            return (0, int(key))
        return (1, key)

    return sorted(keys, key=sort_key)
