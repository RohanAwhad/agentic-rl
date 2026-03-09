"""Tests for New Math Ops metric comparison helper."""

from __future__ import annotations

from src.new_math_ops_adapter.compare import compare_metrics


def test_compare_metrics_computes_expected_deltas():
    baseline = {
        "total": 10,
        "correct": 2,
        "accuracy": 0.2,
        "format_errors": 4,
        "format_error_rate": 0.4,
        "by_family": {
            "normal": {
                "total": 5,
                "correct": 2,
                "format_errors": 1,
                "accuracy": 0.4,
                "format_error_rate": 0.2,
            }
        },
        "by_difficulty": {},
        "by_n_ops": {},
    }
    candidate = {
        "total": 10,
        "correct": 5,
        "accuracy": 0.5,
        "format_errors": 2,
        "format_error_rate": 0.2,
        "by_family": {
            "normal": {
                "total": 5,
                "correct": 4,
                "format_errors": 1,
                "accuracy": 0.8,
                "format_error_rate": 0.2,
            }
        },
        "by_difficulty": {},
        "by_n_ops": {},
    }

    comparison = compare_metrics(baseline, candidate)

    assert comparison["overall"]["accuracy_delta"] == 0.3
    assert comparison["overall"]["format_error_rate_delta"] == -0.2
    assert comparison["by_family"]["normal"]["accuracy_delta"] == 0.4
