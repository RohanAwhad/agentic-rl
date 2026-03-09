"""Compare baseline vs post-training New Math Ops metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.new_math_ops.compare import compare_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare New Math Ops metric files")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/new_math_ops/eval/comparison.json"),
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> None:
    args = build_parser().parse_args()

    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)
    comparison = compare_metrics(baseline, candidate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    overall = comparison["overall"]
    print(f"wrote comparison to {args.output}")
    print(
        "overall accuracy: "
        f"{overall['baseline_accuracy']:.1%} -> {overall['candidate_accuracy']:.1%} "
        f"(delta {overall['accuracy_delta']:+.1%})"
    )
    print(
        "overall format_error_rate: "
        f"{overall['baseline_format_error_rate']:.1%} -> "
        f"{overall['candidate_format_error_rate']:.1%} "
        f"(delta {overall['format_error_rate_delta']:+.1%})"
    )


if __name__ == "__main__":
    main()
