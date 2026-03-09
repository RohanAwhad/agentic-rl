"""Evaluate a model on a deterministic New Math Ops holdout split."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from new_math_ops import (
    ChatMessage,
    PROMPT_VERSION,
    evaluate_dataset_rows,
    load_dataset_rows,
    write_run_artifacts,
)
from openai import AsyncOpenAI

from src.new_math_ops_adapter.dataset import split_train_val


def default_dataset_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "new_math_ops"
        / "data"
        / "new_math_ops_v7_10000"
        / "dataset.jsonl"
    )


def extract_choice_text(choice: object) -> str:
    message = getattr(choice, "message", choice)

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
            else:
                text = getattr(part, "text", "")
            parts.append(str(text))
        return "".join(parts).strip()

    return str(content).strip()


class OpenAICompletionClient:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=list(messages),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return extract_choice_text(response.choices[0])


async def run(args: argparse.Namespace) -> None:
    rows = load_dataset_rows(
        args.dataset,
        args.dataset_limit,
    )
    train_rows, val_rows = split_train_val(
        rows,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )

    split_rows = train_rows if args.split == "train" else val_rows
    if args.limit is not None:
        split_rows = split_rows[: args.limit]

    client = AsyncOpenAI(base_url=args.vllm_url, api_key="dummy")
    completion_client = OpenAICompletionClient(
        client=client,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(
        f"[{args.label}] split={args.split} rows={len(split_rows)} "
        f"model={args.model_name} prompt={PROMPT_VERSION}"
    )

    predictions, metrics = await evaluate_dataset_rows(
        rows=split_rows,
        client=completion_client,
        concurrency=args.concurrency,
    )

    run_config = {
        "label": args.label,
        "dataset": str(args.dataset),
        "dataset_limit": args.dataset_limit,
        "split": args.split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "eval_limit": args.limit,
        "model_name": args.model_name,
        "vllm_url": args.vllm_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "prompt_version": PROMPT_VERSION,
    }

    artifact_paths = write_run_artifacts(
        output_dir=args.output_dir,
        label=args.label,
        predictions=predictions,
        metrics=metrics,
        run_config=run_config,
    )

    print(f"[{args.label}] accuracy={metrics['accuracy']:.1%}")
    print(f"[{args.label}] format_error_rate={metrics['format_error_rate']:.1%}")
    print(f"[{args.label}] wrote metrics to {artifact_paths['metrics']}")

    summary_path = args.output_dir / f"{args.label}_summary.json"
    summary_payload = {
        "label": args.label,
        "total": metrics["total"],
        "correct": metrics["correct"],
        "accuracy": metrics["accuracy"],
        "format_error_rate": metrics["format_error_rate"],
        "metrics_path": str(artifact_paths["metrics"]),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate New Math Ops split")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("results/new_math_ops/eval"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
