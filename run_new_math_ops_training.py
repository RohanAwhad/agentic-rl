"""ART GRPO training on New Math Ops single-turn arithmetic tasks."""

from __future__ import annotations

import os

# Keep vLLM on V0 engine for local ART compatibility.
os.environ["VLLM_USE_V1"] = "0"


def main() -> None:
    import argparse
    import asyncio
    import json
    import logging
    import random
    import time
    from pathlib import Path

    import art
    from art.local.backend import LocalBackend

    from src.new_math_ops.dataset import default_dataset_path, load_train_val_rows
    from src.new_math_ops.prompts import PROMPT_VERSION, build_messages
    from src.new_math_ops.reward import reward_from_choice

    parser = argparse.ArgumentParser(description="Train ART on New Math Ops")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--tasks-per-iter", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=20260306)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-name", default="qwen3-4b-new-math-ops")
    parser.add_argument("--project", default="agentic-rl-poc")
    parser.add_argument("--gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--backend-path", default="./.art-new-math-ops")
    parser.add_argument("--output-dir", type=Path, default=Path("results/new_math_ops"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["LiteLLM", "litellm", "httpx", "openai"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def rollout(
        model: "art.TrainableModel",
        sample: dict,
    ) -> "art.Trajectory":
        async with semaphore:
            client = model.openai_client()
            model_name = model.get_inference_name()

            messages = build_messages(sample["input"])
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            choice = response.choices[0]
            reward, _, _, _ = reward_from_choice(choice, sample["expected_output"])

            trajectory = art.Trajectory(
                messages_and_choices=[
                    messages[0],
                    messages[1],
                    choice,
                ]
            )
            trajectory.reward = reward
            return trajectory

    async def run() -> None:
        train_rows, val_rows = load_train_val_rows(
            args.dataset,
            train_ratio=args.train_ratio,
            seed=args.split_seed,
            limit=args.dataset_limit,
        )
        if not train_rows:
            raise ValueError("no training rows loaded")

        rollouts_per_iter = min(args.tasks_per_iter, len(train_rows)) * args.group_size

        print(f"{'=' * 60}")
        print("New Math Ops ART Training")
        print(f"{'=' * 60}")
        print(f"Dataset: {args.dataset}")
        print(f"Prompt version: {PROMPT_VERSION}")
        print(f"Train rows: {len(train_rows)}")
        print(f"Val rows: {len(val_rows)}")
        print(f"Iterations: {args.iterations}")
        print(f"Tasks per iteration: {args.tasks_per_iter}")
        print(f"Group size: {args.group_size}")
        print(f"Rollouts per iteration: {rollouts_per_iter}")
        print(f"Total planned rollouts: {rollouts_per_iter * args.iterations}")
        print(f"{'=' * 60}")

        model = art.TrainableModel(
            name=args.model_name,
            project=args.project,
            base_model=args.base_model,
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=args.gpu_mem_util),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )
        backend = LocalBackend(in_process=True, path=args.backend_path)
        await model.register(backend)

        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0
        if start_iteration > 0:
            logger.info("Resuming from step %d", start_iteration)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        reward_history: list[float] = []
        accuracy_history: list[float] = []
        timing_history: list[float] = []
        rollout_history: list[int] = []
        start_time = time.time()

        for iteration in range(start_iteration, args.iterations):
            iter_start = time.time()

            iter_rng = random.Random(args.sample_seed + iteration)
            iter_samples = iter_rng.sample(train_rows, min(args.tasks_per_iter, len(train_rows)))

            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(model, sample) for _ in range(args.group_size))
                    for sample in iter_samples
                ),
                pbar_desc=f"Iter {iteration + 1}/{args.iterations}",
            )

            rewards = [
                trajectory.reward for group in train_groups for trajectory in group.trajectories
            ]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            accuracy = (
                sum(1 for value in rewards if value >= 1.0) / len(rewards) if rewards else 0.0
            )

            reward_history.append(mean_reward)
            accuracy_history.append(accuracy)
            rollout_history.append(len(rewards))

            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=args.learning_rate),
            )

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            timing_history.append(iter_time)

            logger.info(
                "Iter %d/%d: mean_reward=%.3f accuracy=%.1f%% rollouts=%d time=%.0fs",
                iteration + 1,
                args.iterations,
                mean_reward,
                accuracy * 100,
                len(rewards),
                iter_time,
            )

            results = {
                "framework": "art",
                "dataset": "new_math_ops",
                "dataset_path": str(args.dataset),
                "prompt_version": PROMPT_VERSION,
                "base_model": args.base_model,
                "model_name": args.model_name,
                "num_iterations": iteration + 1,
                "total_planned_iterations": args.iterations,
                "group_size": args.group_size,
                "tasks_per_iter": args.tasks_per_iter,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_ratio": args.train_ratio,
                "split_seed": args.split_seed,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "concurrency": args.concurrency,
                "reward_history": reward_history,
                "accuracy_history": accuracy_history,
                "timing_history": timing_history,
                "rollout_history": rollout_history,
                "total_time_seconds": elapsed,
                "total_rollouts": sum(rollout_history),
                "final_mean_reward": mean_reward,
                "final_accuracy": accuracy,
            }
            (args.output_dir / "training_results.json").write_text(
                json.dumps(results, indent=2) + "\n",
                encoding="utf-8",
            )

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("New Math Ops Training Complete")
        print(f"{'=' * 60}")
        print(f"Iterations: {args.iterations}")
        print(f"Total rollouts: {sum(rollout_history)}")
        print(f"Reward curve: {[f'{reward:.3f}' for reward in reward_history]}")
        print(f"Accuracy curve: {[f'{acc:.1%}' for acc in accuracy_history]}")
        print(f"Time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
        print(f"{'=' * 60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
