# Devlogs

## 2026-03-06 - New Math Ops RLVR Phase 1+2

- Added New Math Ops integration modules under `src/new_math_ops/`:
  - dataset loader + deterministic train/val split
  - prompt contract (`v2`)
  - final-answer parsing + binary reward helpers
  - eval artifact writer + metrics computation
  - baseline/candidate comparison utilities
- Added runnable scripts:
  - `run_new_math_ops_training.py`
  - `run_new_math_ops_eval.py`
  - `run_new_math_ops_compare.py`
- Added unit tests for dataset/reward/prompts/eval/compare behavior.
- Scope kept to phase 1 and phase 2 implementation only.
