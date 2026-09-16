# 04 Engineering Stack

## Purpose

Fix the project-wide engineering conventions so experiments remain consistent.

## Required stack

- environment and dependencies: `pixi`
- CLI: `argparse`
- config: Hydra / OmegaConf
- tests: `pytest`
- experiment logging: Weights & Biases

Do not introduce new infrastructure unless clearly necessary.

## Configuration rules

Every experiment must be config-controlled.

Required properties:
- baseline-preserving defaults
- explicit memory enable/disable flag
- explicit freeze policy
- explicit branch enable/disable settings
- explicit segment/window settings when recurrence exists

## CLI rules

Runnable entry points should:
- use argparse
- remain pixi-compatible
- avoid ad hoc shell-only workflows
- keep commands reproducible and documented

## Testing rules

Each experiment should specify the relevant subset of:
- config-loading test
- forward-pass smoke test
- shape test
- backward/gradient-flow test
- baseline-equivalence test
- memory reset/detach test
- checkpoint-load test

## Logging rules

Every serious run should log:
- experiment ID
- git commit
- config
- trainable parameter count
- memory settings
- losses
- metrics
- runtime
- peak VRAM
- qualitative outputs where relevant
- failure cases

## Code-change rules

Preferred:
- small, scoped changes
- minimal surface area
- disabled-by-default new features
- no unrelated refactors during experiment implementation

Avoid:
- broad trainer rewrites
- broad head rewrites without accepted spec
- hidden behavioral changes without config gates

## Review rules

Review should prioritize:
- correctness
- baseline preservation
- spec compliance
- gradient routing correctness
- missing tests
- architectural drift
