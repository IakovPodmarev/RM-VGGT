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

## Experiment naming convention

Experiment identifiers such as `E00`, `E01`, and `E01a` may appear only in:

- specifications under `specs/`
- experiment configuration files and values under `training/config/`
- tests under `tests/`, including test file names, test names, fixtures, and
  assertions

Production code must use capability-based names that remain meaningful outside
one experiment. Experiment identifiers must not appear in production source
file or directory names, module names, class or function names, variables,
comments, docstrings, error messages, or runtime branching logic.

For example:

- allowed test: `tests/training/test_e01a_memory_writer.py`
- allowed config: `training/config/e01a_frozen_aggregator_streaming.yaml`
- preferred production module: `vggt/recurrent_memory/segment_model.py`
- preferred production class: `RecurrentMemorySegmentModel`
- prohibited production names: `e01a_segment_model.py`, `E01aSegmentModel`, or
  `if "e01a" in config`

Experiment-specific behavior must be selected through generic capability
settings in configuration rather than hard-coded experiment-name checks.

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
