# Project Goal

## Project name

RM-VGGT: Recurrent Memory VGGT

## Objective

Extend VGGT with recurrent-memory mechanisms in a controlled way, while preserving comparability to the original model and reusing as much of the existing VGGT training/evaluation stack as possible.

## Main scientific goal

Test whether recurrent memory can improve VGGT on long or segmented multi-view sequences by allowing information to persist across windows, without immediately resorting to full end-to-end redesign.

## Immediate engineering goal

Establish a trustworthy baseline and a reusable training pipeline before introducing memory-specific architectural changes.

## Scope

In scope:
- architecture specifications for recurrent-memory extensions of VGGT
- controlled experiments that isolate memory-related changes
- reuse of VGGT dataloading, preprocessing, heads, and losses where possible
- training and evaluation protocols that preserve baseline comparability
- memory-aware wrappers for segmented sequence processing

Out of scope for now:
- large-scale redesign of VGGT heads
- unrelated trainer refactors
- replacing Hydra / argparse / pytest / W&B / pixi stack
- introducing multiple memory ideas in one uncontrolled branch

## Success criteria

The project is successful if:
1. a stable non-recurrent baseline is established
2. recurrent-memory changes are introduced incrementally
3. each experiment is scientifically interpretable
4. memory-enabled models remain comparable to baseline VGGT
5. the codebase remains testable, configurable, and reviewable

## High-level research progression

The intended progression is:

1. understand and preserve the baseline VGGT data flow
2. validate the usable training pipeline
3. add memory-related hooks with baseline-preserving behavior
4. test non-recurrent memory-token variants
5. test recurrent memory-token variants with segment carry
6. evaluate tradeoffs in quality, stability, and compute
