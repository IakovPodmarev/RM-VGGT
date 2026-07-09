# E00 Baseline Reproduction

## Experiment ID and name

`E00_baseline_reproduction`

## Research question

Can we establish a trustworthy VGGT baseline in this repository before any recurrent-memory changes are introduced?

This experiment answers two narrower questions:

1. Can the released VGGT code path be run locally in a controlled way without architecture changes?
2. Can we define a baseline reference point that future RM-VGGT experiments must preserve and compare against?

## Hypothesis

The current repository should be sufficient to establish a usable baseline if we keep the architecture unchanged and separate baseline work into controlled stages:

- inference sanity
- training-pipeline sanity
- baseline reference definition

We expect the codebase to already provide the core components needed for:
- image/data loading
- model forward
- camera/depth prediction
- existing loss computation
- training-loop reuse
- smoke-level validation

## Minimal architectural change

None.

This experiment must not:
- add memory tokens
- add recurrent state
- add segmented wrappers
- modify head structure
- modify token layout
- modify losses

Any implementation work under E00 must be limited to:
- config additions
- smoke/eval entry points if strictly necessary
- baseline tests
- baseline documentation

## Scientific purpose

This is a baseline-establishment experiment.

Its purpose is:
- to create the comparison anchor for all future RM-VGGT experiments
- to prevent uncontrolled architectural drift
- to identify which existing VGGT components can be reused unchanged

This experiment is not intended to test a new scientific modeling idea.

## Data flow

The forward path must remain the released VGGT path.

### Inference / training forward

`images: [B, S, 3, H, W]`
-> `aggregator(images)`
-> `aggregated_tokens_list: list[(B, S, P, 2C)]`
-> `camera_head(aggregated_tokens_list)`
-> `depth_head(aggregated_tokens_list, images, patch_start_idx)`
-> optional point head / track head depending on config

### Camera path

The camera head reads token index `0` from the final aggregated token representation for each frame.

### Depth path

The depth head uses patch tokens starting at `patch_start_idx`.

### Baseline invariants

Any E00 implementation must preserve:
- token ordering
- `patch_start_idx` semantics
- head input conventions
- existing loss inputs and outputs

## Memory/state behavior

None.

This baseline must be strictly non-recurrent.

Required invariant:

- `memory.enabled = false` if any future memory config surface already exists

There is no:
- memory carry
- state reset logic
- detach logic
- recurrent wrapper
- replay backprop

## Baseline stages

E00 is split conceptually into three stages.

### E00a: inference baseline

Goal:
- verify pretrained VGGT inference works locally
- verify outputs are structurally sane

Checks:
- camera outputs exist
- depth outputs exist
- tensor shapes are correct
- no NaN/Inf in standard inference smoke

### E00b: training-pipeline baseline

Goal:
- verify the training pipeline works locally
- preserve dataloading and loss code
- establish one forward/backward/update smoke path


### E00c: baseline reference definition

Goal:
- define the official local baseline config and protocol that future RM-VGGT experiments must compare against

This includes:
- selected dataset profile
- selected enabled branches
- selected freeze policy if applicable
- selected smoke and validation commands

## Training setup

For E00 generally, training setup must stay as close as possible to the released code path.

Allowed:
- reuse existing trainer path
- reuse existing dataloaders
- reuse existing batch preprocessing
- reuse existing losses

Not allowed:
- custom recurrent training loop
- memory-state scheduling
- architecture changes
- new auxiliary objectives

Because the current repo config may describe fine-tuning rather than full paper reproduction, E00 must explicitly document what kind of baseline is being established:
- released-code baseline
- local fine-tuning baseline
- not necessarily full paper-metric reproduction

## Loss