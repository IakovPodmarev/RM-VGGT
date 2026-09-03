# E00 Baseline Reproduction

## Experiment ID and name

`E00_baseline_reproduction`

## Status


E00 established a local, non-recurrent training-pipeline baseline using controlled synthetic tensors. It does not claim real-dataset validation, full pretrained-model reproduction, or reproduction of published VGGT metrics.

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

The accepted E00 training smoke uses the existing `MultitaskLoss` without modification:

- camera loss enabled
- depth loss enabled
- point loss disabled
- track loss disabled

The trainer receives the existing loss dictionary and backpropagates its `objective` value. E00 introduces no auxiliary loss and no memory-related objective.

## Accepted baseline configuration

The E00 reference configuration is `training/config/e00b_training_pipeline_one_sample.yaml` together with the controlled test model and synthetic sequence fixture in `tests/training/test_e00b_pipeline.py`.

The reference behavior is:

- memory and recurrence absent
- camera and depth branches enabled
- point and tracking branches disabled
- patch feature extractor frozen
- remaining aggregator, camera-head, and depth-head paths trainable
- one controlled batch passed through preprocessing, forward, loss, backward, and optimizer update

The small synthetic model used by the test is a pipeline contract, not a quality or throughput benchmark for the released full-size model.

## Validation state

The synthetic-tensor baseline was tested by the project owner. The accepted E00 evidence covers:

- Hydra configuration loading
- dataloader batch construction and expected tensor shapes
- baseline camera and depth prediction keys and shapes
- finite camera and depth losses
- backward propagation through frame attention, global attention, camera head, and depth head
- absence of gradients in the frozen patch feature extractor
- successful optimizer update
- disabled point and tracking heads

This is sufficient for the purpose of E00: confirming that the local non-recurrent training path is structurally usable before memory-specific work begins.

## Dataset-adapter decision

Real CO3D and VKITTI adapter validation is intentionally not part of E00 closure.

The current classes derived from `BaseDataset` are inherited transitional code. They will be rewritten as part of the next memory-token experiment.

Testing the current derived dataset classes would validate interfaces that are expected to be replaced and would not provide a durable acceptance criterion. Their implementation and validation are therefore deferred together.


## E00 outcome

E00 is closed as a **local training-pipeline baseline**, with the following interpretation:

- the original VGGT token layout and head interfaces remain unchanged
- the camera/depth loss and optimizer path can execute end to end on a controlled synthetic batch
- no recurrent state or memory mechanism is present
- no real-dataset, checkpoint-quality, or paper-metric claim is made
- concrete dataset-adapter design and real-data validation move to the next memory-token experiment
