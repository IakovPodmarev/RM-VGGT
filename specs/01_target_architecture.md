# Target Architecture

## Purpose

Define the full intended RM-VGGT architecture direction at a high level, without committing all details into a single experiment.

## High-level target

RM-VGGT should consist of:

1. a VGGT-derived core token-processing model
2. optional memory-token logic inside or around alternating attention
3. a recurrent wrapper that processes long image sequences in segments
4. existing or minimally adapted VGGT heads
5. preserved loss and data pipeline where possible

## Architectural intent

The target architecture should:
- preserve the baseline VGGT token flow as much as possible
- add memory in a controlled and configurable way
- support disabled-memory baseline-equivalent behavior
- make segment-level recurrence explicit rather than implicit

## Core module direction

The future core should support:
- frozen or partially trainable baseline token processing
- memory-token insertion at explicitly declared locations
- read/write state update
- output features compatible with existing heads where possible

Preferred early direction:
- memory conditions existing tokens rather than permanently changing the token layout consumed by dense heads

## Wrapper direction

The wrapper should:
- accept a full image sequence
- split it into windows or segments
- run the core per segment
- carry memory across segments
- optionally keep only one segment graph resident at a time
- define memory reset points clearly

## Head compatibility principle

Strong preference:
- preserve existing camera and depth heads early
- avoid changing dense-head patch-grid assumptions unless necessary
- treat point and track paths as optional/secondary during early RM work

## Trainability philosophy

Expected staged progression:
1. baseline pipeline reuse
2. partial fine-tuning with frozen patch feature extractor
3. memory-only or memory-focused training
4. broader fine-tuning only if needed

## Key architectural constraints

### Constraint 1: dense heads are layout-sensitive
Extra visible tokens can break patch-grid reconstruction.

### Constraint 2: alternating attention is the main intervention zone
Memory should be inserted relative to frame/global attention explicitly.

### Constraint 3: recurrence must be configurable
Need clear controls for:
- enabled/disabled
- segment length
- reset policy
- detach policy
- memory token count
- insertion point

## Full-picture architectural questions

The project must eventually answer:
- where memory is inserted
- whether memory is recurrent or non-recurrent
- whether memory is shared or per-layer
- whether memory should affect all heads or only selected outputs
- how long-range gradients should be handled
- when preserving head compatibility becomes too restrictive