# E01 Frozen Aggregator with Cross-Attention RMT Adaptor

## Status

Approved design specification for the first concrete experiment, `E01a`.

The architecture, streaming boundary, full-BPTT training semantics, loss
aggregation, optimizer behavior, dataset contract, controls, validation, and
implementation sequence are fixed below. Implementation has not started.

## Objective

Test a small recurrent-memory extension around a completely frozen VGGT
aggregator. A trainable memory writer compresses information from each segment
into recurrent read/write memory, while separate trainable cross-attention
adaptors expose incoming memory to the existing camera and depth heads.

The initial proof of concept targets:

- image resolution `518 x 518`
- segment length of 8 frames
- total sequence length of 24 frames (three segments)

## Architectural scope

In scope:

- a completely frozen VGGT aggregator
- an external recurrent memory state
- distinct camera-oriented and patch-oriented memory tokens
- a trainable memory writer
- a camera read adaptor
- a depth read adaptor on the latest aggregator feature level
- the existing camera and DPT heads as trainable prediction heads
- ordered three-segment sequence orchestration
- independent preprocessing and normalization of each arriving segment
- full backpropagation through both inter-segment recurrent boundaries
- camera and depth supervision on every segment
- a sequential VKITTI adapter and memory-disabled control

Out of scope for `E01a`:

- memory replay backpropagation
- truncated backpropagation through time or any inter-segment detach
- segment-level activation replay or checkpointing as a replacement for full
  BPTT
- attention masks
- Transformer-XL key/value caches
- generation mode
- LoRA or unfreezing aggregator layers
- point and tracking heads
- carrying memory between different scenes or dataloader samples
- reconstructing one globally aligned trajectory from independently normalized
  segment predictions
- multi-node or multi-GPU validation; E01a is established on one training
  process before distributed support is considered

## High-level data flow

```text
raw ordered segment t, received without access to future segments
        |
        v
segment-local preprocessing and geometric normalization
        |
        |   (the first camera and valid points of this segment alone define
        |    the segment coordinate frame and scale)
        v
normalized images and targets for segment t
        |
        v
completely frozen VGGT aggregator
        |
        |   (no gradient crosses back into the aggregator;
        |    every downstream module is trainable)
        |
        |-- camera feature tensors
        |       + incoming memory m[t]
        |       -> trainable Camera Read Adaptor
        |       -> trainable Camera Head
        |
        |-- patch feature tensors from cached layers
        |       + incoming memory m[t]
        |       -> trainable Depth Read Adaptor on layer 23 only
        |       -> trainable DPT Head
        |
        `-- layer-23 camera and patch features
                -> spatial compression for writer context
                -> trainable Memory Writer(read=m[t], current segment context)
                -> outgoing memory m[t+1]
```

Prediction heads for segment `t` read only incoming memory `m[t]`. The writer's
output `m[t+1]` is reserved for later segments and is not used to predict the
current segment.

The stream order is strict: split first, then normalize and process one segment
at a time. No normalization statistic, coordinate transform, feature, target,
or prediction from segment `t+1` may be used while processing segment `t`.

## Frozen aggregator contract

The entire aggregator is frozen, including:

- patch embedder
- camera token parameters
- register token parameters
- frame-attention blocks
- global-attention blocks
- their normalization and MLP parameters

The aggregator retains its original token layout and cached feature contract.
No recurrent memory tokens are inserted into the aggregator itself in E01a.

"Frozen aggregator" describes the parameter and gradient boundary, not a
restriction on how its output tensors may be transformed. Aggregator outputs
are detached from the aggregator graph and then consumed by fully trainable
projections, attention layers, adaptors, writer modules, and prediction heads.
Gradients update those downstream modules but never cross the boundary back
into the aggregator.

The frozen aggregator stays in evaluation mode even while downstream modules
train. Each arriving segment is encoded under `torch.no_grad()`. E01a does not
use `torch.inference_mode()` because downstream trainable linear layers may
need to save the frozen feature tensors to compute parameter gradients. The
aggregator is absent from optimizer parameter groups, and its parameters must
have `requires_grad=False` and `grad is None` after backward.

Only cached layers `4`, `11`, `17`, and `23` are retained. Entries at other
layer indices remain `None`, preserving the current aggregator/DPT index
contract. Frozen features remain resident as long as required by the full-BPTT
graph; they are released after the sequence backward finishes.

Everything after the aggregator boundary that participates in E01a is
trainable:

- learned initial read memory, write queries, and memory type embeddings
- writer camera/patch context projections
- all writer self-attention and cross-attention layers
- writer final MLP and recurrent keep gate
- all camera read-adaptor parameters
- all depth read-adaptor parameters
- the complete camera head
- the complete DPT depth head

The point and tracking heads are disabled and are not part of the E01a model.

For the default model and `518 x 518` images, every frame has a `37 x 37` patch
grid, or 1,369 patch tokens. The aggregator's cached frame/global features have
width `2C = 2048` with the default `C = 1024`.

## Recurrent memory representation

The recurrent state has shape:

```text
m[t]: [B, 16, 512]
```

It is partitioned into two typed groups:

```text
camera read memory: 8 tokens x 512 channels
patch read memory:  8 tokens x 512 channels
```

The writer likewise begins from:

```text
camera write queries: 8 learned tokens x 512 channels
patch write queries:  8 learned tokens x 512 channels
```

Camera and patch memory tokens have distinct learned token/type embeddings.
The two groups provide an inductive bias rather than a hard information
boundary: all write queries may consume both camera and patch contexts, and
both read adaptors may consume all 16 incoming memory tokens.

The first segment uses a learned initial read-memory bank with the same typed
partition and shape.

Memory is reset to that learned initial bank at the start of every independent
24-frame sequence. It is never carried across scenes, samples, batch chunks,
validation examples, or optimizer steps. Batch elements must preserve their
identity and order across all three segments.

## Writer inputs

The memory writer uses only features from the latest aggregator layer, layer
`23`.

### Current camera context

The layer-23 camera feature produced by the frozen aggregator for every frame
is projected by a trainable projection from the aggregator feature width into
the 512-channel memory width.

### Current patch context

Layer-23 patch tokens are restored to their per-frame `37 x 37` spatial grid.
Adaptive average pooling reduces this grid to `16 x 16` for the writer only:

```text
[B, 8, 37, 37, 2048]
    -> adaptive average pooling
[B, 8, 16, 16, 2048]
    -> flatten spatial and frame axes
[B, 2048, 2048]
    -> projection
[B, 2048, 512]
```

The full `37 x 37` patch grid remains available to DPT. Writer pooling must not
alter DPT's spatial features.

### Writer position treatment

E01a uses position information relative to the current segment, not an
absolute segment-number embedding:

- the eight camera context tokens receive learned within-segment temporal
  embeddings for positions `0..7`
- every pooled patch token receives the sum of the same learned temporal
  embedding for its frame and a fixed two-dimensional sinusoidal embedding for
  its `16 x 16` pooled spatial location
- incoming memory and write queries receive learned type and slot embeddings

Temporal positions reset for every arriving segment. Recurrence, rather than
an absolute segment index, represents the order of an arbitrarily long stream.
No stochastic dropout is used in these embeddings or in E01a attention/MLP
blocks.

### Explicit connection to incoming read memory

Incoming read memory is provided to the writer as key/value context, not only
through a final residual addition:

```text
camera writer context = [current camera features, camera read-memory tokens]
patch writer context  = [current pooled patches, patch read-memory tokens]
```

The two contexts use separate input projections. Keeping current observations
and previous memory as distinct tokens allows attention to select between them
instead of irreversibly mixing them with elementwise addition.

Camera context and patch context are processed by separate cross-attention
operations. This avoids making the eight per-frame camera tokens compete in a
single softmax with approximately 2,048 pooled patch tokens.

## Memory writer

The writer operates on the concatenated 16 learned write queries at width 512.
It contains three attention blocks followed by one final MLP.

Every writer attention operation uses four heads of width 128, biased query,
key, value, and output projections, and attention dropout `0.0`. Each attention
sublayer has its own pre-normalization and output projection; parameters are
not shared across blocks or between camera and patch cross-attention.

Attention heads partition feature channels, not tokens: each of the four heads
attends to all 16 write queries and all keys in its context. Four heads preserve
the 512-channel attention width while reducing attention-map/softmax overhead
relative to eight heads; E01a has no evidence that eight separate writer
subspaces are needed for only 16 memory slots.

Each writer block uses pre-normalization and residual connections in this
order:

1. self-attention among all 16 write tokens
2. cross-attention from all write tokens to the camera writer context
3. cross-attention from all write tokens to the patch writer context

Both camera-oriented and patch-oriented write queries attend to both contexts.
Self-attention allows the typed memory groups to coordinate and exchange
information.

After the third block, a single position-wise MLP produces the candidate next
memory:

```text
candidate[t+1]: [B, 16, 512]
```

The final candidate MLP is `LayerNorm(512) -> Linear(512, 2048) -> GELU ->
Linear(2048, 512)` with no dropout. The per-token gate MLP is
`LayerNorm(1024) -> Linear(1024, 512) -> GELU -> Linear(512, 1)` over the
concatenated read/candidate pair.

### Gated recurrent update

Plain addition of candidate write tokens to read tokens is not used as the sole
state transition. Instead, the writer produces one scalar keep gate per memory
token:

```text
keep_gate: [B, 16, 1]
```

Conceptually:

```python
candidate = final_mlp(write_tokens)
keep_gate = sigmoid(gate_mlp(concat(read_memory, candidate)))
next_memory = keep_gate * read_memory + (1 - keep_gate) * candidate
```

This retains a direct path for useful information from previous segments while
allowing each memory slot to overwrite obsolete information. Camera slots are
gated against corresponding camera read slots, and patch slots against
corresponding patch read slots.

The keep-gate output projection weight is initialized to zero and its bias to
`+1.0`, giving an initial keep probability of approximately `0.33`. This
provides a stable identity path without making overmemorizing initially
negligible. Camera and patch slots use the same initialization but independent
per-token gate values.

## Camera read adaptor

The camera path has a dedicated three-block cross-attention adaptor.

- Queries: frozen final camera features, one per frame
- Keys/values: all 16 incoming read-memory tokens `m[t]`
- Internal width: 512
- Attention: 8 heads of width 64, with biased projections and dropout `0.0`
- Output: a residual update projected back to the camera head's expected width
- Consumer: the existing trainable camera head

Each adaptor block uses pre-normalization, cross-attention, and a residual
connection. Camera and depth adaptor parameters are not shared.

The adaptor output preserves the camera head's existing per-frame token shape.
Its projected update is multiplied by a learned scalar sigmoid gate initialized
to `0.1` (`logit = -2.1972246`) before residual addition. A small nonzero gate
keeps the initial perturbation limited while allowing gradients to reach memory
and adaptor parameters from the first optimization step.

## Depth read adaptor

The depth path has a separate three-block cross-attention adaptor.

- Queries: the full-resolution layer-23 patch features
- Keys/values: all 16 incoming read-memory tokens `m[t]`
- Internal width: 512
- Attention: 8 heads of width 64, with biased projections and dropout `0.0`
- Output: a residual update projected back to DPT's expected feature width
- Consumer: the existing trainable DPT head

Only cached aggregator layer `23` is adapted in E01a. Cached layers `4`, `11`,
and `17` remain unchanged and are passed directly to DPT.

The adaptor preserves the number and ordering of patch tokens:

```text
[B, 8, 1369, 2048] -> [B, 8, 1369, 2048]
```

Consequently, DPT can continue reshaping the patch tokens into its expected
`37 x 37` grid. No memory token is inserted into that spatial grid.

The depth adaptor uses the same learned scalar sigmoid residual gate and `0.1`
initial value as the camera adaptor. The two gates and all other adaptor
parameters are independent. All writer and adaptor attention/MLP dropout rates
are `0.0` in E01a.

## Component boundaries

The model architecture has four trainable components around the frozen
aggregator:

1. `MemoryWriter`
2. `CameraReadAdaptor`
3. `DepthReadAdaptor`
4. existing camera and DPT heads

The logical segment-level interface is:

```text
aggregator output features, incoming m[t]
    -> camera predictions
    -> depth predictions
    -> outgoing m[t+1]
```

The implementation should preserve three explicit boundaries:

```python
normalize_segment(raw_segment_batch) -> normalized_segment_batch

encode_segment(images) -> (cached_features, patch_start_idx)

forward_segment(
    cached_features,
    images,
    patch_start_idx,
    read_memory,
) -> (predictions, next_memory)
```

`normalize_segment` is a data/trainer concern and operates on CPU tensors from
one segment only. `encode_segment` owns the frozen-aggregator boundary.
`forward_segment` owns all trainable E01 modules and must not call backward or
the optimizer.

The sequence orchestrator accepts an ordered episode of three raw segments,
creates the learned initial memory, invokes these interfaces in order, and
returns the three prediction/loss dictionaries plus final memory for logging or
continued inference. Training policy remains outside the segment model.

Camera and depth branch inputs are separate views of the cached feature list.
The camera branch replaces only the final camera-token features with its
adapted values. The depth branch replaces only layer-23 patch-token features
with its adapted values. Neither branch mutates the shared frozen cache in
place.

## Streaming and normalization contract

An E01a sample is an ordered episode of exactly 24 raw frames divided into
three consecutive, non-overlapping segments:

```text
segment 0: raw frames  0..7
segment 1: raw frames  8..15
segment 2: raw frames 16..23
```

The split occurs before `normalize_camera_extrinsics_and_points_batch` and
before device transfer or model execution. Frame-local file decoding,
deterministic resize, and crop may occur in the dataset before the split because
they do not share information across frames. Any operation using multiple
frames or data-derived sequence statistics occurs only after the current
segment has been isolated. Each segment is normalized independently when it
arrives:

1. slice all frame-indexed fields to the current eight frames
2. use that segment's first camera as its coordinate origin
3. compute the average-distance scale only from valid points in that segment
4. transform that segment's extrinsics, camera points, world points, and depths
5. copy only that normalized segment to the accelerator
6. run the frozen aggregator and trainable E01 segment model
7. carry only `m[t+1]` into the next iteration

The aggregator's fixed ImageNet RGB mean/std transform is also applied only
when that segment enters `Aggregator.forward`; it does not create a
cross-segment statistic.

Consequently, the three segments generally use different coordinate frames and
scales. This is intentional: E01a models an online stream, and recurrent memory
must tolerate the coordinate reset without receiving a future-derived
alignment transform. Predictions from separate segments are evaluated in their
own normalized frames and are not directly concatenated into one trajectory.

The implementation may receive all 24 raw frames from the current dataloader
in one CPU batch for compatibility, but it must expose them to preprocessing
and the model through the ordered segment loop above. A test must prove that
changing frames `8..23` cannot change normalized data, features, predictions,
or loss for segment 0.

## Dataset protocol

The first real-data E01a profile uses VKITTI because it provides ordered video
frames, camera parameters, and depth supervision. The inherited random-nearby
sampling behavior is not valid for this experiment.

The E01a sequential adapter must:

- sample exactly 24 consecutive frames from one VKITTI scene, variation, and
  camera stream
- keep frame IDs strictly increasing with stride `1`
- disallow duplicated or randomly permuted frame IDs
- choose only start indices for which all 24 frames exist
- preserve the same scene/variation/camera identity across all three segments
- return frame IDs and segment indices for audit logging
- disable tracking data and the point prediction branch

The fixed scene-disjoint split is:

- training: `Scene01`, `Scene02`, `Scene06`, and `Scene18`
- validation: `Scene20`

All weather/lighting variations and both camera streams may be used, but a
single 24-frame sample cannot cross a variation or camera boundary. Validation
uses fixed start indices and no random augmentation. The initial E01a profile
also disables training-time random scale, color, grayscale, blur, orientation,
and frame-order augmentation so that recurrence is the controlled change.
Deterministic resize/crop to `518 x 518` and its corresponding intrinsics update
remain enabled.

Before the real-data run, the sequential adapter must pass a small real-sample
inspection that verifies image/depth shapes, strictly ordered IDs, finite
camera/depth targets, and the three independent segment normalizations.

## Full-BPTT sequence training

E01a uses full backpropagation through time across all three segments. This is
the only accepted training mode for the first experiment.

For incoming memory `m[t]`, frozen current features `f[t]`, prediction function
`P`, and writer transition `W`:

```text
pred[t] = P(f[t], m[t])
m[t+1]  = W(f[t], m[t])
```

The trainable graph remains connected through `m[1]` and `m[2]`. Neither state
is detached, cloned into a new leaf, converted through NumPy, or updated in
place. The three weighted segment losses are accumulated into one scalar and
backward is called once:

```text
L_sequence = (L[0] + L[1] + L[2]) / 3
```

This makes the gradient entering each memory equal to the sum of its local
read-path contribution and all reachable future-segment contributions. In
particular:

- `L[0]` trains the segment-0 read adaptors and heads but not the segment-0
  writer through its unused output
- `L[1]` trains the segment-0 writer through `m[1]`
- `L[2]` trains the segment-0 and segment-1 writers through `m[2]`
- the segment-2 writer output `m[3]` is returned for interface consistency but
  has no future-loss contribution in a fixed three-segment episode

The writer parameters are shared across segments, so their gradients accumulate
from every transition that affects a later loss. Detaching at every segment
would eliminate the intended training signal and is forbidden. MRBP, manual
per-segment backward calls, and optimizer steps between segments are also
forbidden in E01a.

## Loss contract

Every segment uses the existing `MultitaskLoss` after that segment's independent
normalization:

- camera loss enabled with weight `5.0` and `loss_type: l1`
- depth loss enabled with weight `1.0`, `gradient_loss_fn: grad`, and
  `valid_range: 0.98`
- point loss disabled
- track loss disabled
- no auxiliary memory, reconstruction, gate, or specialization loss

Each `L[t]` is the segment-local `objective` returned by `MultitaskLoss`. Equal
weighting is deliberate because all segments contain eight frames. Data-driven
depth filtering remains segment-local; it must not compute a quantile over all
24 frames.

Training logs both the arithmetic mean and each segment's individual camera,
depth, and objective terms. Segment 0 is reported separately as the no-history
control position; recurrent improvement is expected, if present, primarily in
segments 1 and 2.

## Optimization and precision

The accepted E01a defaults are:

- optimizer: `AdamW`
- memory writer and read-adaptor learning rate: `1e-4`
- existing camera and depth head learning rate: `5e-5`
- weight decay: `0.05`
- schedule: 5% linear warmup followed by cosine decay
- gradient clipping: global norm `1.0` for writer, adaptors, camera head, and
  depth head
- precision: bfloat16 autocast where supported, using the repository's scaler
  policy
- initial per-device episode batch size: `1`
- gradient accumulation: `1` until the one-episode path is validated; later
  changes must scale `L_sequence` once, not each segment independently
- optimizer update: exactly once after the complete three-segment backward
- training budget: 20 epochs with at most 800 training episodes per epoch
- validation: every epoch with at most 400 fixed validation episodes
- checkpoint: every epoch plus the lowest aggregate validation-objective model

The learned initial memory is a trainable parameter and belongs to the memory
parameter group. The frozen aggregator is excluded from all optimizer and
gradient-clipping groups. The initial smoke/overfit run uses one process and
does not establish distributed correctness.

Every serious run records the experiment ID, git commit, resolved config,
dataset split, random seed, frame IDs, trainable parameter count, memory/gate
settings, aggregate and per-segment losses, learning rates, gradient norms,
runtime, and peak allocated VRAM. Weights & Biases is the canonical experiment
record required by the project engineering stack; existing local TensorBoard
logging may remain as a secondary sink.

## Experimental controls

E01a requires a memory-disabled segmented control with the same:

- sequential VKITTI samples and scene split
- `3 x 8` streaming schedule
- independent per-segment normalization
- frozen aggregator
- trainable camera and depth heads
- losses, optimizer schedule, augmentation policy, and training budget

The control omits the writer and both read adaptors. It is not the E00 baseline:
E00 used a different freeze policy and did not establish real-data behavior.
Comparison with E00 may be reported for context but cannot isolate recurrence.

An optional diagnostic reset control may feed the learned `m[0]` to every
segment instead of carrying memory. It is not required to close E01a and must
be identified separately from the primary memory-disabled control.

## Validation and acceptance criteria

Implementation acceptance requires all of the following:

1. Config loading fixes `total_frames=24`, `segment_frames=8`,
   `num_segments=3`, and `backprop_mode=full`.
2. The sequential adapter returns 24 strictly increasing, non-duplicated frame
   IDs from one sequence.
3. Each segment's normalized first extrinsic is identity up to tolerance and
   each scale is computed from that segment alone.
4. A future-isolation test shows that modifying segments 1 or 2 cannot change
   segment-0 normalized tensors or forward outputs.
5. Forward shapes match the memory, camera, depth, and patch-grid contracts.
6. `m[1]` and `m[2]` retain autograd history during training; no detach occurs
   at their boundaries.
7. A loss using only segment 2 produces nonzero gradients in writer operations
   executed for segments 0 and 1.
8. Aggregator parameters remain unchanged, have `requires_grad=False`, and have
   no gradients after backward.
9. Camera/depth heads, both adaptors, writer, learned initial memory, and
   residual/keep gates receive finite gradients where reachable.
10. One complete 24-frame episode performs exactly one backward and one
    optimizer update without NaN or Inf.
11. Memory resets between independent episodes and validation samples.
12. Checkpoint save/load restores all trainable E01 parameters, optimizer,
    scheduler, and scaler state; recurrent memory itself is not checkpointed.
13. A deterministic one-sample overfit test decreases the sequence objective.
14. Peak VRAM and elapsed time are recorded for the full-BPTT smoke run.

Scientific evaluation reports the primary control and recurrent model with at
least three seeds, per-segment and aggregate losses/metrics, trainable parameter
counts, runtime, and peak VRAM. No improvement claim is made from the synthetic
fixture or one-sample overfit test.

## Implementation sequence

Implementation proceeds in small verified slices:

1. add the E01a config surface and sequential segment/preprocessing contract
2. add `MemoryWriter` with shape, gate, and recurrence tests
3. add camera and depth read adaptors with residual and layout tests
4. compose the frozen-aggregator segment model
5. add the three-segment full-BPTT orchestrator and cross-segment gradient tests
6. integrate the existing loss, optimizer groups, logging, and checkpoint path
7. add the sequential VKITTI adapter and real-sample inspection
8. run one-sample overfit, then the controlled multi-seed experiment

Each slice must report tests, observed shapes, trainable parameter counts, and
blockers before the next slice begins.

## Architectural invariants

E01a must preserve the following:

- The aggregator and every aggregator parameter remain unchanged.
- Memory exists outside the aggregator.
- Camera and depth use distinct read adaptors.
- Both read adaptors may read all 16 memory tokens.
- Current heads use only `m[t]`, never `m[t+1]`.
- Incoming camera memory is explicitly present in camera writer context.
- Incoming patch memory is explicitly present in patch writer context.
- All write queries can attend to both writer contexts.
- Writer pooling affects only writer context, never DPT input resolution.
- Only layer `23` is modified by the depth adaptor in E01a.
- DPT receives exactly 1,369 patch tokens per frame at `518 x 518` resolution.
- No LoRA or aggregator unfreezing is part of E01a.
- Raw data is split before geometric normalization, device transfer, and model
  execution.
- Every eight-frame segment defines its own first-camera coordinate frame and
  valid-point scale without future-segment information.
- Segment order is fixed and only recurrent memory crosses a segment boundary.
- Memory resets at independent episode boundaries.
- Full BPTT spans `m[1]` and `m[2]`; neither boundary is detached.
- All three segment objectives have equal weight and produce one sequence-level
  backward call and optimizer update.
- The model and its segment `forward` methods never call backward internally.

## Known architectural risks

- Typed memory groups may fail to specialize because both groups can access the
  same contexts and both heads can access all memory tokens.
- A `16 x 16` pooled grid retains much more spatial information than global
  pooling but still discards fine detail.
- Three adaptor blocks may be excessive for the first dataset size and could
  overfit.
- Updating only DPT layer `23` may provide insufficient memory influence after
  fusion with unchanged shallower features.
- A per-token scalar keep gate may be too coarse if different channels need
  different retention behavior.
- Independently re-centering and rescaling each segment removes a shared output
  coordinate system. Memory must learn useful latent information across these
  changes without being given an explicit inter-segment transform.
- The final transition to `m[3]` receives no future-loss supervision in a
  fixed-length three-segment episode.
- Full BPTT retains trainable camera/DPT/writer/adaptor activations for all three
  segments and may exceed the target GPU budget at `518 x 518`.
- The initial `0.1` adaptor residual gates trade exact baseline equivalence for
  immediate gradient flow into the new memory path.
- A scene-disjoint VKITTI split has few validation scenes, so multi-seed
  variance and per-sequence results are required.
- Resetting temporal position embeddings at every segment provides no explicit
  absolute time index; all longer-range ordering must be represented in memory.

## Experiment naming

- Experiment family: `E01_frozen_aggregator_with_crossattn_rmt_adaptor`
- First concrete configuration: `E01a`
- `E01a` means full three-segment BPTT with the data, normalization, loss, and
  optimizer contracts in this document.
- Memory replay backpropagation, truncated BPTT, a different coordinate
  normalization policy, or a different segment schedule requires a subsequent
  experiment identifier such as `E01b`; none is defined here.
