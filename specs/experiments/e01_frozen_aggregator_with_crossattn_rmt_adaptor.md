# E01 Frozen Aggregator with Cross-Attention RMT Adaptor

## Status

Approved design specification for the first concrete experiment, `E01a`.

The architecture, streaming boundary, full-BPTT training semantics, loss
aggregation, optimizer behavior, and dataset contract are fixed below. Slices
1–7 were implemented against the original three-segment, eight-frame schedule.
Slice 8 added configurable frame counts. A two-phase real-data smoke at five
frames per segment and three segments completed both updates, validation, and
checkpoint resume on one Tesla T4. Slice 8.1 corrected camera-label validity
and actual optimizer-step accounting. Its gradient audits and a two-update
float16 scale-256 training smoke are recorded below. That smoke closes the
single-T4 feasibility check. Slice 8.2 specifies single-node DDP; it is not
yet implemented. Offline evaluation and point-cloud visualization in slices 9–10
remain unimplemented.

## Objective

Test a small recurrent-memory extension around a completely frozen VGGT
aggregator. A trainable memory writer compresses information from each segment
into recurrent read/write memory, while separate trainable cross-attention
adaptors expose incoming memory to the existing camera and depth heads.

The active proof-of-concept profile targets:

- image resolution `518 x 518`
- `segment_frames = 5`
- `num_segments = 3`
- `episode_frames = 15`

Frame counts are configuration values, not model constants. For every profile,
`episode_frames = segment_frames * num_segments`. The existing pipeline schedule
validation is the authority for this relationship; implementation must reuse
it rather than introduce a parallel validation. The former three-segment,
eight-frame schedule (`episode_frames = 24`) remains a scale-up target, not a
passing criterion for the current five-frame, three-segment smoke. Comparisons
must record their exact schedule.
Existing interfaces that call the episode length `total_frames` may retain
that name, but its value is the derived `episode_frames`, not an independent
setting.

The current E01a YAML defaults to five frames per segment and three segments.
The earlier 15-frame smoke selected five frames by override while the YAML
still defaulted to three. Record the exact invocation and resolved configuration
for every run; do not pool results from different schedules.

## Architectural scope

In scope:

- a completely frozen VGGT aggregator
- an external recurrent memory state
- distinct camera-oriented and patch-oriented memory tokens
- a trainable memory writer
- a camera read adaptor
- a depth read adaptor on the latest aggregator feature level
- the existing camera and DPT heads as trainable prediction heads
- ordered configurable-segment sequence orchestration
- independent preprocessing and normalization of each arriving segment
- full backpropagation through every inter-segment recurrent boundary
- camera and depth supervision on every segment
- a sequential VKITTI adapter and memory-disabled control
- optional single-node DDP after the slice 8.1 trial

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
- multi-node training or cross-node validation; single-node DDP is specified
  separately in slice 8.2

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
episode. It is never carried across scenes, samples, batch chunks, validation
examples, or optimizer steps. Batch elements must preserve their identity and
order across all configured segments.

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
[B, segment_frames, 37, 37, 2048]
    -> adaptive average pooling
[B, segment_frames, 16, 16, 2048]
    -> flatten spatial and frame axes
[B, segment_frames * 256, 2048]
    -> projection
[B, segment_frames * 256, 512]
```

The full `37 x 37` patch grid remains available to DPT. Writer pooling must not
alter DPT's spatial features.

### Writer position treatment

E01a uses position information relative to the current segment, not an
absolute segment-number embedding:

- each camera context token receives a learned within-segment temporal
  embedding for its position `0..segment_frames-1`
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
operations. This avoids making the per-frame camera tokens compete in a
single softmax with `segment_frames * 256` pooled patch tokens.

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

The keep-gate output projection weight is initialized to zero and its bias to approximately
-0.708, giving an initial keep probability of 0.33. This biases the recurrent state toward the
current segment: initially, each update retains roughly one-third of the previous memory and
writes roughly two-thirds of the new candidate. The model can still learn longer retention
where useful. Camera and patch slots use the same initialization but produce independent per-
token gate values.

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
[B, segment_frames, 1369, 2048] -> [B, segment_frames, 1369, 2048]
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

The implementation should preserve four explicit boundaries:

```python
normalize_segment(raw_segment_batch) -> normalized_segment_batch

encode_segment(images) -> (cached_features, patch_start_idx)

forward_segment(
    cached_features,
    images,
    patch_start_idx,
    read_memory,
) -> (predictions, next_memory, diagnostics)

forward(images, read_memory) -> (predictions, next_memory, diagnostics)
```

`normalize_segment` is a data/trainer concern and operates on CPU tensors from
one segment only. `encode_segment` owns the frozen-aggregator boundary.
`forward_segment` owns all trainable E01 modules and must not call backward or
the optimizer. The model's standard `forward` method composes `encode_segment`
and `forward_segment` for exactly one already-normalized segment; incoming
memory remains an explicit required argument.

The recurrent sequence orchestrator does not accept raw frames and does not
split, normalize, or transfer data. It consumes exactly `num_segments` ordered,
independently normalized segment mappings yielded by the outer episode
pipeline. Each yielded segment contains `images` shaped
`[B, segment_frames, 3, H, W]` on its final device and in its final dtype. The
orchestrator creates the learned initial memory once, invokes the model through
its standard module call for each segment in order, and returns
`num_segments` prediction and diagnostic dictionaries plus memory states
`m[0]` through `m[num_segments]`. It must not calculate losses, call backward,
or perform optimizer operations.

Production preprocessing yields prepared segments just in time. Unit tests may
materialize prepared mappings in a list, but the training path must not
normalize or copy future segments to the accelerator before segment 0
is processed. Training and loss-aggregation policy remain outside the segment
model and sequence orchestrator.

Camera and depth branch inputs are separate views of the cached feature list.
The camera branch replaces only the final camera-token features with its
adapted values. The depth branch replaces only layer-23 patch-token features
with its adapted values. Neither branch mutates the shared frozen cache in
place.

## Streaming and normalization contract

An E01a sample is an ordered episode of `episode_frames` raw frames divided into
`num_segments` consecutive, non-overlapping segments of `segment_frames` frames.
The active five-frame, three-segment profile is:

```text
segment 0: raw frames  0..4
segment 1: raw frames  5..9
segment 2: raw frames 10..14
```

The split occurs before `normalize_camera_extrinsics_and_points_batch` and
before device transfer or model execution. Frame-local file decoding,
deterministic resize, and crop may occur in the dataset before the split because
they do not share information across frames. Any operation using multiple
frames or data-derived sequence statistics occurs only after the current
segment has been isolated. Each segment is normalized independently when it
arrives:

1. slice all frame-indexed fields to the current `segment_frames` frames
2. use that segment's first camera as its coordinate origin
3. compute the average-distance scale only from valid points in that segment
4. transform that segment's extrinsics, camera points, world points, and depths
5. copy only that normalized segment to the accelerator
6. run the frozen aggregator and trainable E01 segment model
7. carry only `m[t+1]` into the next iteration

The aggregator's fixed ImageNet RGB mean/std transform is also applied only
when that segment enters `Aggregator.forward`; it does not create a
cross-segment statistic.

Consequently, the segments generally use different coordinate frames and
scales. This is intentional: E01a models an online stream, and recurrent memory
must tolerate the coordinate reset without receiving a future-derived
alignment transform. Training losses use each segment's normalized frame. An
offline evaluator may invert each segment's recorded normalization to express
all predictions in the raw VKITTI frame, then align the complete episode as
specified below. No such conversion or alignment enters model execution.

The outer episode pipeline may receive all `episode_frames` raw frames from the current
dataloader in one CPU batch for compatibility. It splits that CPU episode
first, then prepares and yields only the current segment to the recurrent
orchestrator. Splitting, segment-local normalization, and device transfer stay
outside the orchestrator and occur just in time. A test must prove that
changing frames `segment_frames..episode_frames-1` cannot change normalized
data, features, predictions, or loss for segment 0.

## Dataset protocol

The first real-data E01a profile uses VKITTI because it provides ordered video
frames, camera parameters, and depth supervision. The inherited random-nearby
sampling behavior is not valid for this experiment.

The E01a sequential adapter must:

- sample exactly `episode_frames` consecutive frames from one VKITTI scene, variation, and
  camera stream
- keep frame IDs strictly increasing with stride `1`
- disallow duplicated or randomly permuted frame IDs
- choose complete `episode_frames` windows at a configured positive start
  stride within each contiguous scene/variation/camera run; the active profile
  uses a five-frame stride and 15-frame episodes, so adjacent episodes overlap
  by ten frames
- use every eligible window in each training and validation epoch; incomplete
  trailing fragments do not form episodes
- preserve the same scene/variation/camera identity across all segments
- return frame IDs and segment indices for audit logging
- disable tracking data and the point prediction branch

The fixed scene-disjoint split is:

- training: `Scene01`, `Scene06`, `Scene18`, and `Scene20`
- validation: `Scene02`

All weather/lighting variations and both camera streams may be used, but a
single episode cannot cross a variation or camera boundary. Episode length
comes from the configured segment schedule, and start stride is a separate
configurable source setting. The active profile uses stride five for both
training and validation.
Training shuffles eligible episodes each epoch; validation uses fixed start indices, fixed order,
and no random augmentation. Frames may appear in multiple episodes within
the same split, so validation losses are episode-weighted and adjacent results
are correlated. The old Scene20 validation split is superseded by this
Scene02 split; runs using the two splits must be reported separately.
The initial E01a profile also disables training-time random scale, color,
grayscale, blur, orientation, and frame-order augmentation so that recurrence is the controlled change.
Deterministic resize/crop to `518 x 518` and its corresponding intrinsics update
remain enabled.

Before the real-data run, the sequential adapter must pass a small real-sample
inspection that verifies image/depth shapes, strictly ordered IDs, finite
camera/depth targets, and the independent segment normalizations.

## Slice 8: real-data training smoke test

Use the configured sequential VKITTI source and production model on a real
`episode_frames`-frame episode. The active smoke profile has five frames per
segment and three segments. Execute
the ordered segment preparation, forward,
loss, full-BPTT backward, gradient clipping, both AdamW parameter groups,
scheduler, and one optimizer update. Save a checkpoint, reload it, and complete
one further update. Keep the run small; it establishes pipeline correctness,
not convergence or scientific quality. Use offline logging and record the
exact data/checkpoint configuration, elapsed time, and peak GPU memory.

The first 24-frame `3 x 8` attempt ran out of GPU memory during the depth-head
forward pass before backward or optimizer update. The failing segment was not
identified. The active smoke must record the segment index and allocated/peak
GPU memory at segment entry and before the depth head. A resumed attempt at
six frames per segment ran out of GPU memory in the final segment's depth-head
forward after restoring AdamW state. The five-frame, three-segment two-phase
run completed; its evidence is
`/var/tmp/rm-vggt-slice08-3x5-20261004-01/summary.json` on `isp_tesla`. It
establishes two-update pipeline feasibility at 15 frames, subject to review.
It does not establish that longer schedules fit or pass.

Acceptance requires strictly consecutive frame IDs, the expected shapes and
segment order, finite losses and gradients, nonzero reachable gradients and
parameter updates for the writer, both read adaptors, and enabled heads, and
unchanged frozen aggregator parameters. Verify both parameter-group memberships
and learning rates. The resumed update must use restored trainable model,
optimizer, scheduler, and scaler state. A missing dataset root, pretrained
checkpoint, or suitable GPU is reported as an unmet prerequisite rather than
as a successful smoke run. The one-sample overfit check remains a separate
acceptance criterion for the later controlled experiment.

## Slice 9: offline Sim(3)-aligned evaluation

Scientific pose and point metrics are computed by a separate offline evaluator
from a saved checkpoint and fixed VKITTI validation episodes. They are not
computed inside training, its loss-only validation loop, or checkpoint
selection. The evaluator runs inference without gradients and records episode
identity, scene/variation/camera, exact frame IDs, predictions, required
geometry metadata, per-episode results, and aggregate results. Its production
entry point and symbols use capability-based names.

Training normalizes each configured segment independently. Before alignment,
decode each predicted pose into OpenCV camera-from-world extrinsics, derive
camera centers, and convert predicted poses and depth-derived world points from
their segment-local coordinates to the raw VKITTI coordinate frame by inverting
that segment's known first-camera transform and valid-point scale. Record or
recompute those normalization values from the corresponding raw segment; do
not estimate them from predictions. Use predicted depth, predicted camera
geometry, and the corresponding image pixel/intrinsic convention to derive
predicted 3D points. Check identity, shape, frame order, finite values, and
coordinate conventions before fitting an alignment.

Fit exactly one reflection-free Sim(3), with positive scale, from the
`episode_frames` predicted camera centers to their corresponding ground-truth
camera centers. Apply that
same transform to every predicted camera center, orientation/pose, and
depth-derived point in the episode. Do not fit a separate alignment per
segment or for the point cloud. A degenerate camera trajectory or failed fit
is an explicit evaluation failure, not a silent change of alignment policy.
No ATE or point RMSE is computed before this episode-level alignment.

Translation ATE is the root mean square Euclidean distance between aligned
predicted and ground-truth camera centers over the corresponding episode frames.
Point error is the Euclidean distance between the aligned predicted and
ground-truth 3D points at the same valid frame/pixel location. Point RMSE is
the square root of the mean squared point error across all valid
correspondences in the episode. Report valid-point counts; exclude invalid or
nonfinite points consistently and fail clearly when none remain. This is a
pixel-correspondence metric, not nearest-neighbor cloud distance. Aggregate
episode results without silently replacing an episode-weighted mean with a
point-weighted global RMSE; label any additional aggregate explicitly.

Focused tests cover one nondegenerate perfect prediction (zero error), a
known similarity transform, known residual errors, masks, empty valid sets,
degenerate trajectories, frame-identity mismatches, segment reconstruction,
and reuse of the one pose-derived alignment for points. A small real VKITTI
checkpoint evaluation must produce finite metrics before this slice closes.

## Slice 10: point-cloud visualization

Build a separate visualization entry point on the existing point-cloud/GLB
utilities. Consume the same fixed-episode inference artifacts and coordinate
conversion as the offline evaluator. Reassemble frames in temporal order,
derive world points from predicted depth and camera geometry, and color them
from corresponding RGB pixels. Support valid-point and optional confidence
filtering, predicted and ground-truth geometry, and camera trajectories.
Preserve a documented common coordinate frame for each exported scene; if an
evaluation-aligned overlay is requested, use the episode's single fitted
Sim(3) for both predicted cameras and points. An unaligned prediction may be
shown in its own frame, clearly labeled. Do not let an existing viewer's
automatic scene realignment silently change the stated frame.

Export a headless, nonempty GLB or other standard point-cloud artifact using
existing project utilities where practical. Rendering and optional viewer
dependencies stay outside the training path. Tests cover segment/frame order,
point-to-color correspondence, masks and filtering, coordinate transforms,
output counts, and headless export. Inspect one exported real VKITTI episode
before this slice closes.

## Full-BPTT sequence training

E01a uses full backpropagation through time across all configured segments. This is
the only accepted training mode for the first experiment.

For incoming memory `m[t]`, frozen current features `f[t]`, prediction function
`P`, and writer transition `W`:

```text
pred[t] = P(f[t], m[t])
m[t+1]  = W(f[t], m[t])
```

The trainable graph remains connected through every inter-segment memory state.
No state is detached, cloned into a new leaf, converted through NumPy, or
updated in place. The equally weighted segment losses are accumulated into one
scalar and backward is called once:

```text
L_sequence = sum(L[t] for t in 0..num_segments-1) / num_segments
```

This makes the gradient entering each memory equal to the sum of its local
read-path contribution and all reachable future-segment contributions. In
particular:

- `L[0]` trains the segment-0 read adaptors and heads but not the segment-0
  writer through its unused output
- `L[1]` trains the segment-0 writer through `m[1]`
- `L[2]` trains the segment-0 and segment-1 writers through `m[2]`
- the final writer output `m[num_segments]` is returned for interface
  consistency but has no future-loss contribution in a fixed-length episode

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

Camera supervision follows camera-label validity, independently of depth and
point masks. The sequential VKITTI adaptor guarantees a matched, finite camera
label for every emitted frame, so all its frames contribute. Other adaptors may
supply a frame-aligned boolean `camera_valid_mask`; when absent, the adaptor
must guarantee complete valid labels. Marked-valid targets must be finite and
have positive focal lengths. Camera loss averages over the valid frames and
raises if none remain in a segment, before sequence-loss aggregation. Segment
normalization also requires a valid first camera and at least one valid point
somewhere in the segment for its point-derived scale; an empty point set is an
explicit error. These are eligibility failures, not silent zero-loss segments.
The general dataset-adaptor requirement is in `specs/01_target_architecture.md`.

Each `L[t]` is the segment-local `objective` returned by `MultitaskLoss`. Equal
weighting is deliberate because all segments contain `segment_frames` frames.
Data-driven depth filtering remains segment-local; it must not compute a
quantile over all `episode_frames` frames.

Training logs both the arithmetic mean and each segment's individual camera,
depth, and objective terms. Segment 0 is reported separately as the no-history
control position; recurrent improvement is expected, if present, in later
segments.

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
- optimizer update: one attempt after the complete episode backward; a skipped
  float16-scaled attempt is not a completed update
- training budget: 20 epochs, traversing every eligible training episode
  in each epoch
- validation: every epoch over every eligible fixed validation episode
- update schedule: derive its planned update horizon from the training
  manifest size and epoch count; retain explicit small episode limits for
  smoke and diagnostic runs only
- checkpoint: every epoch plus the lowest aggregate validation-objective model

The ordinary trainer detects whether `GradScaler.step` invoked AdamW. On an
overflow skip it updates the scaler, restores the prior learning rates, logs a
skipped attempt, and leaves `completed_updates` and scheduler progress unchanged.
The next episode retries the same update position. Checkpoints store actual
completed updates and skipped-attempt counts; older checkpoints lacking the
latter restore it as zero. Every eligible episode is still attempted once per
epoch in a full run, so skips may leave fewer completed updates than scheduled
at the end of that run.

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
- identical configured streaming schedule (currently three segments of five frames)
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

The following criteria define eventual experiment acceptance. The later
one-sample overfit and review follow-ups are not prerequisites for starting the
exploratory training run described above:

1. Config loading uses configurable `segment_frames` and `num_segments`, with
   `episode_frames` equal to their product under the existing pipeline
   validation, and `backprop_mode=full`. The active smoke resolves to three
   segments of five frames and 15 episode frames.
2. The sequential adapter returns `episode_frames` strictly increasing,
   non-duplicated frame IDs from one sequence.
3. Each segment's normalized first extrinsic is identity up to tolerance and
   each scale is computed from that segment alone.
4. A future-isolation test shows that modifying any later segment cannot
   change segment-0 normalized tensors or forward outputs.
5. Forward shapes match the memory, camera, depth, and patch-grid contracts.
6. Each inter-segment memory state retains autograd history during training;
   no detach occurs at its boundary.
7. A loss using only the final segment produces nonzero gradients in writer
   operations executed for earlier segments.
8. Aggregator parameters remain unchanged, have `requires_grad=False`, and have
   no gradients after backward.
9. Camera/depth heads, both adaptors, writer, learned initial memory, and
   residual/keep gates receive finite gradients where reachable.
10. One complete configured episode performs exactly one backward and one
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

## Slice 8.1: precision evidence and single-T4 feasibility

The camera-validity and actual-optimizer-step fixes are implemented. Their
targeted audit reported 169 E01 tests passing; the historical record is
`/var/tmp/rm-vggt-slice81-audit/` on `tesla`. At `3 x 5`, three scale-1
float16 AdamW updates gave nonzero current depth-adaptor gradients in 23/41,
3/41, and 1/41 parameter tensors. AdamW movement does not establish a current
gradient. The initial full-precision probe exhausted T4 memory before
backward.

At `3 x 4`, a later diagnostic used three distinct deterministic VKITTI
episodes with matched weights and frame IDs. All nine full-precision backwards
completed in fresh processes without AdamW moments. At initialization and
after two scale-1 updates, scale-256 float16 probes had finite gradients in
every trainable module and nonzero gradients in all 40 depth adaptor body
tensors on all nine episode/snapshot pairs. Body-gradient cosines against
full precision were at least 0.993; relative L2 errors were 0.0085–0.1197.
Scale 4096 was nonfinite in eight of nine pairs. These were no-update probes;
`3 x 4` and `3 x 5` results are different schedules. Evidence is in
`/var/tmp/rm-vggt-depth-scale-20261007-01/` on `tesla`.

A subsequent `3 x 5` T4 smoke with float16 initial loss scale 256 completed
two actual AdamW updates, one after checkpoint resume, with no skips. Both
updates had nonzero current gradients in all 40 depth adaptor body tensors.
The scaler stayed at 256; PyTorch peak allocations were 12,639,475,200 and
14,828,217,344 bytes. This run used the former Scene20 validation split;
the record is
`/var/tmp/rm-vggt-e01a-fp16-scale256-3x5-20261008-01/summary.json`.
It establishes two-update feasibility, not sustained learning.

The earlier proposal for a separate 20-attempt single-T4 trial is
superseded by the successful two-update smoke. Further gradient stability
must be monitored during DDP smoke and exploratory training, including
actual AdamW calls and skips, scaler changes, unscaled depth-adaptor body
norms and zero fractions, clipping, losses, and per-rank memory. Investigate
repeated body-zero gradients, gate-only gradients, repeated skips, OOM, or
nonfinite parameters as they occur. The old smoke used a different validation
split and cannot establish results for the current Scene02 split.

## Slice 8.2: single-node DDP on six Tesla T4 GPUs

After the slice 8.1 feasibility smoke, add optional one-process-per-GPU DDP. First test
two T4s, then all six. Each rank owns one complete ordered episode per
synchronized update, preserving three segments, full BPTT, and memory reset
between episodes. Use one DDP forward covering the complete episode and its
local sequence objective, followed by one backward. The frozen aggregator
remains frozen. DDP replicates model and AdamW state per GPU; it does not pool
memory. Record each rank's peak allocation. An OOM at `3 x 5` fails that
profile and requires a separately labeled design decision.

Shuffle eligible training windows deterministically once per epoch using a
shared seed. Form global groups of at most `world_size` distinct episodes
and assign each real episode to exactly one rank. For a final group with
`R < world_size` real episodes, inactive ranks execute a finite padded
episode with zero objective contribution so all ranks join the same
collectives. Multiply each real rank's local objective by
`world_size / R`; DDP's rank average then yields the mean over the `R`
real episodes. Do not count padding in losses, metrics, or episode totals.
A global update is one synchronized optimizer attempt with effective batch
`R`, normally six. Derive the planned horizon from
`ceil(eligible_train_episodes / world_size) * epochs`, or record an explicit
bounded-run horizon. Label comparisons by global batch and update schedule.

All ranks must agree on gradient finiteness, whether AdamW ran, scaler state,
completed-update count, and scheduler position. A float16 overflow skip
advances none of their optimizers or schedulers. With BF16 or FP32 and disabled
scaling, reject nonfinite gradients before any rank updates. Shard fixed
validation episodes without counting padding, then reduce loss sums and
counts to the same episode-weighted global mean as single-process validation.
Rank zero writes the checkpoint; all ranks restore identical model, optimizer,
scaler, and scheduler states plus rank-specific RNG and data-position state
for exact continuation. Record per-rank failures and shut down cleanly.

Focused checks cover two-rank gradient equivalence to the mean of independent
episode gradients, full recurrent gradient flow, a partial final group,
nonfinite/skip consensus, deterministic disjoint sharding, global validation
aggregation, and resume. Then run a real `3 x 5` float16 scale-256 smoke using the current scene
split on two T4s and six T4s, each with a complete update, validation, and
fresh-process resume. Record actual optimizer calls, unique episodes, losses,
depth-body gradients, scaler changes, and peak VRAM per rank. Once the
six-rank smoke passes, start the exploratory training run with these checks
active; a separate 20-attempt single-T4 run is not a prerequisite. Keep the
single-T4 path as a regression check. Treat DDP throughput and learning as
separate measurements from single-GPU runs.

## Implementation sequence

Implementation proceeds in small verified slices:

1. add the E01a config surface and sequential segment/preprocessing contract
2. add `MemoryWriter` with shape, gate, and recurrence tests
3. add camera and depth read adaptors with residual and layout tests
4. compose the frozen-aggregator segment model
5. add the configurable-segment full-BPTT orchestrator and cross-segment gradient tests
6. integrate the existing loss, optimizer groups, logging, and checkpoint path
7. add the sequential VKITTI adapter and real-sample inspection
8. make frame counts configurable, then run the real VKITTI end-to-end
   training and checkpoint-resume smoke test at the active five-frame,
   three-segment schedule
8.1. correct camera-label eligibility and skipped-update accounting, audit
     depth-adaptor gradients, and establish two-update single-T4 feasibility
     at float16 scale 256
8.2. add single-node DDP with two-rank checks and a six-T4 end-to-end smoke
9. add and test offline episode-level Sim(3)-aligned ATE and point RMSE
10. add and test point-cloud visualization based on existing utilities
11. run one-sample overfit, then the controlled multi-seed experiment

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
- Every segment defines its own first-camera coordinate frame and
  valid-point scale without future-segment information.
- Segment order is fixed and only recurrent memory crosses a segment boundary.
- Memory resets at independent episode boundaries.
- Full BPTT spans every inter-segment memory boundary; none is detached.
- All segment objectives have equal weight and produce one sequence-level
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
- The final transition to `m[num_segments]` receives no future-loss supervision
  in a fixed-length episode.
- Full BPTT retains trainable camera/DPT/writer/adaptor activations for all
  configured segments and may exceed the target GPU budget at `518 x 518`.
- The original slice 8.1 scale-1 audit reached only the depth residual
  gate on its final `3 x 5` episode. Matched `3 x 4` scale-256 probes
  restored finite body gradients, and two later `3 x 5` scale-256 AdamW
  updates kept all 40 body tensors nonzero. Sustained training, zero-element
  fractions, later overflow behavior, and the first operation causing
  scale-1 gradient loss remain open.
- The initial `0.1` adaptor residual gates trade exact baseline equivalence for
  immediate gradient flow into the new memory path.
- A scene-disjoint VKITTI split has few validation scenes, so multi-seed
  variance and per-sequence results are required.
- Resetting temporal position embeddings at every segment provides no explicit
  absolute time index; all longer-range ordering must be represented in memory.

## Experiment naming

- Experiment family: `E01_frozen_aggregator_with_crossattn_rmt_adaptor`
- First concrete configuration: `E01a`
- `E01a` means full episode BPTT with the data, normalization, loss, and
  optimizer contracts in this document. Its active feasibility profile is
  five frames per segment and three segments; the eight-frame-per-segment
  scale-up remains unverified.
- Frame schedule is a logged configuration dimension. Results from different
  schedules must be labeled and evaluated separately, never pooled as one
  controlled comparison. Memory replay backpropagation, truncated BPTT, or a
  different coordinate normalization policy requires a subsequent experiment
  identifier such as `E01b`; none is defined here.
