# Read-only review prompt for slice 8

Connect to `isp_tesla` and review `~/RM-VGGT` as a senior reviewer of
PyTorch training systems and sequential 3D data pipelines. This is a
read-only review. Do not edit, format, commit, reset, clean, or delete any
repository or run files, and do not launch another real-data training run.
You may run targeted, non-destructive tests if needed; state every command
and result. Preserve the worktree, including untracked files.

The repository is currently based on commit
`4d95078a1bce8c5710b80a2ebcf8a7c5bf3157f5` with an uncommitted slice 8
worktree. Recheck its current revision and status at review time. Inspect
`git diff` and all relevant untracked files: `training/smoke_recurrent_training.py`,
`training/smoke_diagnostics.py`, and `tests/e01/test_e01a_training_smoke.py`.
Read the authoritative specs under `specs/`, particularly
`specs/experiments/e01_frozen_aggregator_with_crossattn_rmt_adaptor.md` and
`specs/02_engineering_stack.md`. Do not assume the diff alone contains all
implementation or test changes.

The current design profile is five frames per segment, three segments, 15
ordered VKITTI frames, 518 × 518 images, a frozen aggregator, trainable memory
writer and read adaptors, camera/depth heads, segment-local normalization, and
one full-BPTT update per episode. The reported two-phase smoke on GPU 0 passed:
phase A saved `epoch_0000.pt`, phase B restored model/AdamW/scaler/RNG/progress,
completed a second audited update and validation, and saved `epoch_0001.pt`.
The reported phase B training peak was 14.83 GB on a Tesla T4. Only the E01
suite was reported run: 159 passed. Treat these as claims to verify, not as
proof supplied by the prompt. The report and checkpoints are under
`/var/tmp/rm-vggt-slice08-3x5-20261004-01/`; begin with `summary.json`.
Earlier artifacts include an eight-frame-per-segment forward OOM and a
six-frame-per-segment phase B depth-head OOM after optimizer-state restoration.

Your primary task is to find correctness and scientific-validity problems in
the code, tests, dataflow, and training lifecycle. Follow each operation
across module boundaries. Do not limit the review to syntax, style, or
individual function bugs. In particular:

1. Trace raw VKITTI indexing, scene/variation/camera grouping, consecutive
   frame IDs, deterministic validation starts, image/depth/camera loading,
   masks, resize/crop and intrinsics updates. Check that an episode cannot
   cross a stream boundary and that the configured episode length is used
   everywhere.
2. Trace CPU episode splitting, per-segment first-camera and valid-point
   normalization, device transfer, and model calls. Look for future-frame
   leakage, whole-episode statistics, accidental full-episode GPU transfer,
   wrong frame-indexed fields, aliasing, or batch identity changes. Examine
   both production and test paths.
3. Trace the frozen aggregator boundary and cached feature layout through
   camera/depth adaptors, writer, heads, and loss. Verify patch-grid ordering,
   `patch_start_idx`, depth `[B,S,H,W,1]` and confidence `[B,S,H,W]` layouts,
   shape assertions, dtype/autocast compatibility, absence of cache mutation,
   and that current predictions use incoming memory only.
4. Trace recurrence and gradients: learned initial memory reset per independent
   episode; all inter-segment states stay attached; later loss reaches earlier
   writer operations; final writer output is not falsely required to receive
   a future-loss gradient; all segment objectives contribute equally to one
   backward; no optimizer step or detach occurs between segments. Check that
   tests would catch violations rather than merely echo implementation.
5. Trace optimization: exact coverage and separation of the two AdamW groups,
   frozen aggregator exclusion, initial memory and gate membership, weight
   decay, scheduled rates and step order, unscale/finite-gradient checks,
   global clipping, one actual optimizer step, and AMP skipped-step detection.
   Distinguish genuine gradient-driven changes from weight decay alone.
6. Trace checkpoint save and fresh-process resume. Check model, AdamW moments
   and counters, scaler, RNG, schedule definition/progress, pretrained-versus-
   resume paths, transient memory exclusion, device placement, and whether
   phase B truly starts a new episode and performs a second update. Test the
   restored-state audit for false positives or accidental mutation.
7. Audit smoke evidence itself: dataset/checkpoint identities, resolved config,
   real frame IDs, train/validation scene split, actual parameters and tensor
   shapes, per-segment losses, gradient reachability, parameter changes,
   frozen-aggregator identity, elapsed time, CUDA peak accounting, and
   `epoch_0000.pt`/`epoch_0001.pt`. Determine what is measured versus only
   asserted. Check whether diagnostic hooks, hashes, or retained references
   alter memory enough to affect the near-capacity result or leak graphs.
8. Inspect test quality and omissions. Look for mocks or tiny fixtures being
   mistaken for production evidence, tests that assert the same logic they
   implement, missing negative cases, weak comparisons, hidden skips, order
   dependence, and missing baseline regressions. The E00 suite was not
   reported run; do not imply otherwise. Run a focused check only when it
   resolves a concrete concern and does not start expensive training.
9. Check design and configuration consistency. The updated spec calls five
   frames per segment the active profile, but the checked-in E01a YAML was
   observed with `segment_frames: 3`; the passing smoke used an override of
   five. Verify the current worktree and decide whether this discrepancy can
   cause ordinary E01a runs to use the wrong profile. Do not silently treat a
   15-frame pass as proof that the 24-frame scale-up fits. Check experiment
   naming, docstrings, and any scope drift from slices 9–11.

Search definitions, callers, tests, config, docs, and relevant git history
before concluding that a behavior is correct or unused. For each finding,
give severity, exact `file:line`, the triggering path or input, why it matters
to training or scientific interpretation, and a concrete fix direction.
Prioritize actionable defects and missing acceptance evidence. Separate
confirmed bugs from plausible risks requiring a targeted test. If no defects
are found, state what was actually verified and what remains unverified.

Lead your response with findings ordered by severity. Then give a compact
verdict on code readiness, 15-frame real-data smoke evidence, checkpoint
continuation, test adequacy, and the unverified 24-frame scale-up. Do not
implement fixes; hand findings back to the implementation chat and any spec
decisions to the design chat.
