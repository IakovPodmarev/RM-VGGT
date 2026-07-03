# VGGT Camera Data Flow

This document focuses only on the camera path in VGGT:

- how camera tokens are initialized inside the aggregator
- how those tokens move through the shared backbone
- how the camera head turns the final camera-token features into pose predictions through iterative refinement

All links below point to the actual implementation in this repository.

## 1. Where the camera path starts

At the top level, [`VGGT.forward`](../vggt/models/vggt.py#L61-L69) does:

1. run the shared [`Aggregator`](../vggt/models/aggregator.py#L25-L331)
2. pass the resulting token list into [`CameraHead`](../vggt/heads/camera_head.py#L19-L149)
3. keep the last camera prediction as `predictions["pose_enc"]`
4. keep all refinement-stage predictions as `predictions["pose_enc_list"]`

So the high-level camera path is:

```text
images
  -> aggregator
  -> aggregated_tokens_list
  -> camera_head
  -> pose_enc_list
  -> final pose_enc
```

## 2. Camera token initialization in the aggregator

The aggregator creates its own learned camera token here:

- [`self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))`](../vggt/models/aggregator.py#L125-L128)

It is then initialized with a small normal distribution:

- [`nn.init.normal_(self.camera_token, std=1e-6)`](../vggt/models/aggregator.py#L133-L135)

### 2.1 What the shape `(1, 2, 1, C)` means

For `camera_token`, the dimensions mean:

- first dimension `1`: broadcastable batch axis
- second dimension `2`: two learned versions
- third dimension `1`: one camera token per frame
- fourth dimension `C`: embedding size

The two versions are:

- slot `0`: camera token template for the first frame
- slot `1`: camera token template for all remaining frames

This does **not** mean each frame gets two camera tokens.
It means the model stores two candidate templates and chooses one per frame by expansion.

## 3. Expanding camera tokens across frames

The expansion happens in [`slice_expand_and_flatten`](../vggt/models/aggregator.py#L308-L330), called from:

- [`camera_token = slice_expand_and_flatten(self.camera_token, B, S)`](../vggt/models/aggregator.py#L212-L214)

That function does:

1. take slot `0` for the first frame
2. take slot `1` for every other frame
3. expand to `(B, S, 1, C)`
4. flatten to `(B*S, 1, C)`

Code references:

- slot-0 slice: [`query = token_tensor[:, 0:1, ...]`](../vggt/models/aggregator.py#L322-L323)
- slot-1 slice: [`others = token_tensor[:, 1:, ...]`](../vggt/models/aggregator.py#L324-L325)
- concatenate along frame axis: [`torch.cat([query, others], dim=1)`](../vggt/models/aggregator.py#L326-L327)
- flatten: [`combined.view(B * S, *combined.shape[2:])`](../vggt/models/aggregator.py#L329-L330)

For a sequence of `S=4` frames, the camera-token assignment is:

- frame 0 -> first-frame camera token
- frame 1 -> other-frames camera token
- frame 2 -> other-frames camera token
- frame 3 -> other-frames camera token

So each frame gets exactly one camera token, but frame 0 uses a different learned template from the rest.

## 4. Concatenating camera tokens with the rest of the frame tokens

After expansion, the aggregator concatenates:

- camera token
- aggregator register tokens
- patch tokens from the per-frame ViT encoder

This happens here:

- [`tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)`](../vggt/models/aggregator.py#L216-L217)

At this point, for each frame, token index layout is:

- index `0`: camera token
- following indices: aggregator register tokens
- remaining indices: patch tokens

This index convention matters because the camera head later reads token `0`.

## 5. Camera tokens inside the aggregator backbone

Once concatenated, the camera token is just part of the shared token sequence. It is processed by the same transformer blocks as all other tokens.

The aggregator alternates between:

- frame attention: [`_process_frame_attention`](../vggt/models/aggregator.py#L260-L282)
- global attention: [`_process_global_attention`](../vggt/models/aggregator.py#L284-L305)

### 5.1 What happens to the camera token during frame attention

In frame attention, tokens are processed with shape:

- `(B*S, P, C)`

where `P` is the number of tokens in one frame.

That means the camera token for one frame attends only to tokens from the same frame:

- its own frame’s patch tokens
- its own frame’s register tokens

It does **not** interact with other frames during this stage.

### 5.2 What happens during global attention

In global attention, the sequence is reshaped to:

- `(B, S*P, C)`

Now the camera token can interact across frames because all frame tokens are in one joint token sequence for the scene.

So the camera token evolves in two complementary ways:

- frame attention injects per-frame visual information
- global attention injects cross-frame multi-view information

### 5.3 Important inference detail

At inference, the camera token is **not reinitialized at each layer**.

The sequence is:

1. start from the learned camera-token parameters stored in the model
2. expand them once for the current batch/sequence
3. concatenate them with the other tokens
4. let transformer blocks update them layer by layer

So after block 1, block 2, ..., block N, the camera token is no longer the raw initialized parameter. It becomes a learned feature that has absorbed image and scene information through attention.

## 6. What the aggregator returns to the camera head

The aggregator saves one combined output per alternating-attention stage:

- [`concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)`](../vggt/models/aggregator.py#L250-L253)

So each entry in `aggregated_tokens_list` has shape:

- `(B, S, P, 2C)`

The final entry, `aggregated_tokens_list[-1]`, contains the final per-frame camera token features after all backbone processing.

## 7. Extracting the camera tokens in the camera head

The camera branch begins here:

- [`tokens = aggregated_tokens_list[-1]`](../vggt/heads/camera_head.py#L85-L86)
- [`pose_tokens = tokens[:, :, 0]`](../vggt/heads/camera_head.py#L88-L90)

This means:

- use only the last backbone output
- select token index `0`, which is the camera token

Shape change:

- before selection: `(B, S, P, 2C)`
- after selection: `(B, S, 2C)`

Then the camera tokens are normalized:

- [`pose_tokens = self.token_norm(pose_tokens)`](../vggt/heads/camera_head.py#L89-L90)

These normalized per-frame camera-token features are the visual input to the iterative pose-refinement loop.

## 8. Camera head iterative refinement logic

The refinement loop is implemented in [`CameraHead.trunk_fn`](../vggt/heads/camera_head.py#L95-L141).

This is the logic you asked about, expanded in code order.

### 8.1 Start from a learned empty pose token

The camera head stores:

- [`self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))`](../vggt/heads/camera_head.py#L62-L64)

On the first iteration:

- [`module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))`](../vggt/heads/camera_head.py#L110-L114)

So the first refinement step does **not** start from a previous predicted pose. It starts from a learned “empty pose” prior.

### 8.2 Embed the current pose estimate

The embedding layer is:

- [`self.embed_pose = nn.Linear(self.target_dim, dim_in)`](../vggt/heads/camera_head.py#L63-L64)

On later iterations:

- previous pose estimate is detached
- then projected into the same feature space as the visual camera tokens

Code:

- [`pred_pose_enc = pred_pose_enc.detach()`](../vggt/heads/camera_head.py#L115-L117)
- [`module_input = self.embed_pose(pred_pose_enc)`](../vggt/heads/camera_head.py#L116-L117)

So each iteration conditions on the current pose estimate.

### 8.3 Use the pose estimate to modulate the visual camera token

The model generates adaptive modulation parameters from `module_input`:

- [`shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)`](../vggt/heads/camera_head.py#L119-L120)

Those parameters are produced by:

- [`self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))`](../vggt/heads/camera_head.py#L66-L67)

Then the visual camera-token feature `pose_tokens` is modulated:

- [`pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)`](../vggt/heads/camera_head.py#L122-L123)
- [`pose_tokens_modulated = pose_tokens_modulated + pose_tokens`](../vggt/heads/camera_head.py#L123-L124)

Supporting pieces:

- adaptive norm without affine parameters: [`self.adaln_norm`](../vggt/heads/camera_head.py#L69-L70)
- modulation helper: [`modulate`](../vggt/heads/camera_head.py#L144-L149)

Interpretation:

- `pose_tokens` contains visual scene evidence from the aggregator
- `module_input` contains the current pose hypothesis
- adaptive modulation lets the current pose hypothesis condition how the visual evidence is processed next

## 9. Run the transformer trunk

After modulation, the conditioned camera-token features go through a short transformer trunk:

- [`pose_tokens_modulated = self.trunk(pose_tokens_modulated)`](../vggt/heads/camera_head.py#L126-L126)

The trunk is built as:

- [`self.trunk = nn.Sequential(*[Block(... ) for _ in range(trunk_depth)])`](../vggt/heads/camera_head.py#L50-L56)

Default:

- `trunk_depth = 4`

So each iteration refines the per-frame camera-token features with an additional small transformer that sits on top of the shared backbone.

## 10. Predict a pose delta

The trunk output is normalized and passed through an MLP branch:

- [`pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))`](../vggt/heads/camera_head.py#L127-L128)

Relevant modules:

- [`self.trunk_norm`](../vggt/heads/camera_head.py#L58-L60)
- [`self.pose_branch = Mlp(... out_features=self.target_dim ...)`](../vggt/heads/camera_head.py#L70-L71)

The output dimension is:

- `self.target_dim = 9` for `absT_quaR_FoV`: [`CameraHead.__init__`](../vggt/heads/camera_head.py#L40-L43)

So each iteration predicts a 9D delta for:

- translation `(3)`
- quaternion rotation `(4)`
- field of view `(2)`

## 11. Accumulate the delta into the pose estimate

If this is the first iteration:

- [`pred_pose_enc = pred_pose_enc_delta`](../vggt/heads/camera_head.py#L130-L131)

Otherwise:

- [`pred_pose_enc = pred_pose_enc + pred_pose_enc_delta`](../vggt/heads/camera_head.py#L132-L133)

So the model refines pose additively over iterations rather than predicting the full final pose from scratch each time.

## 12. Apply output activations

After accumulation, the pose encoding is activated with:

- [`activated_pose = activate_pose(...)`](../vggt/heads/camera_head.py#L135-L138)

The activation logic is in [`activate_pose`](../vggt/heads/head_act.py#L12-L35).

It splits the 9D pose encoding into:

- translation
- quaternion
- field-of-view terms

and applies per-part activations:

- translation: [`trans_act`](../vggt/heads/camera_head.py#L34-L35)
- quaternion: [`quat_act`](../vggt/heads/camera_head.py#L34-L35)
- field of view: [`fl_act="relu"` by default`](../vggt/heads/camera_head.py#L34-L36)

By default:

- translation stays linear
- quaternion stays linear
- field-of-view values are pushed positive with ReLU

## 13. Repeat

Each activated prediction is appended to:

- [`pred_pose_enc_list`](../vggt/heads/camera_head.py#L108-L109)
- append step: [`pred_pose_enc_list.append(activated_pose)`](../vggt/heads/camera_head.py#L139-L139)

Then the loop repeats, now using the current pose estimate as the next iteration’s conditioning input.

So the full per-iteration logic is:

```text
current pose estimate
  -> embed_pose
  -> poseLN_modulation
  -> adaptive modulation of visual camera token
  -> transformer trunk
  -> pose delta
  -> additive update
  -> activate pose fields
  -> next iteration
```

## 14. Why there are two different "camera tokens"

There are two different learned objects involved here, and they are easy to confuse.

### 14.1 Aggregator camera token

This is the token in [`Aggregator`](../vggt/models/aggregator.py#L125-L128):

- shape `(1, 2, 1, embed_dim)`
- used as a visual special token inside the backbone
- updated by frame/global attention together with patch tokens

This is the token whose final feature becomes `pose_tokens`.

### 14.2 Camera-head empty pose token

This is the token in [`CameraHead`](../vggt/heads/camera_head.py#L62-L64):

- shape `(1, 1, 9)`
- used only to seed the first iteration of pose refinement
- not part of the aggregator token sequence

So:

- aggregator camera token = visual latent token
- empty pose token = initialization for iterative pose decoding

## 15. End-to-end summary

The full camera path is:

1. The aggregator creates a learned first-frame camera token and a learned shared non-first-frame camera token.
2. It expands them across frames with [`slice_expand_and_flatten`](../vggt/models/aggregator.py#L308-L330).
3. It concatenates those camera tokens with register tokens and patch tokens.
4. Alternating frame/global attention updates those camera tokens layer by layer using both per-frame and cross-frame evidence.
5. The camera head extracts token `0` from the final aggregator output for every frame.
6. It normalizes those visual camera-token features.
7. It starts iterative refinement from a learned empty pose token.
8. At each iteration it embeds the current pose estimate, uses it to modulate the visual camera token, runs a transformer trunk, predicts a pose delta, adds the delta, activates the output, and repeats.
9. The last iteration becomes `predictions["pose_enc"]`, while all stages are kept in `predictions["pose_enc_list"]`.
