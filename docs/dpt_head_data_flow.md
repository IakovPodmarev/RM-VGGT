# DPT Head Data Flow

This document explains the data flow of [`DPTHead`](../vggt/heads/dpt_head.py#L21-L483) in this repository.

It covers:

- what `DPTHead` takes as input
- how it pulls patch tokens from `aggregated_tokens_list`
- how it reconstructs 2D feature maps from token sequences
- how it fuses multiple backbone stages
- how it returns either dense features or dense predictions plus confidence

This is the same head class used for:

- depth prediction
- point-map prediction
- track-feature extraction when `feature_only=True`

## 1. Where `DPTHead` is used

At the VGGT model level, [`VGGT.__init__`](../vggt/models/vggt.py#L22-L27) creates:

- one `DPTHead` for point maps
- one `DPTHead` for depth maps

And [`TrackHead`](../vggt/heads/track_head.py#L48-L57) creates another `DPTHead` in `feature_only=True` mode for tracking features.

So there are three practical uses of the same core module:

- dense depth output
- dense world-point output
- dense intermediate feature output

## 2. What `DPTHead` expects

The main entry point is [`DPTHead.forward`](../vggt/heads/dpt_head.py#L115-L170).

Inputs:

- `aggregated_tokens_list`: list of token tensors from the aggregator
- `images`: original input images with shape `(B, S, 3, H, W)`
- `patch_start_idx`: index where patch tokens begin in the aggregator token sequence
- optional `frames_chunk_size`

The key thing to understand is that `DPTHead` does **not** operate on raw images directly. It operates on the aggregator’s token outputs, using `images` only to recover spatial dimensions and target resolution.

## 3. Constructor configuration

Important constructor fields are set in [`DPTHead.__init__`](../vggt/heads/dpt_head.py#L43-L113):

- `patch_size`: patch size used by the upstream tokenizer
- `output_dim`: number of final output channels
- `activation`: activation applied to the prediction channels
- `conf_activation`: activation applied to the confidence channel
- `features`: fused feature width used inside the DPT decoder
- `out_channels`: per-stage projection widths
- `intermediate_layer_idx`: which backbone stages to use
- `pos_embed`: whether to inject additional 2D positional embeddings
- `feature_only`: whether to stop at fused features instead of producing predictions
- `down_ratio`: output downscaling factor

By default, the head uses four backbone stages:

- [`intermediate_layer_idx = [4, 11, 17, 23]`](../vggt/heads/dpt_head.py#L51-L64)

So this DPT decoder is explicitly multi-scale. It does not use only the last backbone layer.

## 4. High-level flow

The internal flow is:

```text
aggregated_tokens_list
  -> choose four backbone stages
  -> remove special tokens
  -> reshape patch tokens back to 2D maps
  -> per-stage projection
  -> per-stage resize to aligned scales
  -> multi-scale fusion
  -> final upsampling
  -> optional extra pos embedding
  -> feature_only ? return features : predict outputs + confidence
```

## 5. Frame chunking

[`DPTHead.forward`](../vggt/heads/dpt_head.py#L115-L170) supports processing frames in chunks.

If:

- `frames_chunk_size is None`, or
- `frames_chunk_size >= S`

then the whole sequence is processed at once:

- [`return self._forward_impl(...)`](../vggt/heads/dpt_head.py#L139-L141)

Otherwise:

1. frames are sliced into chunks
2. each chunk is decoded independently
3. outputs are concatenated back along the sequence dimension

Relevant code:

- chunk loop: [`for frames_start_idx in range(0, S, frames_chunk_size)`](../vggt/heads/dpt_head.py#L150-L164)
- concat results: [`torch.cat(..., dim=1)`](../vggt/heads/dpt_head.py#L166-L170)

This is a memory-management feature. It does not change the head logic.

## 6. Choosing the patch tokens

The real work begins in [`_forward_impl`](../vggt/heads/dpt_head.py#L172-L247).

For each selected backbone layer, the head first removes special tokens:

- [`x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]`](../vggt/heads/dpt_head.py#L205-L206)

This matters because aggregator tokens are laid out as:

- camera token
- aggregator register tokens
- patch tokens

`patch_start_idx` tells `DPTHead` where the patch grid starts, so it ignores the non-spatial special tokens and keeps only the patch-token sequence.

## 7. Optional frame slicing

If the head is processing only a frame chunk, it slices the sequence dimension:

- [`x = x[:, frames_start_idx:frames_end_idx]`](../vggt/heads/dpt_head.py#L208-L210)

This keeps the layer features aligned with the chunked `images` tensor.

## 8. Turning token sequences back into feature maps

This is the most important shape conversion in the head.

After chunking, the code does:

- [`x = x.reshape(B * S, -1, x.shape[-1])`](../vggt/heads/dpt_head.py#L212-L212)
- [`x = self.norm(x)`](../vggt/heads/dpt_head.py#L214-L214)
- [`x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))`](../vggt/heads/dpt_head.py#L216-L216)

Where:

- `patch_h = H // self.patch_size`
- `patch_w = W // self.patch_size`

from:

- [`patch_h, patch_w = H // self.patch_size, W // self.patch_size`](../vggt/heads/dpt_head.py#L198-L200)

So the sequence goes:

- patch tokens: `(B, S, N, C)`
- flatten scenes/frames: `(B*S, N, C)`
- transpose channels forward
- reshape into a spatial map: `(B*S, C, patch_h, patch_w)`

This is how `DPTHead` reconstructs a 2D grid from transformer tokens.

## 9. Per-stage projection and resize

Each of the four selected backbone maps is then processed by:

1. a `1x1` channel projection
2. optional positional embedding
3. a resize layer

Code:

- projection: [`x = self.projects[dpt_idx](x)`](../vggt/heads/dpt_head.py#L218-L218)
- positional embedding: [`x = self._apply_pos_embed(x, W, H)`](../vggt/heads/dpt_head.py#L219-L220)
- resize: [`x = self.resize_layers[dpt_idx](x)`](../vggt/heads/dpt_head.py#L221-L221)

The projection layers are created here:

- [`self.projects = nn.ModuleList([... Conv2d(kernel_size=1) ...])`](../vggt/heads/dpt_head.py#L68-L71)

The resize layers are:

- stage 1: `ConvTranspose2d(... stride=4)` for large upsampling
- stage 2: `ConvTranspose2d(... stride=2)` for moderate upsampling
- stage 3: `Identity()` for no resize
- stage 4: `Conv2d(... stride=2)` for downsampling

Defined in:

- [`self.resize_layers`](../vggt/heads/dpt_head.py#L73-L87)

### Why the resize layers are different

The selected backbone stages correspond to different semantic depths but they all start from the same patch grid resolution. The DPT decoder reshapes them into a pyramid by resizing them differently, so later fusion combines:

- one relatively fine map
- one medium map
- one base map
- one coarser map

This is how the token hierarchy is turned into a multi-scale image hierarchy.

## 10. Additional positional embedding inside DPT

If `pos_embed=True`, `DPTHead` adds its own 2D sinusoidal-style positional embedding to feature maps.

This happens in:

- [`_apply_pos_embed`](../vggt/heads/dpt_head.py#L249-L259)

The steps are:

1. build a normalized UV grid with [`create_uv_grid`](../vggt/heads/utils.py#L66-L109)
2. convert the grid into channel embeddings with [`position_grid_to_embed`](../vggt/heads/utils.py#L11-L33)
3. scale it by `ratio`
4. broadcast it over the batch
5. add it to the feature map

Code references:

- UV grid creation: [`create_uv_grid(...)`](../vggt/heads/dpt_head.py#L255-L256)
- UV-to-embedding: [`position_grid_to_embed(...)`](../vggt/heads/dpt_head.py#L255-L257)

This is extra spatial conditioning inside the DPT decoder itself, separate from the upstream transformer’s positional handling.

## 11. Collecting the four stage maps

Inside the loop, each processed map is appended:

- [`out.append(x)`](../vggt/heads/dpt_head.py#L223-L223)

After all four selected backbone stages are processed:

- `out` is a list of four aligned feature maps

Those maps are then fused by:

- [`out = self.scratch_forward(out)`](../vggt/heads/dpt_head.py#L226-L227)

## 12. The `scratch` modules

The DPT decoder builds a set of conv adapters in:

- [`self.scratch = _make_scratch(out_channels, features, expand=False)`](../vggt/heads/dpt_head.py#L89-L89)

`_make_scratch` creates four `3x3` convs:

- [`scratch.layer1_rn`](../vggt/heads/dpt_head.py#L328-L340)
- [`scratch.layer2_rn`](../vggt/heads/dpt_head.py#L331-L333)
- [`scratch.layer3_rn`](../vggt/heads/dpt_head.py#L334-L336)
- [`scratch.layer4_rn`](../vggt/heads/dpt_head.py#L337-L340)

These adapt all four stage maps into the common decoder width `features`.

The head also adds four fusion blocks:

- [`self.scratch.refinenet1`](../vggt/heads/dpt_head.py#L93-L96)
- [`self.scratch.refinenet2`](../vggt/heads/dpt_head.py#L93-L96)
- [`self.scratch.refinenet3`](../vggt/heads/dpt_head.py#L93-L96)
- [`self.scratch.refinenet4`](../vggt/heads/dpt_head.py#L93-L96)

## 13. Multi-scale fusion in `scratch_forward`

Fusion happens in [`scratch_forward`](../vggt/heads/dpt_head.py#L261-L291).

The process is:

1. adapt each stage map with `layer*_rn`
2. start from the deepest/coarsest stage
3. repeatedly fuse with the next higher-resolution stage
4. upsample during each fusion step
5. finish with `output_conv1`

Code flow:

- initial adapted maps: [`layer_1_rn` ... `layer_4_rn`](../vggt/heads/dpt_head.py#L273-L276)
- deepest stage start: [`out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])`](../vggt/heads/dpt_head.py#L278-L278)
- fuse next level: [`self.scratch.refinenet3(out, layer_3_rn, ...)`](../vggt/heads/dpt_head.py#L281-L281)
- fuse next level: [`self.scratch.refinenet2(out, layer_2_rn, ...)`](../vggt/heads/dpt_head.py#L284-L284)
- fuse finest level: [`self.scratch.refinenet1(out, layer_1_rn)`](../vggt/heads/dpt_head.py#L287-L287)
- final conv: [`out = self.scratch.output_conv1(out)`](../vggt/heads/dpt_head.py#L290-L290)

This is the decoder’s main “coarse-to-fine” path.

## 14. What a `FeatureFusionBlock` does

Fusion is implemented by [`FeatureFusionBlock`](../vggt/heads/dpt_head.py#L389-L456).

Its forward path is:

1. start from the current coarse feature
2. if residual input exists, run it through `ResidualConvUnit` and add it
3. run another `ResidualConvUnit`
4. upsample
5. apply a `1x1` output conv

Code:

- optional residual add: [`output = self.skip_add.add(output, res)`](../vggt/heads/dpt_head.py#L440-L442)
- second residual unit: [`output = self.resConfUnit2(output)`](../vggt/heads/dpt_head.py#L444-L444)
- interpolate: [`output = custom_interpolate(...)`](../vggt/heads/dpt_head.py#L453-L453)
- output conv: [`output = self.out_conv(output)`](../vggt/heads/dpt_head.py#L454-L454)

So each fusion block both:

- merges information from another scale
- upsamples toward a finer resolution

## 15. Why `custom_interpolate` exists

Upsampling uses [`custom_interpolate`](../vggt/heads/dpt_head.py#L459-L484) instead of calling `F.interpolate` directly everywhere.

Its purpose is:

- avoid interpolation failures when the tensor would exceed an internal `INT_MAX` threshold

If the tensor is too large, it:

1. splits the batch into chunks
2. interpolates chunk by chunk
3. concatenates the results back

Otherwise it just calls `nn.functional.interpolate` directly.

## 16. Final resizing to target output resolution

After `scratch_forward`, the fused feature map is resized to:

- `(patch_h * patch_size / down_ratio, patch_w * patch_size / down_ratio)`

Code:

- [`out = custom_interpolate(...)](../vggt/heads/dpt_head.py#L228-L234)

With the common default `down_ratio=1`, this returns to approximately the original image resolution `(H, W)`.

If `down_ratio=2`, the output is half-resolution.

That is exactly what the tracking feature extractor wants:

- `TrackHead` builds `DPTHead(... feature_only=True, down_ratio=2, pos_embed=False)` in [`track_head.py`](../vggt/heads/track_head.py#L48-L57)

## 17. `feature_only=True` mode

If `feature_only=True`, the head stops after fused features:

- [`if self.feature_only: return out.view(B, S, *out.shape[1:])`](../vggt/heads/dpt_head.py#L239-L240)

So the output shape is:

- `(B, S, C_feat, H_out, W_out)`

This mode is used by the track head as a dense feature extractor rather than a prediction head.

## 18. Prediction mode

If `feature_only=False`, the head continues with:

- [`out = self.scratch.output_conv2(out)`](../vggt/heads/dpt_head.py#L242-L242)
- [`preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)`](../vggt/heads/dpt_head.py#L243-L243)

`output_conv2` is built as:

- `Conv2d -> ReLU -> Conv2d`

in:

- [`self.scratch.output_conv2 = nn.Sequential(...)`](../vggt/heads/dpt_head.py#L109-L113)

So the fused decoder feature map is converted into:

- `output_dim` channels total

Then `activate_head` interprets:

- the first `output_dim - 1` channels as prediction channels
- the last channel as confidence

See:

- [`activate_head`](../vggt/heads/head_act.py#L61-L112)

## 19. How predictions and confidence are split

Inside [`activate_head`](../vggt/heads/head_act.py#L61-L112):

1. channels are moved to the end
2. prediction channels are separated from the confidence channel
3. the prediction channels are activated according to `activation`
4. the confidence channel is activated according to `conf_activation`

Code:

- channel-last reorder: [`fmap = out.permute(0, 2, 3, 1)`](../vggt/heads/head_act.py#L73-L74)
- split: [`xyz = fmap[:, :, :, :-1]`, `conf = fmap[:, :, :, -1]`](../vggt/heads/head_act.py#L76-L78)

Prediction activation depends on the branch:

- depth head uses `activation="exp"`
- point head uses `activation="inv_log"`

Confidence usually uses:

- `conf_activation="expp1"`

which gives:

- [`conf_out = 1 + conf.exp()`](../vggt/heads/head_act.py#L103-L104)

## 20. Branch-specific meanings

The same `DPTHead` class means different things depending on `output_dim` and `activation`.

### 20.1 Depth head

In [`VGGT.__init__`](../vggt/models/vggt.py#L24-L27), depth uses:

- `output_dim=2`
- `activation="exp"`

So after `activate_head`:

- first channel = depth
- last channel = depth confidence

Final shapes:

- `depth`: `(B, S, H, W, 1)`
- `depth_conf`: `(B, S, H, W)`

### 20.2 Point head

Point maps use:

- `output_dim=4`
- `activation="inv_log"`

So after `activate_head`:

- first 3 channels = 3D point values
- last channel = point confidence

Final shapes:

- `world_points`: `(B, S, H, W, 3)`
- `world_points_conf`: `(B, S, H, W)`

### 20.3 Track-feature head

Track features use:

- `feature_only=True`
- `down_ratio=2`
- `pos_embed=False`

So they return decoder features, not predictions.

## 21. Final reshape back to `(B, S, ...)`

After prediction/confidence are computed, the head restores batch and sequence axes:

- [`preds = preds.view(B, S, *preds.shape[1:])`](../vggt/heads/dpt_head.py#L245-L245)
- [`conf = conf.view(B, S, *conf.shape[1:])`](../vggt/heads/dpt_head.py#L246-L246)

This is the final output contract for the rest of VGGT.

## 22. Shape summary

For a standard input:

- `images`: `(B, S, 3, H, W)`
- selected backbone tokens at one stage: `(B, S, N, C)`

the main shape flow is:

| Stage | Shape |
| --- | --- |
| Patch-token slice from one backbone layer | `(B, S, N, C)` |
| Flatten batch and sequence | `(B*S, N, C)` |
| Rebuild 2D feature map | `(B*S, C, patch_h, patch_w)` |
| After per-stage projection/resize | `(B*S, C_i, H_i, W_i)` |
| After fusion | `(B*S, features, H_f, W_f)` |
| After final resize | `(B*S, features, H_out, W_out)` |
| `feature_only=True` output | `(B, S, features, H_out, W_out)` |
| prediction channels before activation | `(B*S, output_dim, H_out, W_out)` |
| prediction output | `(B, S, H_out, W_out, output_dim-1)` |
| confidence output | `(B, S, H_out, W_out)` |

Where:

- `patch_h = H // patch_size`
- `patch_w = W // patch_size`

## 23. End-to-end summary

The DPT head is a dense decoder on top of the aggregator token pyramid.

It works by:

1. selecting four aggregator stages
2. dropping the non-spatial special tokens
3. reshaping patch tokens back into 2D maps
4. projecting and resizing those maps into a multi-scale pyramid
5. fusing coarse and fine information through a DPT-style decoder
6. either:
   - returning fused dense features, or
   - predicting dense outputs plus confidence

So although the backbone representation is token-based, `DPTHead` is the part that turns it back into image-aligned dense predictions.
