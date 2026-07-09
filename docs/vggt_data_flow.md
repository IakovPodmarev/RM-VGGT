# VGGT Data Flow

This document explains how data moves through VGGT at inference time, from raw images to cameras, depth, point maps, and tracks. It focuses on the actual code path in this repository and links each stage to the implementation.

## 1. Entry points

Most users enter VGGT through one of these paths:

- [`load_and_preprocess_images`](../vggt/utils/load_fn.py#L97-L230) loads image files, converts them to RGB tensors in `[0, 1]`, resizes them, and batches them as `(S, 3, H, W)`.
- [`VGGT.forward`](../vggt/models/vggt.py#L29-L96) is the main model API. It accepts either `(S, 3, H, W)` or `(B, S, 3, H, W)`.
- [`demo_viser.py`](../demo_viser.py#L359-L377), [`demo_gradio.py`](../demo_gradio.py#L73-L97), and [`demo_colmap.py`](../demo_colmap.py#L74-L90) show the full inference pipeline in practice.

Important input conventions:

- `S` is the number of frames/views in one scene.
- `B` is batch size.
- `query_points`, when provided for tracking, are pixel coordinates shaped `(N, 2)` or `(B, N, 2)`.

Inside [`VGGT.forward`](../vggt/models/vggt.py#L54-L61), the model adds a batch dimension when needed and then sends the images into the shared backbone:

```text
images -> aggregator -> aggregated_tokens_list, patch_start_idx
                           |-> camera_head
                           |-> depth_head
                           |-> point_head
                           |-> track_head (only if query_points is provided)
```

## 2. Top-level model orchestration

The top-level module is [`VGGT`](../vggt/models/vggt.py#L17-L27).

Its constructor wires together:

- [`Aggregator`](../vggt/models/aggregator.py#L25-L331): shared multi-view token backbone
- [`CameraHead`](../vggt/heads/camera_head.py#L19-L149): predicts camera pose encoding
- [`DPTHead`](../vggt/heads/dpt_head.py#L21-L320): used twice
  - once for depth
  - once for point maps
- [`TrackHead`](../vggt/heads/track_head.py#L12-L104): predicts point tracks

In [`VGGT.forward`](../vggt/models/vggt.py#L61-L96), the outputs are assembled into a dictionary:

- `pose_enc`, `pose_enc_list`
- `depth`, `depth_conf`
- `world_points`, `world_points_conf`
- `track`, `vis`, `conf` if tracking is requested
- `images` during inference mode

One subtle but important detail: `VGGT.forward` returns `pose_enc`, not `extrinsic` and `intrinsic`. Converting `pose_enc` into explicit camera matrices happens later in utility/demo code, not inside the core model.

## 3. Shared backbone: the aggregator

The backbone is implemented in [`vggt/models/aggregator.py`](../vggt/models/aggregator.py#L25-L331).

### 3.1 Input normalization and patch embedding

[`Aggregator.forward`](../vggt/models/aggregator.py#L184-L258) first:

1. reads input shaped `(B, S, 3, H, W)`
2. normalizes RGB values with ImageNet-style mean/std buffers
3. reshapes to `(B*S, 3, H, W)`
4. applies the patch embedder

Relevant code:

- normalization: [`Aggregator.forward`](../vggt/models/aggregator.py#L195-L205)
- patch embed construction: [`Aggregator.__build_patch_embed__`](../vggt/models/aggregator.py#L143-L182)

By default the patch embedder is a DINOv2 ViT variant (`dinov2_vitl14_reg`), and the patch size is `14`.

### 3.2 Special tokens

After patch embedding, VGGT prepends learned special tokens:

- 1 camera token
- 4 register tokens by default

The relevant setup is here:

- token definitions: [`Aggregator.__init__`](../vggt/models/aggregator.py#L125-L141)
- `patch_start_idx = 1 + num_register_tokens`: [`Aggregator.__init__`](../vggt/models/aggregator.py#L130-L131)

So with the default settings:

- token index `0` is the camera token
- token indices `1:5` are register tokens
- patch tokens begin at index `5`

Another non-obvious detail is that special tokens are stored as `(1, 2, X, C)` and expanded with [`slice_expand_and_flatten`](../vggt/models/aggregator.py#L308-L330):

- slot `0` is used only for the first frame
- slot `1` is reused for all remaining frames

That means the first frame gets a distinct learned camera/register token embedding from the rest of the sequence.

### 3.3 Positional encoding

If rotary embeddings are enabled, the aggregator builds 2D positions for patch tokens and sets the special-token positions to zero:

- RoPE setup: [`Aggregator.__init__`](../vggt/models/aggregator.py#L76-L78)
- position creation and special-token handling: [`Aggregator.forward`](../vggt/models/aggregator.py#L219-L228)

### 3.4 Alternating attention

The main idea in VGGT is alternating between:

- frame attention: attention within each frame independently
- global attention: attention across all frame tokens jointly

This happens in [`Aggregator.forward`](../vggt/models/aggregator.py#L233-L258) using:

- [`_process_frame_attention`](../vggt/models/aggregator.py#L260-L282)
- [`_process_global_attention`](../vggt/models/aggregator.py#L284-L305)

The shape changes are the core of the data flow:

- frame attention uses `(B*S, P, C)`
- global attention uses `(B, S*P, C)`

where `P` is the number of tokens per frame after adding special tokens.

Each stage produces intermediates shaped `(B, S, P, C)`. VGGT concatenates the frame-attention and global-attention representations along the channel axis:

- concatenation step: [`Aggregator.forward`](../vggt/models/aggregator.py#L250-L253)

So each saved backbone output in `aggregated_tokens_list` has shape:

`(B, S, P, 2*C)`

With default settings:

- `C = 1024`
- head input channels become `2048`
- the list length is `24` because the default depth is `24` and one concatenated output is saved per alternating-attention stage

## 4. Camera branch

The camera branch is implemented in [`vggt/heads/camera_head.py`](../vggt/heads/camera_head.py#L19-L149).

### 4.1 What it reads

[`CameraHead.forward`](../vggt/heads/camera_head.py#L73-L93) uses only the last backbone output:

- `tokens = aggregated_tokens_list[-1]`
- `pose_tokens = tokens[:, :, 0]`

So the camera branch reads the camera token at index `0` for every frame.

Input shape to the camera head trunk:

- before token selection: `(B, S, P, 2C)`
- after selecting camera token: `(B, S, 2C)`

### 4.2 Iterative pose refinement

The actual refinement loop is in [`CameraHead.trunk_fn`](../vggt/heads/camera_head.py#L95-L141).

The logic is:

1. start from a learned empty pose token
2. embed the current pose estimate
3. use it to modulate the visual camera token via adaptive layer norm
4. run a transformer trunk
5. predict a pose delta
6. accumulate the delta into the pose estimate
7. apply output activations
8. repeat

The default number of iterations is `4`.

The output pose encoding has 9 values per frame:

- translation: `[:3]`
- quaternion rotation: `[3:7]`
- vertical and horizontal field of view: `[7:]`

Code references:

- target dimensionality: [`CameraHead.__init__`](../vggt/heads/camera_head.py#L40-L43)
- iterative update loop: [`CameraHead.trunk_fn`](../vggt/heads/camera_head.py#L110-L139)
- pose activation: [`activate_pose`](../vggt/heads/head_act.py#L16-L39)

### 4.3 Converting `pose_enc` into camera matrices

This conversion is outside the model core, in [`pose_encoding_to_extri_intri`](../vggt/utils/pose_enc.py#L62-L124).

That function:

- converts quaternion to rotation matrix
- builds a `3x4` OpenCV-style extrinsic matrix `[R | t]`
- reconstructs intrinsics from field of view and image size
- assumes principal point at image center

Relevant code:

- decoding: [`pose_encoding_to_extri_intri`](../vggt/utils/pose_enc.py#L102-L120)
- example usage in a demo: [`demo_viser.py`](../demo_viser.py#L365-L372)

This separation matters when reading outputs:

- inside `VGGT.forward`: camera output is `pose_enc`
- after post-processing: camera output becomes `extrinsic` and `intrinsic`

## 5. Depth branch

The depth branch is a [`DPTHead`](../vggt/heads/dpt_head.py#L21-L320) configured in [`VGGT.__init__`](../vggt/models/vggt.py#L24-L27) with:

- `output_dim=2`
- `activation="exp"`
- `conf_activation="expp1"`

### 5.1 What it reads

The DPT head uses four backbone stages:

- `intermediate_layer_idx = [4, 11, 17, 23]`

See [`DPTHead.__init__`](../vggt/heads/dpt_head.py#L51-L64).

Inside [`DPTHead._forward_impl`](../vggt/heads/dpt_head.py#L172-L247), each selected layer does:

1. remove special tokens with `x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]`
2. optionally slice frames
3. reshape patch tokens into a 2D feature map
4. project channels
5. resize to a common multi-scale pyramid

The patch-token extraction is here:

- [`DPTHead._forward_impl`](../vggt/heads/dpt_head.py#L205-L221)

This is why `patch_start_idx` from the aggregator is important: it tells each dense head where the patch grid begins.

### 5.2 Multi-scale fusion and output

After the four feature maps are built, the DPT head fuses them:

- fusion path: [`DPTHead.scratch_forward`](../vggt/heads/dpt_head.py#L261-L291)

Then it upsamples back to image resolution and applies the final activation head:

- interpolation and output conv: [`DPTHead._forward_impl`](../vggt/heads/dpt_head.py#L226-L247)
- activation split into prediction/confidence: [`activate_head`](../vggt/heads/head_act.py#L65-L116)

For the depth branch:

- raw output channels = 2
- `activate_head` interprets them as:
  - first channel: depth
  - second channel: confidence

Because activation is `"exp"`, depth values are exponentiated and become strictly positive:

- [`activate_head`](../vggt/heads/head_act.py#L90-L92)

Final depth outputs:

- `depth`: `(B, S, H, W, 1)`
- `depth_conf`: `(B, S, H, W)`

## 6. Point-map branch

The point-map branch is another [`DPTHead`](../vggt/heads/dpt_head.py#L21-L320), configured in [`VGGT.__init__`](../vggt/models/vggt.py#L24-L27) with:

- `output_dim=4`
- `activation="inv_log"`
- `conf_activation="expp1"`

Its internal data flow is identical to the depth branch up until the final activation.

For the point branch:

- raw output channels = 4
- `activate_head` interprets them as:
  - first 3 channels: XYZ world coordinates
  - last channel: confidence

Relevant activation path:

- [`activate_head`](../vggt/heads/head_act.py#L77-L116)

Because activation is `"inv_log"`, the XYZ coordinates are passed through the inverse log transform:

- [`inverse_log_transform`](../vggt/heads/head_act.py#L119-L129)

Final point-map outputs:

- `world_points`: `(B, S, H, W, 3)`
- `world_points_conf`: `(B, S, H, W)`

## 7. Depth-to-point post-processing path

The repo often prefers reconstructing 3D points from predicted depth plus predicted cameras instead of using the point-map branch directly.

That path is:

`depth + pose_enc -> extrinsic/intrinsic -> unprojection -> world points`

Code references:

- camera decoding: [`pose_encoding_to_extri_intri`](../vggt/utils/pose_enc.py#L62-L124)
- depth unprojection: [`unproject_depth_map_to_point_map`](../vggt/utils/geometry.py#L15-L44)
- per-frame world conversion: [`depth_to_world_coords_points`](../vggt/utils/geometry.py#L47-L84)
- pixel-to-camera conversion: [`depth_to_cam_coords_points`](../vggt/utils/geometry.py#L87-L117)

The demos use this path explicitly:

- [`demo_gradio.py`](../demo_gradio.py#L77-L93)
- [`demo_colmap.py`](../demo_colmap.py#L80-L90)

Important geometric convention:

- extrinsics are OpenCV-style camera-from-world transforms
- unprojection inverts them to get camera-to-world before placing points in world coordinates

That inversion happens in [`closed_form_inverse_se3`](../vggt/utils/geometry.py#L120-L169).

## 8. Tracking branch

The tracking branch is implemented in:

- [`TrackHead`](../vggt/heads/track_head.py#L12-L104)
- [`BaseTrackerPredictor`](../vggt/heads/track_modules/base_track_predictor.py#L17-L209)

### 8.1 When it runs

Tracking is optional. In [`VGGT.forward`](../vggt/models/vggt.py#L85-L91), the track branch runs only if:

- tracking is enabled in the model
- `query_points` is provided

### 8.2 Feature extraction for tracking

[`TrackHead`](../vggt/heads/track_head.py#L48-L57) first builds dense feature maps by reusing a DPT head in `feature_only=True` mode with `down_ratio=2`.

So the data flow is:

`aggregated_tokens_list -> DPT feature extractor -> feature_maps -> tracker`

The feature extractor output shape is:

- `(B, S, C, H/2, W/2)`

See:

- feature extraction: [`TrackHead.forward`](../vggt/heads/track_head.py#L91-L103)

### 8.3 Iterative point tracking

[`BaseTrackerPredictor.forward`](../vggt/heads/track_modules/base_track_predictor.py#L82-L209) performs iterative refinement.

The core steps are:

1. normalize feature maps
2. scale query points into feature-map coordinates
3. initialize coordinates in every frame from the query points
4. sample query features from frame `0`
5. build correlation features against all frames
6. encode motion relative to the first frame
7. run the update transformer
8. predict coordinate deltas and feature updates
9. keep the first-frame coordinates fixed
10. predict visibility and confidence

Important implementation details:

- query points are assumed to be defined in the first frame because features are sampled from `fmaps[:, 0]`: [`BaseTrackerPredictor.forward`](../vggt/heads/track_modules/base_track_predictor.py#L110-L114)
- coordinates are forced to remain equal to the original query in frame 0: [`BaseTrackerPredictor.forward`](../vggt/heads/track_modules/base_track_predictor.py#L184-L186)
- the tracker returns predictions at image scale after undoing stride/downsampling: [`BaseTrackerPredictor.forward`](../vggt/heads/track_modules/base_track_predictor.py#L188-L193)

Final tracking outputs:

- `track`: last iteration, `(B, S, N, 2)`
- `vis`: `(B, S, N)`
- `conf`: `(B, S, N)`

## 9. Shape summary

With default settings and image tensor input shaped `(B, S, 3, H, W)`:

| Stage | Main tensor shape | Notes |
| --- | --- | --- |
| Preprocessed images | `(B, S, 3, H, W)` | `VGGT.forward` adds `B` if missing |
| Patch tokens | `(B*S, patch_h*patch_w, C)` | after patch embed |
| Tokens with specials | `(B*S, P, C)` | camera + registers + patches |
| Saved backbone stage | `(B, S, P, 2C)` | frame/global outputs concatenated |
| Camera token stream | `(B, S, 2C)` | token index `0` only |
| Depth output | `(B, S, H, W, 1)` | plus `depth_conf` |
| Point output | `(B, S, H, W, 3)` | plus `world_points_conf` |
| Track features | `(B, S, C_t, H/2, W/2)` | from DPT feature extractor |
| Track output | `(B, S, N, 2)` | last refinement iteration |

Default special-token layout:

| Token index range | Meaning |
| --- | --- |
| `0` | camera token |
| `1:5` | register tokens |
| `5:` | patch tokens |

## 10. End-to-end inference path

The practical end-to-end pipeline used by the demos is:

1. Load image files with [`load_and_preprocess_images`](../vggt/utils/load_fn.py#L97-L230).
2. Run [`VGGT.forward`](../vggt/models/vggt.py#L29-L96) or call the heads manually.
3. Decode [`pose_enc`](../vggt/utils/pose_enc.py#L62-L124) into `extrinsic` and `intrinsic`.
4. Use either:
   - direct `world_points` from the point-map branch, or
   - `depth` + cameras via [`unproject_depth_map_to_point_map`](../vggt/utils/geometry.py#L15-L44).
5. If requested, run the tracking branch with query points in frame 0 coordinates.

## 11. What is learned jointly vs. derived later

Learned directly by the model:

- shared multi-view token representation
- camera pose encoding
- dense depth
- dense point map
- track trajectories / visibility / confidence

Derived after the model:

- `extrinsic` and `intrinsic` matrices from `pose_enc`
- depth-based world points from `depth + cameras`
- visualization/export artifacts such as COLMAP files or viewer point clouds

That distinction explains why some scripts appear to "predict cameras" even though the raw model output is actually `pose_enc`.
