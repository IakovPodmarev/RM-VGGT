# Vision Transformer Data Flow

This document explains the data flow of the Vision Transformer implementation used in this repository, centered on [`DinoVisionTransformer`](../vggt/layers/vision_transformer.py#L42-L330). It follows the real code path and links each stage to the implementation.

## 1. Where this ViT is used

The implementation lives in:

- [`vggt/layers/vision_transformer.py`](../vggt/layers/vision_transformer.py#L42-L397)
- [`vggt/layers/patch_embed.py`](../vggt/layers/patch_embed.py#L25-L84)
- [`vggt/layers/block.py`](../vggt/layers/block.py#L27-L246)
- [`vggt/layers/attention.py`](../vggt/layers/attention.py#L21-L93)

Inside VGGT, this ViT is not used as a classifier. It is used as a patch-token extractor inside the shared aggregator:

- ViT construction inside the aggregator: [`Aggregator.__build_patch_embed__`](../vggt/models/aggregator.py#L143-L182)
- aggregator consuming `x_norm_patchtokens`: [`Aggregator.forward`](../vggt/models/aggregator.py#L203-L208)

That is an important distinction:

- standalone ViT API returns a feature dictionary
- VGGT usually keeps only `x_norm_patchtokens`

## 2. Top-level structure

The core class is [`DinoVisionTransformer`](../vggt/layers/vision_transformer.py#L42-L330).

Its main components are:

- patch embedding layer: [`self.patch_embed`](../vggt/layers/vision_transformer.py#L106-L107)
- class token: [`self.cls_token`](../vggt/layers/vision_transformer.py#L109-L110)
- positional embedding: [`self.pos_embed`](../vggt/layers/vision_transformer.py#L109-L110)
- optional register tokens: [`self.register_tokens`](../vggt/layers/vision_transformer.py#L111-L114)
- transformer block stack: [`self.blocks`](../vggt/layers/vision_transformer.py#L137-L165)
- final norm: [`self.norm`](../vggt/layers/vision_transformer.py#L166-L167)
- optional mask token: [`self.mask_token`](../vggt/layers/vision_transformer.py#L169-L169)

The forward path is:

```text
image
  -> patch embedding
  -> optional masking
  -> prepend cls token
  -> add / interpolate positional embedding
  -> insert register tokens
  -> transformer blocks
  -> final layer norm
  -> split output into cls / registers / patches
```

## 3. Input contract

For the normal single-tensor path, [`forward_features`](../vggt/layers/vision_transformer.py#L252-L271) expects:

- `x`: `(B, C, H, W)`
- optional `masks`

The model also supports a nested-list path through [`forward_features_list`](../vggt/layers/vision_transformer.py#L228-L250), but that path is only useful with xFormers-style nested attention and is not the common VGGT path.

## 4. Patch embedding

Patchification is done by [`PatchEmbed`](../vggt/layers/patch_embed.py#L25-L84).

### 4.1 What `PatchEmbed` does

[`PatchEmbed.forward`](../vggt/layers/patch_embed.py#L65-L78) does:

1. verify that `H` and `W` are divisible by patch size
2. apply a strided convolution with `kernel_size = stride = patch_size`
3. flatten the spatial grid into a token sequence
4. optionally apply normalization

The convolution is:

- [`self.proj`](../vggt/layers/patch_embed.py#L62-L63)

So for input `(B, C, H, W)` and patch size `(P_h, P_w)`:

- output grid size is `(H / P_h, W / P_w)`
- number of patch tokens is `(H / P_h) * (W / P_w)`
- output token shape is `(B, N, D)`

where:

- `N` = number of patches
- `D` = embedding dimension

### 4.2 Why this matters in VGGT

VGGT’s aggregator often builds this ViT with:

- patch size `14`
- `num_register_tokens > 0`
- large embedding sizes such as `1024`

But the aggregator discards the class/register outputs and uses only:

- [`x_norm_patchtokens`](../vggt/models/aggregator.py#L207-L208)

## 5. Token preparation

Token assembly happens in [`prepare_tokens_with_masks`](../vggt/layers/vision_transformer.py#L214-L226).

This function is the most important part of the ViT data flow.

### 5.1 Step 1: convert image to patch tokens

The first line is:

- [`x = self.patch_embed(x)`](../vggt/layers/vision_transformer.py#L215-L216)

Shape change:

- input: `(B, C, H, W)`
- output: `(B, N, D)`

### 5.2 Step 2: optional patch masking

If a mask is provided, masked patch tokens are replaced with `mask_token`:

- masking: [`torch.where(... self.mask_token ...)`](../vggt/layers/vision_transformer.py#L217-L219)

This happens before the class token is added.

### 5.3 Step 3: prepend the class token

The class token is inserted at the front:

- [`torch.cat((self.cls_token.expand(...), x), dim=1)`](../vggt/layers/vision_transformer.py#L220-L220)

Shape change:

- before: `(B, N, D)`
- after: `(B, N+1, D)`

Token layout at this point:

- index `0`: class token
- indices `1:`: patch tokens

### 5.4 Step 4: add positional embeddings

The token sequence then receives positional embeddings:

- position addition: [`x = x + self.interpolate_pos_encoding(x, w, h)`](../vggt/layers/vision_transformer.py#L221-L221)

The positional embedding table itself is learned as:

- [`self.pos_embed`](../vggt/layers/vision_transformer.py#L109-L110)

### 5.5 Step 5: insert register tokens

If register tokens are enabled, they are inserted between the class token and the patch tokens:

- register insertion: [`prepare_tokens_with_masks`](../vggt/layers/vision_transformer.py#L223-L224)

Final token layout becomes:

- index `0`: class token
- indices `1 : 1 + num_register_tokens`: register tokens
- remaining indices: patch tokens

This exact layout is what downstream code relies on when slicing outputs.

## 6. Positional embedding interpolation

Variable input resolution is handled by [`interpolate_pos_encoding`](../vggt/layers/vision_transformer.py#L180-L212).

### 6.1 What it interpolates

The learned `pos_embed` is stored for:

- one class token
- a square patch grid

Code:

- class token position: [`class_pos_embed = pos_embed[:, 0]`](../vggt/layers/vision_transformer.py#L187-L188)
- patch positions: [`patch_pos_embed = pos_embed[:, 1:]`](../vggt/layers/vision_transformer.py#L188-L188)

### 6.2 How resizing works

The function:

1. computes the target patch grid from `w // patch_size` and `h // patch_size`
2. reshapes patch positions into a 2D map
3. bicubically interpolates them
4. flattens them back into a sequence
5. concatenates the original class-token position back on top

Relevant code:

- target grid: [`interpolate_pos_encoding`](../vggt/layers/vision_transformer.py#L190-L203)
- interpolation: [`nn.functional.interpolate(...)`](../vggt/layers/vision_transformer.py#L204-L209)
- reassembly: [`torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)`](../vggt/layers/vision_transformer.py#L211-L212)

One subtle point: register tokens are not part of `pos_embed`. They are inserted only after position addition.

## 7. Transformer block stack

After token preparation, [`forward_features`](../vggt/layers/vision_transformer.py#L256-L263) runs the token sequence through `self.blocks`.

The blocks are built in:

- [`DinoVisionTransformer.__init__`](../vggt/layers/vision_transformer.py#L137-L165)

Each block is a [`Block`](../vggt/layers/block.py#L27-L98), typically using:

- [`MemEffAttention`](../vggt/layers/attention.py#L75-L93) for factory models like `vit_small`, `vit_base`, `vit_large`, `vit_giant2`

Factory definitions:

- [`vit_small`](../vggt/layers/vision_transformer.py#L341-L352)
- [`vit_base`](../vggt/layers/vision_transformer.py#L355-L366)
- [`vit_large`](../vggt/layers/vision_transformer.py#L369-L380)
- [`vit_giant2`](../vggt/layers/vision_transformer.py#L383-L397)

## 8. Inside one transformer block

The core residual block is [`Block.forward`](../vggt/layers/block.py#L77-L98).

Its structure is:

```text
x
  -> norm1
  -> attention
  -> optional LayerScale
  -> residual add
  -> norm2
  -> MLP / FFN
  -> optional LayerScale
  -> residual add
```

### 8.1 Attention path

Attention is built in [`Attention`](../vggt/layers/attention.py#L21-L72).

The sequence is:

1. project tokens to Q, K, V with one linear layer
2. reshape into multi-head format
3. optionally normalize Q and K
4. optionally apply RoPE
5. run scaled dot-product attention
6. merge heads
7. apply output projection

Code references:

- QKV projection: [`self.qkv`](../vggt/layers/attention.py#L42-L42)
- reshape/split: [`Attention.forward`](../vggt/layers/attention.py#L50-L54)
- optional RoPE: [`Attention.forward`](../vggt/layers/attention.py#L56-L58)
- fused attention: [`F.scaled_dot_product_attention`](../vggt/layers/attention.py#L60-L61)
- output projection: [`self.proj`](../vggt/layers/attention.py#L46-L47), [`Attention.forward`](../vggt/layers/attention.py#L69-L71)

Shape inside attention:

- input: `(B, N, D)`
- Q/K/V after split: `(B, heads, N, D_head)`
- output: `(B, N, D)`

### 8.2 Feed-forward path

The default feed-forward module is [`Mlp`](../vggt/layers/mlp.py#L16-L40).

Its internal flow is:

`Linear -> activation -> dropout -> Linear -> dropout`

Code:

- [`Mlp.forward`](../vggt/layers/mlp.py#L34-L40)

The hidden width is:

- `mlp_hidden_dim = int(dim * mlp_ratio)`: [`Block.__init__`](../vggt/layers/block.py#L67-L71)

### 8.3 LayerScale and DropPath

Each residual branch may be scaled by [`LayerScale`](../vggt/layers/layer_scale.py#L15-L22) and regularized with [`DropPath`](../vggt/layers/drop_path.py#L26-L34).

Relevant setup:

- attention branch scale/drop: [`Block.__init__`](../vggt/layers/block.py#L64-L65)
- MLP branch scale/drop: [`Block.__init__`](../vggt/layers/block.py#L72-L73)

At inference time, the block behaves as standard residual addition. The more complex stochastic-depth code paths mainly matter in training:

- standard inference path: [`Block.forward`](../vggt/layers/block.py#L95-L97)

## 9. Chunked blocks

The ViT can optionally group blocks into chunks using [`BlockChunk`](../vggt/layers/vision_transformer.py#L35-L39).

Construction:

- chunking logic: [`DinoVisionTransformer.__init__`](../vggt/layers/vision_transformer.py#L154-L164)

This is mainly a wrapping/layout concern for training and distributed setups. The core data flow remains the same:

- tokens still pass through all transformer blocks in order
- chunking only changes how those blocks are stored and iterated

## 10. Output contract of `forward_features`

After the block stack, the model applies a final layer norm:

- [`x_norm = self.norm(x)`](../vggt/layers/vision_transformer.py#L264-L264)

Then it returns a dictionary:

- [`forward_features`](../vggt/layers/vision_transformer.py#L264-L271)

The fields are:

- `x_norm_clstoken`: normalized class token, shape `(B, D)`
- `x_norm_regtokens`: normalized register tokens, shape `(B, R, D)`
- `x_norm_patchtokens`: normalized patch tokens, shape `(B, N, D)`
- `x_prenorm`: token sequence before the final norm, shape `(B, 1 + R + N, D)`
- `masks`: original mask input

Here `R = num_register_tokens`.

This split is defined by the token layout established earlier:

- class token at `0`
- register tokens at `1 : R+1`
- patch tokens at `R+1 :`

## 11. `forward` behavior

The top-level [`forward`](../vggt/layers/vision_transformer.py#L325-L330) has a slightly unusual contract:

- if `is_training=True` it returns the full feature dictionary from `forward_features`
- otherwise it returns `self.head(ret["x_norm_clstoken"])`

In this implementation:

- [`self.head = nn.Identity()`](../vggt/layers/vision_transformer.py#L166-L167)

So inference-mode `forward(..., is_training=False)` returns just the normalized class token unless the head is replaced.

In VGGT, this top-level `forward` is usually not what matters. The aggregator directly consumes the patch-token part of the feature dictionary.

## 12. Intermediate layer extraction

The model also exposes [`get_intermediate_layers`](../vggt/layers/vision_transformer.py#L299-L323).

This utility:

1. prepares tokens
2. runs the transformer
3. captures selected block outputs
4. optionally normalizes them
5. strips off class and register tokens
6. optionally reshapes patch tokens back into feature maps

Relevant code:

- non-chunked capture: [`_get_intermediate_layers_not_chunked`](../vggt/layers/vision_transformer.py#L273-L283)
- chunked capture: [`_get_intermediate_layers_chunked`](../vggt/layers/vision_transformer.py#L285-L297)
- final selection/reshape: [`get_intermediate_layers`](../vggt/layers/vision_transformer.py#L307-L323)

This is conceptually similar to what dense heads often need: patch-only features from multiple transformer depths.

## 13. Nested-list path

[`forward_features_list`](../vggt/layers/vision_transformer.py#L228-L250) supports a list of image tensors plus masks instead of a single batch tensor.

That path:

1. prepares tokens per input independently
2. runs the same block stack over a list
3. uses nested-tensor attention behavior when available
4. returns one feature dictionary per input

The nested behavior is implemented through:

- [`NestedTensorBlock.forward`](../vggt/layers/block.py#L239-L246)

This path is more specialized and not the standard VGGT inference route.

## 14. Factory model sizes

The factory helpers choose standard ViT scales:

| Factory | Embed dim | Depth | Heads |
| --- | --- | --- | --- |
| [`vit_small`](../vggt/layers/vision_transformer.py#L341-L352) | 384 | 12 | 6 |
| [`vit_base`](../vggt/layers/vision_transformer.py#L355-L366) | 768 | 12 | 12 |
| [`vit_large`](../vggt/layers/vision_transformer.py#L369-L380) | 1024 | 24 | 16 |
| [`vit_giant2`](../vggt/layers/vision_transformer.py#L383-L397) | 1536 | 40 | 24 |

All of them use:

- `MemEffAttention` through `partial(Block, attn_class=MemEffAttention)`

## 15. Shape summary

For a standard single-image batch input `(B, C, H, W)`:

| Stage | Shape |
| --- | --- |
| Input image | `(B, C, H, W)` |
| Patch embeddings | `(B, N, D)` |
| After cls token | `(B, N+1, D)` |
| After register insertion | `(B, N+1+R, D)` |
| After each transformer block | `(B, N+1+R, D)` |
| Final normalized class token | `(B, D)` |
| Final normalized register tokens | `(B, R, D)` |
| Final normalized patch tokens | `(B, N, D)` |

Where:

- `N = (H / patch_h) * (W / patch_w)`
- `R = num_register_tokens`
- `D = embed_dim`

## 16. How VGGT uses this ViT

In VGGT, the ViT is effectively used as a learned patch tokenizer plus transformer encoder.

The practical path is:

1. [`Aggregator.__build_patch_embed__`](../vggt/models/aggregator.py#L143-L182) constructs a DINOv2-style ViT.
2. [`Aggregator.forward`](../vggt/models/aggregator.py#L203-L205) calls it on flattened frames.
3. If the ViT returns a dictionary, the aggregator keeps only:
   - [`patch_tokens = patch_tokens["x_norm_patchtokens"]`](../vggt/models/aggregator.py#L207-L208)
4. The aggregator then adds its own multi-frame camera/register tokens and runs alternating frame/global attention on top.

So within the larger VGGT model, this ViT is the per-frame feature encoder, not the final geometry predictor.
