# Recurrent Memory Transformer Data Flow (Standalone)

This is a self-contained version of the data-flow documentation for this repository. It avoids file links and instead includes the key code snippets directly, so you can read it without jumping between files.

## 1. Where the model is

The main model code is in:

- `recurrent_memory_transformer_pytorch/recurrent_memory_transformer.py`

That file contains:

- helper functions such as `exists`, `default`, `frac_gradient`
- `RotaryEmbedding`
- `Attention`
- `RecurrentMemoryTransformer`
- `RecurrentMemoryTransformerWrapper`

The main training example is in:

- `train.py`

That file shows the actual repo training configuration and how the wrapper is intended to be used.

## 2. Exact training setup in this repo

The repository trains on `enwik8` with this exact setup:

```python
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 2048
SEQ_LEN = 2048
```

The instantiated model in `train.py` is:

```python
model = RecurrentMemoryTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    seq_len = 512,
    use_flash_attn = True,
    num_memory_tokens = 128,
    use_xl_memories = True,
    xl_mem_len = 256
)

model = RecurrentMemoryTransformerWrapper(model)
model.cuda()
```

The key practical detail is:

- training samples are length `2048`
- the core model segment length is `512`

So each training sample is processed as 4 recurrent segments:

```text
2048-token sample
  -> segment 0: 512 tokens
  -> segment 1: 512 tokens
  -> segment 2: 512 tokens
  -> segment 3: 512 tokens
```

Training uses memory replay backpropagation:

```python
loss = model(
    next(train_loader),
    memory_replay_backprop = True,
    mrbp_loss_weight = 1. / GRADIENT_ACCUMULATE_EVERY
)
```

So this repo’s real training path is segmented recurrent language-model training, not one giant retained 2048-token transformer graph.

## 3. Core segment-level model

The core class is:

```python
class RecurrentMemoryTransformer(Module):
```

Its job is to process one segment of tokens, optionally using:

- recurrent read/write memories
- per-layer XL memories

At a high level:

```text
token ids
  -> token embeddings
  -> prepend read memories
  -> append write memories
  -> transformer layers
  -> unpack sequence
  -> logits from the middle token region
  -> write memories returned for the next segment
  -> detached XL memories returned for the next segment
```

## 4. Token embedding and positions

The segment forward starts like this:

```python
b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)

assert n <= self.seq_len

pos = torch.arange(n, device = device)

x = self.token_emb(x)

if exists(self.pos_emb):
    x = x + self.pos_emb(pos)

x = frac_gradient(x, self.emb_gradient_frac)
```

What happens here:

- input `x` starts as token ids shaped `(B, N)`
- `token_emb` turns it into `(B, N, D)`
- optional absolute positional embeddings are added
- `frac_gradient` scales how much gradient flows through embeddings

That helper is:

```python
def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)
```

So when `emb_gradient_frac < 1`, the forward value stays the same, but part of the gradient is stopped.

## 5. Read and write memories

The model builds the write-memory block first:

```python
def init_memory(self, batch):
    return repeat(self.memory_tokens, 'm d -> b m d', b = batch)
```

and in `forward`:

```python
write_memories = self.init_memory(b)

if exists(read_memories) and self.add_write_to_next_write_mem:
    maybe_detach = torch.detach if self.next_write_mem_stop_grad else identity
    write_memories = write_memories + maybe_detach(read_memories)
```

So each segment starts with learned write-memory tokens of shape `(B, M, D)`.

Read memories are prepared like this:

```python
if exists(read_memories):
    if read_memories.ndim == 2:
        read_memories = repeat(read_memories, 'n d -> b n d', b = b)

    read_mem_length = mem_length
    read_memories = read_memories + self.read_memory_emb
elif self.always_have_read_memories:
    read_mem_length = mem_length
    read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = b)
else:
    read_mem_length = 0
    read_memories = x[:, 0:0]
```

So the model either:

- uses incoming recurrent memories
- uses learned read-memory embeddings on the first step
- or uses no read-memory slice at all

## 6. Packing the working sequence

The model concatenates read memories, main tokens, and write memories into one sequence:

```python
x, ps = pack([read_memories, x, write_memories], 'b * d')
```

Conceptually:

```text
[read memories] [main tokens] [write memories]
      M               N              M
```

If the main token block is `(B, N, D)` and both memory blocks are `(B, M, D)`, then the packed sequence becomes:

- `(B, 2M + N, D)`

This is the central architectural idea of this implementation:

- the model reads from the left memory block
- the model writes the next recurrent state into the right memory block

## 7. Masking and custom causal behavior

If a token mask exists, it is padded over the memory slots:

```python
if exists(mask):
    mask = F.pad(mask, (read_mem_length, mem_length), value = True)
```

If `causal=True` and `memory_not_causal=True`, the model builds a custom attention mask:

```python
if self.use_custom_causal_attn_mask:
    causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).tril()

    causal_mask = F.pad(causal_mask, (0, mem_length, read_mem_length, 0), value = False)
    causal_mask = F.pad(causal_mask, (read_mem_length, 0, 0, mem_length), value = True)

    causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        mask = mask & causal_mask
    else:
        mask = causal_mask
```

The intent is:

- main tokens stay causally masked with respect to one another
- memory tokens are treated differently from ordinary causal positions

The model can also mask out the read-memory block:

```python
if read_mem_length > 0 and mask_out_read_memories:
    read_mem_mask = torch.arange(x.shape[-2], device = device) < read_mem_length

    if exists(mask):
        mask = mask & ~read_mem_mask
    else:
        mask = read_mem_mask
```

## 8. Rotary positions with memories

If rotary embeddings are enabled, the model gives ordinary token positions to the main sequence but keeps memory slots at position `0`:

```python
if exists(self.rotary_pos_emb):
    mem_rel_dist = 10000

    q_pos = pos + mem_rel_dist

    if has_xl_memories:
        xl_mem_length = xl_memories[0].shape[-2]
        q_pos += xl_mem_length

    q_pos = F.pad(q_pos, (read_mem_length, mem_length), value = 0)
    q_rotary_emb = self.rotary_pos_emb(q_pos)
```

The key order for keys is:

```text
[xl memories] [read memories] [main sequence] [write memories]
```

And the code builds key positions accordingly:

```python
if has_xl_memories:
    k_pos = torch.arange(xl_mem_length, device = device) + mem_rel_dist
    k_pos = torch.cat((k_pos, q_pos), dim = -1)
else:
    k_pos = q_pos

k_pos = F.pad(k_pos, (1, 0), value = mem_rel_dist - 1)
k_rotary_emb = self.rotary_pos_emb(k_pos)

rotary_emb = (q_rotary_emb, k_rotary_emb)
```

So:

- main tokens have normal relative order
- XL memories look like older content
- read/write memory slots are positionless memory anchors
- the null key/value gets its own extra position

## 9. Attention internals

Each attention block begins with normalization and QKV projection:

```python
x = self.norm(x)

q = self.to_q(x)
k, v = self.to_kv(x).chunk(2, dim = -1)

q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
```

So the block computes per-head queries, keys, and values from the packed sequence.

### 9.1 Null key/value

Each layer prepends a learned null key/value:

```python
nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)

k = torch.cat((nk, k), dim = -2)
v = torch.cat((nv, v), dim = -2)
```

This gives the model an explicit "attend to nothing" option and also prevents issues when a whole row is masked.

### 9.2 Value residual mixing

The first attention layer computes its normal value tensor and saves it as a reference. Later layers can blend their own value tensor with that saved first-layer value tensor.

The learned mixing module is:

```python
if accept_value_residual:
    self.learned_value_residual_mix = nn.Sequential(
        Linear(dim, heads),
        Rearrange('b n h -> b h n 1'),
        nn.Sigmoid()
    )
```

The actual blend is:

```python
orig_v = v

if exists(self.learned_value_residual_mix):
    mix = self.learned_value_residual_mix(x)
    v = v.lerp(value_residual, mix)
```

And the first saved residual is established in the main layer loop:

```python
for attn, ff in self.layers:
    x, xl_memories, attn_values = attn(
        x,
        mask = mask,
        xl_memories = next(xl_memories_iter, None),
        rotary_emb = rotary_emb,
        value_residual = value_residual
    )

    value_residual = default(value_residual, attn_values)
    new_xl_memories.append(xl_memories)

    x = ff(x)
```

Short version:

- layer 1 produces the reference value tensor
- later layers can softly mix back toward it

### 9.3 XL memory cache

Each attention layer stores its current keys and values:

```python
next_xl_memories = torch.stack((k, v))
```

If previous XL memories exist, they are concatenated in front:

```python
if exists(xl_memories):
    kx, vx = xl_memories
    k = torch.cat((kx, k), dim = -2)
    v = torch.cat((vx, v), dim = -2)
```

So attention effectively sees:

```text
[past xl cache] [null kv] [current packed sequence kv]
```

## 10. Transformer stack outputs

Before the layers, the model expands residual streams:

```python
x = self.expand_streams(x)
```

After all attention/feedforward layers, it reduces them again:

```python
x = self.reduce_streams(x)
x = self.norm(x)
```

If XL memories are enabled, the model returns detached tail slices:

```python
if self.use_xl_memories:
    next_xl_memories = list(map(lambda t: torch.detach(t[..., -self.xl_mem_len:, :]), new_xl_memories))
```

Then it unpacks the sequence:

```python
read_memories, x, write_memories = unpack(x, ps, 'b * d')
```

Only the middle token block is converted to logits:

```python
logits = self.to_logits(x)
```

The output contract is:

```python
if not return_loss:
    return logits, write_memories, next_xl_memories

loss = F.cross_entropy(
    rearrange(logits, 'b n c -> b c n'),
    labels,
    ignore_index = self.ignore_index
)

return loss, write_memories, next_xl_memories
```

So the recurrent state comes from `write_memories`, not from the token logits.

## 11. Long-sequence wrapper

The wrapper class is:

```python
class RecurrentMemoryTransformerWrapper(Module):
```

Its job is to take a long sequence and split it into `seq_len` chunks:

```python
segments = x.split(seq_len, dim = -1)
total_length = x.shape[-1]
num_segments = len(segments)
segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))
```

If labels are not given but training loss is requested, it creates next-token labels automatically:

```python
labels = None
if (return_loss or memory_replay_backprop) and not exists(labels):
    x, labels = x[:, :-1], x[:, 1:]
```

Then it loops over segments in order:

```python
for step, (segment, mask_segment, label_segment, loss_weight) in enumerate(
    zip_longest(segments, mask_segments, label_segments, segment_length_frac)
):
    with forward_context():
        output, memories, xl_memories = self.transformer(
            segment,
            memories,
            mask = mask_segment,
            labels = label_segment
        )

    if exists(truncate_at_step) and divisible_by(step + 1, truncate_at_step):
        memories = memories.detach()
```

So each segment:

- receives recurrent memories from the previous segment
- returns the next recurrent memories
- optionally receives XL memory caches
- contributes weighted loss or logits

## 12. Generation

Generation is implemented in the wrapper because it needs to carry memories across segment boundaries.

First, it catches up on all full past segments:

```python
*past_segments, curr_segment = prime.split(seq_len, dim = -1)

for past_segment in past_segments:
    _, memories, xl_memories = self.transformer(
        past_segment,
        memories,
        xl_memories = xl_memories
    )
```

Then it samples token by token from the current segment:

```python
for ind in range(length - start_len):
    logits, next_memories, next_xl_memories = self.transformer(
        curr_segment,
        memories,
        xl_memories = xl_memories
    )

    logits = logits[:, -1]

    filtered_logits = top_k(logits, thres = filter_thres)
    sampled = gumbel_sample(filtered_logits, temperature = temperature)
    sampled = rearrange(sampled, 'b -> b 1')

    curr_segment = torch.cat((curr_segment, sampled), dim = -1)
```

When a segment fills up, it rolls forward:

```python
if divisible_by(curr_segment.shape[-1] - 1, seq_len):
    memories = next_memories
    xl_memories = next_xl_memories

    past_segment, curr_segment = curr_segment[..., :seq_len], curr_segment[..., -1:]
    past_segments.append(past_segment)
```

So generation preserves long-range state through recurrent memories and XL memories even though each core model call only sees one segment at a time.

## 13. Memory replay backpropagation

The repo’s most unusual training path is `memory_replay_backprop=True`.

First, the wrapper runs the full segment sequence with gradients disabled:

```python
forward_context = nullcontext if not memory_replay_backprop else torch.no_grad
```

During that pass, it stores only the recurrent states:

```python
replay_buffer = [memories]
xl_segments = [xl_memories]
```

Then it replays segments backward:

```python
memories_grad = torch.zeros_like(replay_buffer[-1])

reversed_inputs = zip_longest(*map(reversed, [
    range(num_segments),
    segments,
    replay_buffer[:-1],
    xl_segments[:-1],
    mask_segments,
    label_segments,
    segment_length_frac,
]))
```

For each segment in reverse order:

```python
for step, segment, segment_memories, segment_xl_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
    is_first = step == 0

    if exists(segment_memories):
        segment_memories.requires_grad_()

    loss, next_segment_memories, _ = self.transformer(
        segment,
        segment_memories,
        mask = mask_segment,
        xl_memories = segment_xl_memories,
        labels = label_segment
    )

    weighted_loss = loss * loss_weight * mrbp_loss_weight

    weighted_loss.backward(retain_graph = True)
    next_segment_memories.backward(memories_grad)
```

Then it propagates the memory gradient one segment further back:

```python
if is_first:
    continue

if exists(truncate_at_step) and divisible_by(step, truncate_at_step):
    memories_grad.zero_()
else:
    memories_grad.copy_(segment_memories.grad.data)
```

This lets the model train over long sequences while avoiding the cost of storing one giant graph across every segment.

## 14. Shape summary

For batch size `B`, segment length `N`, memory length `M`, model dim `D`, vocabulary size `V`:

- input ids: `(B, N)`
- token embeddings: `(B, N, D)`
- read memories: `(B, M, D)` when present
- write memories: `(B, M, D)`
- packed transformer input: `(B, 2M + N, D)`
- logits: `(B, N, V)`
- returned XL cache per layer: `(2, B, H, xl_mem_len, Dh)`

The most important architectural takeaway is:

- only the middle token region produces logits
- the right memory region becomes the recurrent state for the next segment
- the wrapper is what turns many short segment passes into one long-context model
