"""Focused E01a contracts for the standalone recurrent memory writer."""

from __future__ import annotations

import pytest
import torch

from vggt.recurrent_memory.memory_writer import MemoryWriter


def _small_writer(**overrides: int) -> MemoryWriter:
    """Build a CPU-sized E01a writer retaining its typed two-context design."""
    dimensions = {
        "feature_dim": 8,
        "memory_dim": 8,
        "num_frames": 2,
        "memory_slots_per_type": 2,
        "patch_grid_size": 3,
        "pooled_grid_size": 2,
        "num_blocks": 1,
        "num_heads": 2,
    }
    dimensions.update(overrides)
    return MemoryWriter(**dimensions)


def _inputs(
    writer: MemoryWriter, batch_size: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create finite independent inputs conforming to one test writer."""
    camera = torch.randn(batch_size, writer.num_frames, writer.feature_dim)
    patches = torch.randn(
        batch_size,
        writer.num_frames,
        writer.patch_grid_size**2,
        writer.feature_dim,
    )
    memory = torch.randn(batch_size, writer.total_memory_slots, writer.memory_dim)
    return camera, patches, memory


def test_e01a_default_contract_exposes_typed_parameters_and_initial_memory():
    """Default E01a dimensions and learned typed state follow the approved contract."""
    writer = MemoryWriter()

    assert writer.memory_type_embedding.shape == (2, 512)
    assert writer.query_type_embedding.shape == (2, 512)
    assert writer.slot_embedding.shape == (16, 512)
    assert writer.initial_memory_bank.shape == (1, 16, 512)
    assert writer.write_queries.shape == (1, 16, 512)
    assert writer.patch_position_embedding.shape == (256, 512)
    assert "patch_position_embedding" not in writer.state_dict()

    initial = writer.initial_memory(2)
    assert initial.shape == (2, 16, 512)
    initial[0].zero_()
    assert writer.initial_memory_bank.any()
    assert initial[1].any()


def test_e01a_patch_position_embedding_uses_approved_row_major_sinusoid():
    """The fixed 16-by-16 float32 buffer uses the specified row/column order."""
    writer = MemoryWriter()
    position = writer.patch_position_embedding

    assert position.dtype == torch.float32
    torch.testing.assert_close(position[0, :128], torch.zeros(128))
    torch.testing.assert_close(position[0, 128:256], torch.ones(128))
    torch.testing.assert_close(position[0, 256:384], torch.zeros(128))
    torch.testing.assert_close(position[0, 384:], torch.ones(128))
    omega0 = torch.tensor(1.0)
    torch.testing.assert_close(position[16, 0], torch.sin(omega0))
    torch.testing.assert_close(position[1, 256], torch.sin(omega0))


def test_e01a_lightweight_forward_returns_finite_state_and_gate():
    """A small configurable writer produces the promised finite output shapes."""
    writer = _small_writer()
    camera, patches, memory = _inputs(writer)

    next_memory, keep_gate = writer(camera, patches, memory)

    assert next_memory.shape == memory.shape
    assert keep_gate.shape == (*memory.shape[:2], 1)
    assert torch.isfinite(next_memory).all()
    assert torch.isfinite(keep_gate).all()


def test_e01a_gate_starts_at_approximately_one_third_keep_probability():
    """Zero gate-output weights and its bias yield the required initial retention."""
    writer = _small_writer()
    camera, patches, memory = _inputs(writer)

    _, keep_gate = writer(camera, patches, memory)

    torch.testing.assert_close(keep_gate, torch.full_like(keep_gate, 0.33), atol=0.01, rtol=0)


def test_e01a_default_patch_pooling_has_2048_observation_tokens_efficiently():
    """Default geometry pools eight 37-by-37 frames to 8 times 16 squared tokens."""
    writer = _small_writer(
        feature_dim=2,
        memory_dim=4,
        num_frames=8,
        patch_grid_size=37,
        pooled_grid_size=16,
        num_heads=1,
    )
    camera, patches, memory = _inputs(writer, batch_size=1)

    _, patch_context = writer._build_contexts(camera, patches, memory)

    assert patch_context.shape == (1, 8 * 16 * 16 + 2, 4)


def test_e01a_contexts_append_only_their_corresponding_typed_memory():
    """Camera and patch contexts append their own incoming typed-memory groups only."""
    writer = _small_writer()
    camera = torch.zeros(1, 2, 8)
    patches = torch.zeros(1, 2, 9, 8)
    memory = torch.zeros(1, 4, 8)
    memory[:, :2] = 3
    memory[:, 2:] = 7

    camera_context, patch_context = writer._build_contexts(camera, patches, memory)

    torch.testing.assert_close(camera_context[:, -2:], writer._memory_attention_tokens(memory[:, :2], 0))
    torch.testing.assert_close(patch_context[:, -2:], writer._memory_attention_tokens(memory[:, 2:], 1))


def test_e01a_all_write_slots_query_both_contexts():
    """Every one of the typed write slots is supplied to both cross-attention calls."""
    writer = _small_writer()
    camera, patches, memory = _inputs(writer, batch_size=1)
    seen_query_lengths: list[int] = []

    def record_query_length(_module, args: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        seen_query_lengths.append(args[0].shape[1])

    block = writer.blocks[0]
    camera_handle = block.camera_cross_attention.register_forward_hook(record_query_length)
    patch_handle = block.patch_cross_attention.register_forward_hook(record_query_length)
    try:
        writer(camera, patches, memory)
    finally:
        camera_handle.remove()
        patch_handle.remove()

    assert seen_query_lengths == [4, 4]


def test_e01a_read_memory_affects_the_next_state_without_input_mutation():
    """Incoming state participates in both contexts and remains untouched by writing."""
    writer = _small_writer()
    camera, patches, memory = _inputs(writer, batch_size=1)
    original_camera, original_patches, original_memory = (camera.clone(), patches.clone(), memory.clone())

    next_memory, _ = writer(camera, patches, memory)
    changed_memory = memory + 0.5
    changed_next_memory, _ = writer(camera, patches, changed_memory)

    assert not torch.allclose(next_memory, changed_next_memory)
    torch.testing.assert_close(camera, original_camera)
    torch.testing.assert_close(patches, original_patches)
    torch.testing.assert_close(memory, original_memory)


def test_e01a_backward_reaches_every_required_writer_component():
    """One transition gives finite nonzero gradients to all E01a trainable paths."""
    writer = _small_writer()
    camera, patches, memory = _inputs(writer, batch_size=1)
    memory.requires_grad_()

    next_memory, keep_gate = writer(camera, patches, memory)
    (next_memory.square().mean() + keep_gate.mean()).backward()

    assert memory.grad is not None and torch.isfinite(memory.grad).all() and memory.grad.abs().any()
    required_prefixes = (
        "write_queries",
        "camera_projection",
        "patch_projection",
        "blocks",
        "candidate_mlp",
        "gate_mlp",
    )
    for prefix in required_prefixes:
        gradients = [parameter.grad for name, parameter in writer.named_parameters() if name.startswith(prefix)]
        assert gradients
        assert any(
            gradient is not None and torch.isfinite(gradient).all() and gradient.abs().any()
            for gradient in gradients
        ), prefix


@pytest.mark.parametrize(
    ("camera_shape", "patch_shape", "memory_shape", "message"),
    [
        ((1, 8), (1, 2, 9, 8), (1, 4, 8), "camera_features"),
        ((1, 2, 8), (1, 2, 8, 8), (1, 4, 8), "patch_features"),
        ((1, 2, 8), (1, 2, 9, 8), (1, 4), "read_memory"),
        ((1, 3, 8), (1, 2, 9, 8), (1, 4, 8), "num_frames"),
        ((1, 2, 8), (1, 2, 9, 7), (1, 4, 8), "feature_dim"),
        ((1, 2, 8), (1, 2, 9, 8), (1, 5, 8), "memory slots"),
    ],
)
def test_e01a_invalid_input_shapes_fail_with_clear_errors(
    camera_shape: tuple[int, ...],
    patch_shape: tuple[int, ...],
    memory_shape: tuple[int, ...],
    message: str,
) -> None:
    """Rank, width, frame, and patch-grid contract violations name their cause."""
    writer = _small_writer()

    with pytest.raises(ValueError, match=message):
        writer(
            torch.randn(camera_shape),
            torch.randn(patch_shape),
            torch.randn(memory_shape),
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"memory_dim": 7, "num_heads": 2}, "divisible"),
        ({"patch_grid_size": 0}, "patch_grid_size"),
        ({"pooled_grid_size": 0}, "pooled_grid_size"),
    ],
)
def test_e01a_incompatible_writer_geometry_fails_clearly(
    kwargs: dict[str, int], message: str
) -> None:
    """Constructor rejects attention and pooling geometries it cannot realize."""
    with pytest.raises(ValueError, match=message):
        _small_writer(**kwargs)
