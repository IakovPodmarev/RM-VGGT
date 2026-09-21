"""Focused E01a contracts for standalone camera and depth read adaptors."""

from __future__ import annotations

import pytest
import torch

from vggt.rm_adaptor.camera_read_adaptor import CameraReadAdaptor
from vggt.rm_adaptor.depth_read_adaptor import DepthReadAdaptor


def _small_adaptors() -> tuple[CameraReadAdaptor, DepthReadAdaptor]:
    """Build independent CPU-sized adaptors retaining the E01a block structure."""
    dimensions = {
        "feature_dim": 8,
        "memory_dim": 8,
        "num_frames": 2,
        "memory_tokens": 4,
        "num_blocks": 3,
        "num_heads": 2,
    }
    return CameraReadAdaptor(**dimensions), DepthReadAdaptor(**dimensions)


def _inputs(
    adaptor: CameraReadAdaptor | DepthReadAdaptor, batch_size: int = 2, patches: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create camera, patch, and memory tensors matching an adaptor configuration."""
    camera = torch.randn(batch_size, adaptor.num_frames, adaptor.feature_dim)
    depth = torch.randn(batch_size, adaptor.num_frames, patches, adaptor.feature_dim)
    memory = torch.randn(batch_size, adaptor.memory_tokens, adaptor.memory_dim)
    return camera, depth, memory


def test_e01a_default_read_adaptor_constructors_expose_fixed_geometry():
    """Both defaults match the approved E01a feature, memory, block, and head counts."""
    camera, depth = CameraReadAdaptor(), DepthReadAdaptor()

    for adaptor in (camera, depth):
        assert adaptor.feature_dim == 2048
        assert adaptor.memory_dim == 512
        assert adaptor.memory_tokens == 16
        assert len(adaptor.blocks) == 3
        assert adaptor.num_heads == 8
        assert all(block.cross_attention.num_heads == 8 for block in adaptor.blocks)


def test_e01a_camera_forward_preserves_camera_layout_and_is_finite():
    """Camera adaptation keeps one finite feature vector per input frame."""
    camera_adaptor, _ = _small_adaptors()
    camera, _, memory = _inputs(camera_adaptor)

    output = camera_adaptor(camera, memory)

    assert output.shape == camera.shape
    assert torch.isfinite(output).all()


def test_e01a_depth_forward_preserves_patch_layout_order_and_finiteness():
    """Depth flattening presents and restores frame-major, patch-major ordering exactly."""
    _, depth_adaptor = _small_adaptors()
    _, depth, memory = _inputs(depth_adaptor, batch_size=1, patches=3)
    depth.copy_(torch.arange(depth.numel(), dtype=depth.dtype).reshape_as(depth))
    seen_flattened: list[torch.Tensor] = []

    def record_input(_module, args: tuple[torch.Tensor, ...]) -> None:
        seen_flattened.append(args[0].detach().clone())

    handle = depth_adaptor.input_projection.register_forward_pre_hook(record_input)
    try:
        output = depth_adaptor(depth, memory)
    finally:
        handle.remove()

    assert output.shape == depth.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(seen_flattened[0], depth.reshape(1, 6, 8))


def test_e01a_read_adaptor_gates_start_at_one_tenth():
    """The independent scalar residual gates begin at the specified 0.1 value."""
    camera, depth = _small_adaptors()

    torch.testing.assert_close(torch.sigmoid(camera.residual_gate), torch.tensor(0.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.sigmoid(depth.residual_gate), torch.tensor(0.1), atol=1e-6, rtol=0)


def test_e01a_effectively_zero_gate_is_identity():
    """An effectively closed residual gate leaves camera and depth inputs unchanged."""
    camera_adaptor, depth_adaptor = _small_adaptors()
    camera, depth, memory = _inputs(camera_adaptor, batch_size=1)
    with torch.no_grad():
        camera_adaptor.residual_gate.fill_(-100)
        depth_adaptor.residual_gate.fill_(-100)

    torch.testing.assert_close(camera_adaptor(camera, memory), camera, atol=1e-6, rtol=0)
    torch.testing.assert_close(depth_adaptor(depth, memory), depth, atol=1e-6, rtol=0)


def test_e01a_camera_and_depth_adaptor_parameters_are_unshared():
    """The two read paths own distinct projections, blocks, norms, and gates."""
    camera, depth = _small_adaptors()

    assert {id(parameter) for parameter in camera.parameters()}.isdisjoint(
        {id(parameter) for parameter in depth.parameters()}
    )


def test_e01a_changing_read_memory_changes_each_adaptor_output():
    """Both adaptors consume all incoming recurrent-memory information."""
    camera_adaptor, depth_adaptor = _small_adaptors()
    camera, depth, memory = _inputs(camera_adaptor, batch_size=1)

    changed_memory = memory.clone()
    changed_memory[..., 0] += 0.5
    assert not torch.allclose(camera_adaptor(camera, memory), camera_adaptor(camera, changed_memory))
    assert not torch.allclose(depth_adaptor(depth, memory), depth_adaptor(depth, changed_memory))


def test_e01a_every_query_attends_to_all_memory_tokens_in_every_block():
    """Camera and flattened depth queries each use all memory keys and values."""
    camera_adaptor, depth_adaptor = _small_adaptors()
    camera, depth, memory = _inputs(camera_adaptor, batch_size=1, patches=3)
    seen: list[tuple[int, int]] = []

    def record_shapes(_module, args: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        seen.append((args[0].shape[1], args[1].shape[1]))

    handles = [
        block.cross_attention.register_forward_hook(record_shapes)
        for adaptor in (camera_adaptor, depth_adaptor)
        for block in adaptor.blocks
    ]
    try:
        camera_adaptor(camera, memory)
        depth_adaptor(depth, memory)
    finally:
        for handle in handles:
            handle.remove()

    assert seen == [(2, 4)] * 3 + [(6, 4)] * 3


def test_e01a_backward_reaches_memory_and_every_required_adaptor_path():
    """Three-block camera adaptation has finite, nonzero reachable gradients."""
    camera_adaptor, _ = _small_adaptors()
    camera, _, memory = _inputs(camera_adaptor, batch_size=1)
    memory.requires_grad_()

    camera_adaptor(camera, memory).square().mean().backward()

    assert memory.grad is not None and torch.isfinite(memory.grad).all() and memory.grad.abs().any()
    for prefix in ("input_projection", "blocks", "output_projection", "residual_gate"):
        gradients = [parameter.grad for name, parameter in camera_adaptor.named_parameters() if name.startswith(prefix)]
        assert gradients
        assert any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().any() for gradient in gradients), prefix


def test_e01a_read_adaptors_do_not_mutate_inputs():
    """Features and recurrent memory retain their values after both adaptor calls."""
    camera_adaptor, depth_adaptor = _small_adaptors()
    camera, depth, memory = _inputs(camera_adaptor)
    originals = (camera.clone(), depth.clone(), memory.clone())

    camera_adaptor(camera, memory)
    depth_adaptor(depth, memory)

    for actual, original in zip((camera, depth, memory), originals):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize(
    ("adaptor_type", "feature_shape", "memory_shape", "message"),
    [
        (CameraReadAdaptor, (1, 8), (1, 4, 8), "camera_features"),
        (DepthReadAdaptor, (1, 2, 8), (1, 4, 8), "patch_features"),
        (CameraReadAdaptor, (1, 3, 8), (1, 4, 8), "num_frames"),
        (DepthReadAdaptor, (1, 2, 3, 7), (1, 4, 8), "feature_dim"),
        (CameraReadAdaptor, (2, 2, 8), (1, 4, 8), "batch"),
        (CameraReadAdaptor, (1, 2, 8), (1, 5, 8), "memory_tokens"),
        (CameraReadAdaptor, (1, 2, 8), (1, 4, 7), "memory width"),
    ],
)
def test_e01a_invalid_read_adaptor_shapes_fail_clearly(
    adaptor_type: type[CameraReadAdaptor] | type[DepthReadAdaptor],
    feature_shape: tuple[int, ...],
    memory_shape: tuple[int, ...],
    message: str,
) -> None:
    """Rank, batch, frame, feature, and memory contract errors identify their cause."""
    adaptor = adaptor_type(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_heads=2)
    with pytest.raises(ValueError, match=message):
        adaptor(torch.randn(feature_shape), torch.randn(memory_shape))


def test_e01a_device_and_dtype_incompatibilities_fail_clearly():
    """Features and memory must match the adaptor parameter device and dtype."""
    adaptor = CameraReadAdaptor(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_heads=2)
    camera, _, memory = _inputs(adaptor, batch_size=1)

    with pytest.raises(ValueError, match="dtype"):
        adaptor(camera.double(), memory.double())
    with pytest.raises(ValueError, match="device"):
        adaptor(torch.empty_like(camera, device="meta"), memory)
