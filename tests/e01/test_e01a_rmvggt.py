"""E01a integration contracts for the RMVGGT model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

import vggt.models.RMVGGT as rmvggt_module
from vggt.models.RMVGGT import RMVGGT
from vggt.recurrent_memory.camera_read_adaptor import CameraReadAdaptor
from vggt.recurrent_memory.depth_read_adaptor import DepthReadAdaptor
from vggt.recurrent_memory.memory_writer import MemoryWriter


class _FakeAggregator(nn.Module):
    """Return a sparse 24-layer cache and record whether encoding has gradients."""

    def __init__(self, feature_dim: int = 8, patch_start_idx: int = 2) -> None:
        """Initialize a tiny trainable-looking aggregator for freeze-boundary tests."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.feature_dim = feature_dim
        self.patch_start_idx = patch_start_idx
        self.grad_enabled: bool | None = None

    def forward(self, images: Tensor) -> tuple[list[Tensor | None], int]:
        """Create sparse cache tensors from images without modeling VGGT internals."""
        self.grad_enabled = torch.is_grad_enabled()
        batch, frames = images.shape[:2]
        base = images.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        tokens = base[:, :, None, :].expand(batch, frames, 6, self.feature_dim) * self.scale
        cache: list[Tensor | None] = [None] * 24
        for index in (4, 11, 17, 23):
            cache[index] = tokens.clone()
        return cache, self.patch_start_idx


class _FakeCameraHead(nn.Module):
    """Produce camera outputs while retaining its received cache for inspection."""

    def __init__(self) -> None:
        """Initialize a minimal trainable camera prediction projection."""
        super().__init__()
        self.projection = nn.Linear(8, 1)
        self.received: list[Tensor | None] | None = None

    def forward(self, cache: list[Tensor | None]) -> list[Tensor]:
        """Return one pose-encoding iteration from the final camera token."""
        self.received = cache
        return [self.projection(cache[23][:, :, 0])]


class _FakeDepthHead(nn.Module):
    """Produce depth outputs while retaining its received cache for inspection."""

    def __init__(self) -> None:
        """Initialize a minimal trainable patch prediction projection."""
        super().__init__()
        self.projection = nn.Linear(8, 1)
        self.received: list[Tensor | None] | None = None

    def forward(self, cache: list[Tensor | None], images: Tensor, patch_start_idx: int) -> tuple[Tensor, Tensor]:
        """Return depth and confidence projections from final-layer patch tokens."""
        self.received = cache
        output = self.projection(cache[23][:, :, patch_start_idx:])
        return output, output + 1


def _model() -> tuple[RMVGGT, _FakeAggregator, _FakeCameraHead, _FakeDepthHead]:
    """Build injected components with a two-frame, four-patch geometry."""
    aggregator = _FakeAggregator()
    camera_head, depth_head = _FakeCameraHead(), _FakeDepthHead()
    writer = MemoryWriter(feature_dim=8, memory_dim=8, num_frames=2, memory_slots_per_type=2, patch_grid_size=2, pooled_grid_size=2, num_blocks=1, num_heads=2)
    camera_adaptor = CameraReadAdaptor(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_blocks=3, num_heads=2)
    depth_adaptor = DepthReadAdaptor(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_blocks=3, num_heads=2)
    return RMVGGT(aggregator, camera_head, depth_head, writer, camera_adaptor, depth_adaptor), aggregator, camera_head, depth_head


def _inputs(model: RMVGGT) -> tuple[Tensor, Tensor]:
    """Create one normalized-looking image segment and compatible incoming memory."""
    return torch.randn(1, 2, 3, 4, 4), model.initial_memory(1)


def test_e01a_rmvggt_freezes_aggregator_and_keeps_other_modules_trainable():
    """The frozen boundary remains eval while heads and recurrent modules train normally."""
    model, aggregator, _, _ = _model()
    model.train()

    assert not aggregator.training
    assert all(not parameter.requires_grad for parameter in aggregator.parameters())
    assert model.camera_head.training and model.depth_head.training
    assert model.memory_writer.training and model.camera_read_adaptor.training


def test_e01a_rmvggt_encode_segment_uses_no_grad_and_initial_memory_delegates():
    """Encoding has no history and initial memory comes directly from the writer."""
    model, aggregator, _, _ = _model()
    images, _ = _inputs(model)
    cache, patch_start_idx = model.encode_segment(images)

    assert aggregator.grad_enabled is False
    assert patch_start_idx == 2
    assert all(value is None or value.grad_fn is None for value in cache)
    assert model.initial_memory(2).shape == (2, 4, 8)


def test_e01a_rmvggt_uses_original_features_incoming_memory_and_isolated_caches():
    """Writer/adaptors and branch heads receive exactly their designated tensors."""
    model, _, camera_head, depth_head = _model()
    images, memory = _inputs(model)
    cache, patch_start_idx = model.encode_segment(images)
    original = [value.clone() if value is not None else None for value in cache]
    seen: dict[str, Tensor] = {}
    writer_forward, camera_forward, depth_forward = model.memory_writer.forward, model.camera_read_adaptor.forward, model.depth_read_adaptor.forward

    def write(camera: Tensor, patches: Tensor, read: Tensor) -> tuple[Tensor, Tensor]:
        seen.update(writer_camera=camera, writer_patches=patches, writer_memory=read)
        return writer_forward(camera, patches, read)

    def camera(camera: Tensor, read: Tensor) -> Tensor:
        seen.update(camera_memory=read)
        return camera_forward(camera, read)

    def depth(patches: Tensor, read: Tensor) -> Tensor:
        seen.update(depth_memory=read)
        return depth_forward(patches, read)

    model.memory_writer.forward = write
    model.camera_read_adaptor.forward = camera
    model.depth_read_adaptor.forward = depth
    predictions, next_memory, diagnostics = model.forward_segment(cache, images, patch_start_idx, memory)

    torch.testing.assert_close(seen["writer_camera"], original[23][:, :, 0])
    torch.testing.assert_close(seen["writer_patches"], original[23][:, :, 2:])
    assert seen["writer_memory"] is memory is seen["camera_memory"] is seen["depth_memory"]
    assert camera_head.received[23][:, :, 1:].equal(original[23][:, :, 1:])
    assert depth_head.received[23][:, :, :2].equal(original[23][:, :, :2])
    for index in (4, 11, 17):
        assert camera_head.received[index] is cache[index] is depth_head.received[index]
    assert all(torch.equal(value, baseline) for value, baseline in zip(cache, original) if value is not None)
    assert set(predictions) == {"pose_enc", "pose_enc_list", "depth", "depth_conf"}
    assert set(diagnostics) == {"memory_keep_gate", "camera_residual_gate", "depth_residual_gate"}
    assert next_memory.shape == memory.shape and diagnostics["memory_keep_gate"].requires_grad


def test_e01a_rmvggt_prediction_and_memory_losses_follow_separate_gradient_paths():
    """Current predictions avoid writer state while next-memory loss reaches the writer."""
    model, aggregator, _, _ = _model()
    images, memory = _inputs(model)
    memory.requires_grad_()
    memory.retain_grad()
    cache, patch_start_idx = model.encode_segment(images)
    predictions, next_memory, _ = model.forward_segment(cache, images, patch_start_idx, memory)
    next_memory.retain_grad()
    (predictions["pose_enc"].mean() + predictions["depth"].mean()).backward(retain_graph=True)

    assert next_memory.grad is None
    transition_gradients = [
        parameter.grad
        for name, parameter in model.memory_writer.named_parameters()
        if name != "initial_memory_bank"
    ]
    assert all(gradient is None for gradient in transition_gradients)
    assert memory.grad is not None and memory.grad.abs().any()
    assert model.camera_head.projection.weight.grad is not None
    assert model.depth_head.projection.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.camera_read_adaptor.parameters())
    assert any(parameter.grad is not None for parameter in model.depth_read_adaptor.parameters())
    assert all(parameter.grad is None for parameter in aggregator.parameters())

    model.zero_grad(set_to_none=True)
    next_memory.mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().any() for parameter in model.memory_writer.parameters())


def test_e01a_rmvggt_rejects_invalid_cache_geometry():
    """Absent layer 23, invalid patch index, and cache/image mismatch fail clearly."""
    model, _, _, _ = _model()
    images, memory = _inputs(model)
    cache, patch_start_idx = model.encode_segment(images)
    missing = list(cache)
    missing[23] = None
    for invalid_cache, invalid_images, invalid_start, message in (
        (missing, images, patch_start_idx, "layer 23"),
        (cache[:23], images, patch_start_idx, "layer 23"),
        (cache, images, 6, "patch_start_idx"),
        (cache, images[:, :1], patch_start_idx, "batch/frame"),
    ):
        try:
            model.forward_segment(invalid_cache, invalid_images, invalid_start, memory)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid segment inputs must raise ValueError")


def test_e01a_rmvggt_rejects_incompatible_memory_dtype_and_cache_device():
    """Memory and cache tensors must stay on the compatible dtype and device."""
    model, _, _, _ = _model()
    images, memory = _inputs(model)
    cache, patch_start_idx = model.encode_segment(images)
    try:
        model.forward_segment(cache, images, patch_start_idx, memory.double())
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("incompatible memory dtype must raise ValueError")
    meta_cache = list(cache)
    meta_cache[23] = cache[23].to("meta")
    try:
        model.forward_segment(meta_cache, images, patch_start_idx, memory)
    except ValueError as error:
        assert "device" in str(error)
    else:
        raise AssertionError("incompatible cache device must raise ValueError")


def test_e01a_rmvggt_rejects_matching_input_dtype_and_device_that_mismatch_modules():
    """Matching inputs still fail clearly when they differ from trainable modules."""
    model, _, _, _ = _model()
    images, memory = _inputs(model)
    cache, patch_start_idx = model.encode_segment(images)
    double_cache = [value.double() if value is not None else None for value in cache]
    try:
        model.forward_segment(double_cache, images.double(), patch_start_idx, memory.double())
    except ValueError as error:
        assert "dtype" in str(error)
    else:
        raise AssertionError("module dtype mismatch must raise ValueError")
    meta_cache = [value.to("meta") if value is not None else None for value in cache]
    try:
        model.forward_segment(meta_cache, images.to("meta"), patch_start_idx, memory.to("meta"))
    except ValueError as error:
        assert "device" in str(error)
    else:
        raise AssertionError("module device mismatch must raise ValueError")


def test_e01a_rmvggt_passes_configured_patch_size_to_default_depth_head(monkeypatch):
    """A non-default aggregator patch size reaches the default depth-head constructor."""
    seen: dict[str, int] = {}

    class RecordingDepthHead(nn.Module):
        """Record constructor keywords without constructing a production depth head."""

        def __init__(self, **kwargs: int | str) -> None:
            """Store the requested depth-head geometry for assertion."""
            super().__init__()
            seen.update(kwargs)

    monkeypatch.setattr(rmvggt_module, "DPTHead", RecordingDepthHead)
    aggregator = _FakeAggregator()
    writer = MemoryWriter(feature_dim=8, memory_dim=8, num_frames=2, memory_slots_per_type=2, patch_grid_size=2, pooled_grid_size=2, num_blocks=1, num_heads=2)
    camera_adaptor = CameraReadAdaptor(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_blocks=3, num_heads=2)
    depth_adaptor = DepthReadAdaptor(feature_dim=8, memory_dim=8, num_frames=2, memory_tokens=4, num_blocks=3, num_heads=2)

    RMVGGT(aggregator, _FakeCameraHead(), None, writer, camera_adaptor, depth_adaptor, patch_size=2, embed_dim=4)

    assert seen["patch_size"] == 2
