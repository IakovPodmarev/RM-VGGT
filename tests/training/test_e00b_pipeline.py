# ruff: noqa: E402

from contextlib import contextmanager
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.base_dataset import BaseDataset
from data.dynamic_dataloader import DynamicTorchDataset
from loss import MultitaskLoss
from train_utils.freeze import freeze_modules
from train_utils.optimizer import construct_optimizers
from trainer import Trainer
from vggt.models.vggt import VGGT


class DummySequenceDataset(BaseDataset):
    def __init__(self, common_conf=None, common_config=None, length: int = 2):
        common_conf = common_conf if common_conf is not None else common_config
        super().__init__(common_conf)
        self.training = common_conf.training
        self.len_train = length

    def get_data(
        self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, img_per_seq=2
    ):
        image_count = img_per_seq
        target_shape = self.get_target_shape(aspect_ratio)
        base_h = int(target_shape[0]) + 14
        base_w = int(target_shape[1]) + 14

        images = []
        depths = []
        extrinsics = []
        intrinsics = []
        world_points = []
        cam_points = []
        point_masks = []

        for frame_idx in range(image_count):
            image = np.zeros((base_h, base_w, 3), dtype=np.uint8)
            image[..., 0] = (seq_index + frame_idx + 1) * 20
            image[..., 1] = np.arange(base_w, dtype=np.uint8)[None, :]
            image[..., 2] = np.arange(base_h, dtype=np.uint8)[:, None]

            depth_map = np.full(
                (base_h, base_w), 2.0 + 0.1 * frame_idx, dtype=np.float32
            )
            extrinsic = np.concatenate(
                [
                    np.eye(3, dtype=np.float32),
                    np.array([[0.1 * frame_idx], [0.0], [0.0]], dtype=np.float32),
                ],
                axis=1,
            )
            intrinsic = np.array(
                [
                    [20.0, 0.0, base_w / 2.0],
                    [0.0, 20.0, base_h / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            (
                proc_image,
                proc_depth,
                proc_extrinsic,
                proc_intrinsic,
                proc_world_points,
                proc_cam_points,
                proc_point_mask,
                _,
            ) = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extri_opencv=extrinsic,
                intri_opencv=intrinsic,
                original_size=np.array([base_h, base_w]),
                target_image_shape=target_shape,
            )

            images.append(proc_image)
            depths.append(proc_depth)
            extrinsics.append(proc_extrinsic)
            intrinsics.append(proc_intrinsic)
            world_points.append(proc_world_points)
            cam_points.append(proc_cam_points)
            point_masks.append(proc_point_mask)

        return {
            "seq_name": f"dummy_seq_{seq_index}",
            "ids": np.arange(image_count, dtype=np.int64),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
        }


def _make_common_config(training: bool) -> dict:
    return {
        "fix_img_num": 2,
        "fix_aspect_ratio": 1.0,
        "load_track": False,
        "track_num": 8,
        "training": training,
        "inside_random": False,
        "img_size": 28,
        "patch_size": 14,
        "rescale": True,
        "rescale_aug": False,
        "landscape_check": False,
        "debug": False,
        "get_nearby": False,
        "load_depth": True,
        "img_nums": [2, 2],
        "max_img_per_gpu": 2,
        "allow_duplicate_img": True,
        "repeat_batch": False,
        "augs": {
            "cojitter": False,
            "cojitter_ratio": 0.0,
            "scales": None,
            "aspects": [1.0, 1.0],
            "color_jitter": None,
            "gray_scale": False,
            "gau_blur": False,
        },
    }


def _make_tiny_model(enable_point: bool = False, enable_track: bool = False) -> VGGT:
    aggregator_kwargs = {
        "img_size": 28,
        "patch_size": 14,
        "embed_dim": 32,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "num_register_tokens": 2,
        "patch_embed": "conv",
        "cached_layer_indices": (0, 1, 2, 3),
    }
    dpt_kwargs = {
        "patch_size": 14,
        "features": 16,
        "out_channels": [16, 32, 64, 64],
        "intermediate_layer_idx": [0, 1, 2, 3],
    }
    return VGGT(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        enable_camera=True,
        enable_depth=True,
        enable_point=enable_point,
        enable_track=enable_track,
        aggregator_kwargs=aggregator_kwargs,
        camera_head_kwargs={"trunk_depth": 1, "num_heads": 4, "mlp_ratio": 2},
        depth_head_kwargs=dpt_kwargs,
        point_head_kwargs=dpt_kwargs,
        track_head_kwargs={
            "features": 16,
            "hidden_size": 32,
            "corr_levels": 2,
            "corr_radius": 2,
        },
    )


def _make_loader(training: bool):
    return DynamicTorchDataset(
        dataset={
            "_target_": "data.composed_dataset.ComposedDataset",
            "dataset_configs": [
                {
                    "_target_": "tests.training.test_e00b_pipeline.DummySequenceDataset",
                    "length": 2,
                }
            ],
        },
        common_config=OmegaConf.create(_make_common_config(training)),
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        seed=0,
        max_img_per_gpu=2,
    )


def _get_one_batch():
    loader = _make_loader(training=True).get_loader(epoch=0)
    batch = next(iter(loader))
    return batch


def _make_trainer_stub():
    trainer = Trainer.__new__(Trainer)
    trainer.data_conf = OmegaConf.create(
        {"train": {"common_config": {"repeat_batch": False}}}
    )
    trainer.loss = MultitaskLoss(
        camera={"weight": 5.0, "loss_type": "l1"},
        depth={"weight": 1.0, "gradient_loss_fn": "grad", "valid_range": 0.98},
        point=None,
        track=None,
    )
    trainer.steps = {"train": 0, "val": 0}
    trainer.rank = 0
    trainer.logging_conf = SimpleNamespace(log_freq=1, log_visuals=False)
    trainer.tb_writer = SimpleNamespace(
        log=lambda *args, **kwargs: None, log_visuals=lambda *args, **kwargs: None
    )
    trainer._update_and_log_scalars = lambda *args, **kwargs: None
    trainer._log_tb_visuals = lambda *args, **kwargs: None
    return trainer


@contextmanager
def _single_process_dist_context():
    yield


@pytest.fixture(scope="module", autouse=True)
def single_process_dist():
    with _single_process_dist_context():
        yield


def test_e00b_config_loads():
    with initialize_config_dir(
        version_base=None, config_dir=str(TRAINING_ROOT / "config")
    ):
        cfg = compose(config_name="e00b_training_pipeline_one_sample")

    assert cfg.exp_name == "e00b_training_pipeline_one_sample"
    assert cfg.model.enable_camera is True
    assert cfg.model.enable_depth is True
    assert cfg.model.enable_point is False
    assert cfg.model.enable_track is False
    assert cfg.optim.frozen_module_names == ["aggregator.patch_embed"]
    assert cfg.limit_train_batches == 1
    assert cfg.max_epochs == 1


def test_one_batch_dataloader_smoke():
    batch = _get_one_batch()

    assert batch["images"].shape == (1, 2, 3, 28, 28)
    assert batch["depths"].shape == (1, 2, 28, 28)
    assert batch["extrinsics"].shape == (1, 2, 3, 4)
    assert batch["intrinsics"].shape == (1, 2, 3, 3)
    assert batch["point_masks"].dtype == torch.bool
    assert batch["point_masks"].all()


def test_forward_loss_backward_optimizer_and_gradients():
    trainer = _make_trainer_stub()
    batch = trainer._process_batch(_get_one_batch())

    model = _make_tiny_model(enable_point=False, enable_track=False)
    model.train()
    freeze_modules(model, patterns=["aggregator.patch_embed"])

    predictions = model(images=batch["images"])
    assert set(predictions.keys()) == {
        "pose_enc",
        "pose_enc_list",
        "depth",
        "depth_conf",
    }
    assert predictions["pose_enc"].shape == (1, 2, 9)
    assert predictions["depth"].shape == (1, 2, 28, 28, 1)
    assert predictions["depth_conf"].shape == (1, 2, 28, 28)

    loss_dict = trainer._step(batch, model, "train", {})
    assert torch.isfinite(loss_dict["loss_camera"])
    assert torch.isfinite(loss_dict["loss_conf_depth"])
    assert torch.isfinite(loss_dict["loss_reg_depth"])
    assert torch.isfinite(loss_dict["loss_grad_depth"])
    assert torch.isfinite(loss_dict["objective"])

    optims = construct_optimizers(
        model,
        OmegaConf.create(
            {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                },
                "options": None,
            }
        ),
    )
    optimizer = optims[0]
    optimizer.zero_grad(set_to_none=True)
    loss_dict["objective"].backward()

    frame_grad = model.aggregator.frame_blocks[0].attn.qkv.weight.grad
    global_grad = model.aggregator.global_blocks[0].attn.qkv.weight.grad
    camera_grad = model.camera_head.pose_branch.fc2.weight.grad
    depth_grad = model.depth_head.projects[0].weight.grad

    assert frame_grad is not None and torch.count_nonzero(frame_grad).item() > 0
    assert global_grad is not None and torch.count_nonzero(global_grad).item() > 0
    assert camera_grad is not None and torch.count_nonzero(camera_grad).item() > 0
    assert depth_grad is not None and torch.count_nonzero(depth_grad).item() > 0

    patch_embed_params = list(model.aggregator.patch_embed.parameters())
    assert patch_embed_params
    assert all(param.requires_grad is False for param in patch_embed_params)
    assert all(param.grad is None for param in patch_embed_params)
    assert model.point_head is None
    assert model.track_head is None

    optimizer.step(where=1.0)


def test_freeze_policy_covers_disabled_head_names_when_present():
    model = _make_tiny_model(enable_point=True, enable_track=True)
    freeze_modules(
        model, patterns=["aggregator.patch_embed", "point_head", "track_head"]
    )

    assert all(
        param.requires_grad is False
        for param in model.aggregator.patch_embed.parameters()
    )
    assert all(param.requires_grad is False for param in model.point_head.parameters())
    assert all(param.requires_grad is False for param in model.track_head.parameters())
    assert any(
        param.requires_grad for param in model.aggregator.frame_blocks.parameters()
    )
    assert any(
        param.requires_grad for param in model.aggregator.global_blocks.parameters()
    )
    assert any(param.requires_grad for param in model.camera_head.parameters())
    assert any(param.requires_grad for param in model.depth_head.parameters())
