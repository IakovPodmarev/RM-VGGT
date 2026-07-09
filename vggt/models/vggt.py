# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        aggregator_kwargs=None,
        camera_head_kwargs=None,
        point_head_kwargs=None,
        depth_head_kwargs=None,
        track_head_kwargs=None,
    ):
        super().__init__()

        aggregator_kwargs = aggregator_kwargs or {}
        camera_head_kwargs = camera_head_kwargs or {}
        point_head_kwargs = point_head_kwargs or {}
        depth_head_kwargs = depth_head_kwargs or {}
        track_head_kwargs = track_head_kwargs or {}

        aggregator_defaults = {
            "img_size": img_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
        }
        aggregator_config = {**aggregator_defaults, **aggregator_kwargs}
        self.aggregator = Aggregator(**aggregator_config)

        head_dim = 2 * aggregator_config["embed_dim"]
        track_patch_size = aggregator_config["patch_size"]

        if enable_camera:
            self.camera_head = CameraHead(dim_in=head_dim, **camera_head_kwargs)
        else:
            self.camera_head = None

        if enable_point:
            point_defaults = {
                "dim_in": head_dim,
                "output_dim": 4,
                "activation": "inv_log",
                "conf_activation": "expp1",
            }
            self.point_head = DPTHead(**{**point_defaults, **point_head_kwargs})
        else:
            self.point_head = None

        if enable_depth:
            depth_defaults = {
                "dim_in": head_dim,
                "output_dim": 2,
                "activation": "exp",
                "conf_activation": "expp1",
            }
            self.depth_head = DPTHead(**{**depth_defaults, **depth_head_kwargs})
        else:
            self.depth_head = None

        if enable_track:
            track_defaults = {
                "dim_in": head_dim,
                "patch_size": track_patch_size,
            }
            self.track_head = TrackHead(**{**track_defaults, **track_head_kwargs})
        else:
            self.track_head = None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[
                    -1
                ]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = (
                images  # store the images for visualization during inference
            )

        return predictions
