"""Rank-wide gradient decision for one synchronized optimizer update."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class DistributedGradientCoordinator:
    """Agree on gradient finiteness before any rank clips or steps."""

    def __call__(self, model: nn.Module, optimizer: Any, scaler: Any) -> bool:
        """Return shared finiteness and prepare native GradScaler skip state.

        Call once per rank after backward and unscale, with an initialized
        process group and the same collective order on every rank. A disabled
        scaler raises ValueError on all ranks for any nonfinite gradient. An
        enabled scaler records global overflow for its normal step/update path.
        """
        if not dist.is_initialized():
            raise RuntimeError("distributed gradient coordination requires an initialized process group")
        parameter = next(model.parameters(), None)
        if parameter is None:
            raise ValueError("distributed gradient coordination requires model parameters")
        finite = torch.ones((), dtype=torch.int32, device=parameter.device)
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                finite *= torch.isfinite(parameter.grad).all().to(dtype=torch.int32)
        found_infs = None
        if scaler.is_enabled():
            # ponytail: PyTorch 2.4.1 has no public post-unscale overflow merge;
            # revisit this accessor when upgrading its GradScaler implementation.
            found_infs = scaler._found_inf_per_device(optimizer)
            if not found_infs:
                raise RuntimeError("GradScaler recorded no inf checks for the optimizer")
            for found_inf in found_infs.values():
                finite *= (found_inf == 0).to(device=finite.device, dtype=torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            if not scaler.is_enabled():
                raise ValueError("nonfinite gradient on a distributed rank")
            for found_inf in found_infs.values():
                found_inf.fill_(1.0)
            return False
        return True
