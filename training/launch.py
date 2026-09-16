# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from data.episode import validate_segment_dimensions
from hydra import initialize, compose
from omegaconf import DictConfig
from trainer import Trainer


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose and validate a launch configuration before trainer construction.

    Args:
        config_name: Hydra config name without the ``.yaml`` extension.
        overrides: Optional Hydra override expressions applied during composition.

    Returns:
        The composed configuration after experiment-specific validation.

    Raises:
        ValueError: If E01a streaming dimensions are inconsistent.
        hydra.errors.HydraException: If Hydra cannot compose the requested config.

    Invariants:
        E01a dimensions are validated immediately after composition and before
        ``Trainer`` construction. Configurations without the E01a section
        retain their existing composition behavior.
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    if "e01a" in cfg:
        e01a = cfg.e01a
        validate_segment_dimensions(
            total_frames=e01a.total_frames,
            segment_frames=e01a.segment_frames,
            num_segments=e01a.num_segments,
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Train model with configurable YAML file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file (without .yaml extension, default: default)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
