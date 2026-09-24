"""Argparse launch for configured training capabilities."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.episode import validate_segment_dimensions
from hydra import compose, initialize
from hydra.utils import get_class
from omegaconf import DictConfig
from trainer import Trainer


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose launch config and validate any generic recurrent schedule.

    Args:
        config_name: Config filename without extension.
        overrides: Optional Hydra override expressions.

    Returns:
        The composed, unresolved DictConfig, preserving ordinary config
        loading behavior and interpolation.

    Raises:
        ValueError: If configured sequence dimensions do not partition.
        hydra.errors.HydraException: If composition fails.
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    if "sequence" in cfg:
        sequence = cfg.sequence
        validate_segment_dimensions(
            total_frames=sequence.total_frames,
            segment_frames=sequence.segment_frames,
            num_segments=sequence.num_segments,
        )
    return cfg


def make_trainer(cfg: DictConfig, *, trainer_factory=None):
    """Select a configured trainer class without experiment-name branching.

    Args:
        cfg: Composed launch config.
        trainer_factory: Optional ordinary-trainer factory for injected tests.

    Returns:
        Recurrent capability trainer when trainer_target is configured;
        otherwise the existing generic trainer with its original keyword
        configuration convention.

    Raises:
        ValueError: Propagated when the selected trainer lacks required
            episode sources or receives unsupported settings.
    """
    target = cfg.get("trainer_target")
    if target:
        return get_class(target)(cfg)
    factory = trainer_factory or Trainer
    return factory(**cfg)


def main() -> None:
    """Parse a config name, construct its selected trainer, and run it."""
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file (without .yaml extension, default: default)",
    )
    args = parser.parse_args()
    trainer = make_trainer(load_config(args.config))
    trainer.run()


if __name__ == "__main__":
    main()
