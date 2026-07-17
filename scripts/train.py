from __future__ import annotations

import hydra
import jax
from omegaconf import DictConfig

from dirt.train.trainer import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    jax.distributed.initialize()
    run_training(cfg)


if __name__ == "__main__":
    main()
