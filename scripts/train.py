from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dirt.train.trainer import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
