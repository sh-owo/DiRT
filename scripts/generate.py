from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dirt.inference.generate import run_generation


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_generation(cfg)


if __name__ == "__main__":
    main()
