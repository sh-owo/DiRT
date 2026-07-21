from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dirt.inference import run_generation


@hydra.main(version_base=None, config_path="../configs", config_name="inference/default")
def main(cfg: DictConfig) -> None:
    result = run_generation(cfg)
    print(result)


if __name__ == "__main__":
    main()
