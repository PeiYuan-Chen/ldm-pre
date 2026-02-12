import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    instantiate(cfg.job)


if __name__ == "__main__":
    main()
