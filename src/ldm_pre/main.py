import ray
import hydra
from hydra.utils import instantiate

from ldm_pre.config import register_configs, MainConfig

register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: MainConfig) -> None:
    ray.init()
    instantiate(cfg.job)


if __name__ == "__main__":
    main()
