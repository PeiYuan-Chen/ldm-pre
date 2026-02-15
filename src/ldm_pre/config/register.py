from hydra.core.config_store import ConfigStore

from .main_config import MainConfig
from .jobs import Flux2KleinConfig


def register_configs():
    cs = ConfigStore.instance()

    # main config
    cs.store(name="base_config", node=MainConfig)

    # jobs
    cs.store(group="job", name="base_flux2_klein", node=Flux2KleinConfig)
