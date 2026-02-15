from dataclasses import dataclass, field

from ldm_pre.jobs.flux2_klein import Config


@dataclass
class Flux2KleinConfig:
    _target_: str = "ldm_pre.jobs.flux2_klein.run"
    cfg: Config = field(default_factory=Config)
