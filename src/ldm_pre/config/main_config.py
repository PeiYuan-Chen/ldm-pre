from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class MainConfig:
    job: Any = MISSING
