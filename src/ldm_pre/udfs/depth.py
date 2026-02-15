import io
from typing import Literal, Any

import torch
from PIL import Image

from ldm_pre.models import DepthAnythingV2Model
from ldm_pre.schema import Cols
from ldm_pre.utils import bytes_to_image, image_to_bytes


class DepthAnything:
    def __init__(
        self,
        cols: Cols,
        ckpt_path: str,
        *,
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
    ):
        self.model = DepthAnythingV2Model(
            ckpt_path,
            encoder=encoder,
            device=("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        self.cols = cols

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        image = bytes_to_image(row[self.cols.image_bytes])
        depth_image = self.model(image)
        depth_bytes = image_to_bytes(depth_image, format="PNG")
        row[self.cols.condition_image_bytes] = depth_bytes
        return row
