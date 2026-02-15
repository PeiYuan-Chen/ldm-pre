from typing import Literal

import cv2
from PIL import Image
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


class DepthAnythingV2Model:
    def __init__(
        self,
        ckpt_path: str,
        *,
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
        device: str = "cuda",
    ):
        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model = self.model.to(device).eval()

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_img = np.array(image)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        # cv2.imread(image_path)
        depth = self.model.infer_image(raw_img)  # HxW raw depth map in numpy
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        return Image.fromarray(depth)
