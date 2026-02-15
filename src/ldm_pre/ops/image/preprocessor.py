from typing import Literal
import math

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


TF_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "nearest-exact": InterpolationMode.NEAREST_EXACT,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
}


class ImagePreprocessor:
    def __init__(
        self,
        do_resize: bool = True,
        resize_mode: Literal["default", "crop"] = "crop",
        do_convert_rgb: bool = True,
        do_normalize: bool = True,
        resample: Literal[
            "nearest",
            "nearest-exact",
            "bilinear",
            "bicubic",
            "box",
            "hamming",
            "lanczos",
        ] = "bicubic",
        antialias: bool = True,
    ) -> None:
        self.do_resize = do_resize
        self.resize_mode = resize_mode
        self.do_convert_rgb = do_convert_rgb
        self.do_normalize = do_normalize
        self.resample = resample
        self.antialias = antialias

    def __call__(self, image: Image.Image, height: int, width: int) -> np.ndarray:
        if self.do_convert_rgb:
            image = self.convert_to_rgb(image)
        if self.do_resize:
            image = self.resize(image, height, width)
        image = self.pil_to_numpy(image)
        if self.do_normalize:
            image = self.normalize(image)
        # to CHW
        image = image.transpose(2, 0, 1)  # TODO np.ascontiguousarray
        return image

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        image = np.array(image).astype(np.float32) / 255.0
        return image

    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image

    def resize(self, image: Image.Image, height: int, width: int) -> Image.Image:
        if self.resize_mode == "default":
            image = F.resize(
                image,
                (height, width),
                TF_INTERPOLATION[self.resample],
                antialias=self.antialias,
            )
        elif self.resize_mode == "crop":
            image = self._resize_and_crop(image, height, width)
        else:
            raise ValueError(f"Invalid resize mode: {self.resize_mode}")

        return image

    @staticmethod
    def normalize(images: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return 2.0 * images - 1.0

    def _resize_and_crop(
        self,
        image: Image.Image,
        height: int,
        width: int,
    ) -> Image.Image:
        scale = max(height / image.height, width / image.width)
        resize_size = (
            math.ceil(image.height * scale),
            math.ceil(image.width * scale),
        )
        image = F.resize(
            image,
            resize_size,
            interpolation=TF_INTERPOLATION[self.resample],
            antialias=self.antialias,
        )
        image = F.center_crop(image, (height, width))
        return image
