from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ldm_pre.schema import Cols
from ldm_pre.ops.image import ImagePreprocessor
from ldm_pre.utils import bytes_to_image


class Transform(ABC):
    def __init__(
        self,
        cols: Cols,
        target_height: int,
        target_width: int,
        *args,
        **kwargs,
    ) -> None:
        self.cols = cols
        self.target_height = target_height
        self.target_width = target_width
        self.image_preprocessor = ImagePreprocessor(resize_mode="crop")

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        res = {self.cols.hash: row[self.cols.hash]}
        image = bytes_to_image(row[self.cols.image_bytes])
        res[self.cols.image] = self.image_preprocessor(
            image,
            self.target_height,
            self.target_width,
        )
        if self.cols.condition_image_bytes in row:
            condition_image = bytes_to_image(row[self.cols.condition_image_bytes])
            res[self.cols.condition_image] = self.image_preprocessor(
                condition_image,
                self.target_height,
                self.target_width,
            )

        for caption_col in self.cols.captions:
            input_ids, attention_mask = self.tokenize(row[caption_col])
            res[f"{caption_col}_{self.cols.input_ids}"] = input_ids
            res[f"{caption_col}_{self.cols.attention_mask}"] = attention_mask

        return res

    @abstractmethod
    def tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]: ...
