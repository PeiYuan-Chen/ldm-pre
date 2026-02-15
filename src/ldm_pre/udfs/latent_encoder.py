from abc import ABC, abstractmethod

import numpy as np
import torch

from ldm_pre.schema import Cols
from ldm_pre.utils import tensor_to_numpy


class LatentEncoder(ABC):
    def __init__(
        self,
        cols: Cols,
        *args,
        **kwargs,
    ) -> None:
        self.cols = cols

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        res = {self.cols.hash: batch[self.cols.hash]}

        with torch.no_grad():
            res[self.cols.image_latents] = self.encode_images(batch[self.cols.image])
            if self.cols.condition_image in batch:
                res[self.cols.condition_image_latents] = tensor_to_numpy(
                    self.encode_images(batch[self.cols.condition_image])
                )
            for caption_col in self.cols.captions:
                res[f"{caption_col}_{self.cols.text_embeddings}"] = tensor_to_numpy(
                    self.encode_text(
                        batch[f"{caption_col}_{self.cols.input_ids}"],
                        batch[f"{caption_col}_{self.cols.attention_mask}"],
                    )
                )

        return res

    @abstractmethod
    def encode_images(self, images: np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def encode_text(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> torch.Tensor: ...
