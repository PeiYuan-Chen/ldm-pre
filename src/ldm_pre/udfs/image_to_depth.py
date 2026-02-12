import io
from typing import Literal, Any
import posixpath

import fsspec
from PIL import Image
import torch

from ldm_pre.models import DepthAnythingV2Model
from ldm_pre.schema import Cols


class ImageToDepthUDF:
    def __init__(
        self,
        ckpt_path: str,
        root_uri: str,
        *,
        output_prefix: str = "depth",
        output_ext: str = "png",
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
        **storage_options,
    ):
        self.model = DepthAnythingV2Model(
            ckpt_path,
            encoder=encoder,
            device=("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        self.fs, self.root_path = fsspec.core.url_to_fs(root_uri, **storage_options)
        self.output_prefix = output_prefix
        self.output_ext = output_ext

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        with self.fs.open(posixpath.join(self.root_path, row[Cols.IMAGE_RELPATH])) as f:
            data = f.read()
        image = Image.open(io.BytesIO(data))
        depth_image = self.model(image)

        depth_image_relpath = posixpath.join(
            self.output_prefix, f"{row[Cols.SAMPLE_ID]}.{self.output_ext}"
        )
        with self.fs.open(
            posixpath.join(self.root_path, depth_image_relpath), "wb"
        ) as f:
            depth_image.save(f)

        row[Cols.COND_IMG_RELPATH] = depth_image_relpath
        return row
