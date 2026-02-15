from typing import Sequence
from dataclasses import dataclass


@dataclass
class Cols:
    hash: str = "image_hash"
    image_path: str = "image_path"
    height: str = "height"
    width: str = "width"
    captions: Sequence[str] = ("caption",)
    # image io specific columns
    image_uri: str = "image_uri"
    image_bytes: str = "image_bytes"
    image: str = "image"
    # bucket specific columns
    bucket_id: str = "bucket_id"
    target_height: str = "target_height"
    target_width: str = "target_width"
    # latent/embedding columns
    image_latents: str = "image_latents"
    input_ids: str = "input_ids"
    attention_mask: str = "attention_mask"
    text_embeddings: str = "text_embeddings"
    # condition columns
    condition_image: str = "condition_image"
    condition_image_bytes: str = "condition_image_bytes"
    condition_image_latents: str = "condition_image_latents"
