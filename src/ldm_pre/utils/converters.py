import io

from PIL import Image
import torch
import numpy as np


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16).numpy()
    else:
        arr = tensor.numpy()
    return arr


def bytes_to_image(bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(bytes))
    img.load()  # force loading
    return img


def image_to_bytes(image: Image.Image, format: str = "PNG", **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=format, **kwargs)
    return buf.getvalue()
