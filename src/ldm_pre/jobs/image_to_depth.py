from typing import Literal

import ray.data as rd

from ldm_pre.udfs import ImageToDepthUDF


def run(
    input_uri: str,
    output_uri: str,
    ckpt_path: str,
    root_uri: str,
    *,
    output_prefix: str = "depth",
    output_ext: str = "png",
    encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl",
    **storage_options,
) -> None:
    ds = rd.read_parquet(input_uri)

    ds = ds.map(
        ImageToDepthUDF,
        fn_constructor_kwargs={
            "ckpt_path": ckpt_path,
            "root_uri": root_uri,
            "output_prefix": output_prefix,
            "output_ext": output_ext,
            "encoder": encoder,
            **storage_options,
        },
        compute=rd.ActorPoolStrategy(size=1),
        num_gpus=1,
    )

    ds.write_parquet(output_uri)
