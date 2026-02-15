import posixpath
from dataclasses import dataclass, field

import fsspec
from omegaconf import MISSING
import ray
from ray.data.expressions import col, download, lit

from ldm_pre.schema import Cols
from ldm_pre.udfs import BucketAssigner, DepthAnything
from .udfs import Flux2KleinTransform, Flux2KleinLatentEncoder
from ldm_pre.io.datasink import MDSDataSink


@dataclass(frozen=True, kw_only=True)
class Config:
    input_uri: str = MISSING
    manifest_uri: str = MISSING
    output_uri: str = MISSING
    root_uri: str = MISSING
    buckets: list[list[int]] = MISSING
    cols: Cols = field(default_factory=Cols)
    batch_size: int = 1
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.2-klein-base-9B"
    condition: str | None = "depth"
    ckpt_path: str = "ckpts/depth-anything-v2-vitl.pth"


def run(cfg: Config):
    ds = ray.data.read_parquet(cfg.input_uri)

    # assign bucket
    ds = ds.map(
        BucketAssigner, fn_constructor_kwargs={"buckets": cfg.buckets, "cols": cfg.cols}
    )
    ds.write_parquet(cfg.manifest_uri, partition_cols=[cfg.cols.bucket_id])

    for bucket_id in range(len(cfg.buckets)):
        bucket_uri = posixpath.join(
            cfg.manifest_uri, f"{cfg.cols.bucket_id}={bucket_id}"
        )
        fs, path = fsspec.core.url_to_fs(bucket_uri)
        if not fs.exists(path):
            continue
        ds = ray.data.read_parquet(bucket_uri)

        # download images
        ds = (
            ds.with_column(
                cfg.cols.image_uri,
                lit(cfg.root_uri) + col(cfg.cols.image_path),
            )
            .with_column(
                cfg.cols.image_bytes,
                download(cfg.cols.image_uri),
            )
            .select_columns(
                [
                    cfg.cols.hash,
                    cfg.cols.image_bytes,
                    cfg.cols.target_height,
                    cfg.cols.target_width,
                    *cfg.cols.captions,
                ]
            )
        )
        if cfg.condition == "depth":
            ds = ds.map(
                DepthAnything,
                fn_constructor_kwargs={
                    "cols": cfg.cols,
                    "ckpt_path": cfg.ckpt_path,
                },
                num_gpus=1,
            )

        # 1. cpu transform
        ds = ds.map(
            Flux2KleinTransform,
            fn_constructor_kwargs={
                "cols": cfg.cols,
                "target_height": cfg.buckets[bucket_id][0],
                "target_width": cfg.buckets[bucket_id][1],
                "pretrained_model_name_or_path": cfg.pretrained_model_name_or_path,
            },
        )

        # 2. gpu encode
        ds = ds.map_batches(
            Flux2KleinLatentEncoder,
            fn_constructor_kwargs={
                "cols": cfg.cols,
                "pretrained_model_name_or_path": cfg.pretrained_model_name_or_path,
            },
            batch_size=cfg.batch_size,
            num_gpus=1,
        )
        # 3. write to mds
        ds.write_datasink(
            MDSDataSink(
                posixpath.join(cfg.output_uri, f"{cfg.cols.bucket_id}={bucket_id}")
            )
        )
