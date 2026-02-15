import logging
import posixpath
from typing import Union, Iterable, Optional

from ray._common.retry import call_with_retry
from ray.data.context import DataContext
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.planner.plan_write_op import WRITE_UUID_KWARG_NAME
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.datasink import Datasink, WriteResult
from ray.data.datasource.filename_provider import (
    FilenameProvider,
    _DefaultFilenameProvider,
)
import pyarrow as pa
import numpy as np
from streaming.base import MDSWriter
from streaming.base.util import merge_index


logger = logging.getLogger(__name__)


class MDSDataSink(Datasink[None]):
    def __init__(
        self,
        path: str,
        *,
        columns: dict[str, str] | None = None,
        keep_local: bool = True,
        compression: Optional[str] = None,
        hashes: Optional[list[str]] = None,
        size_limit: Optional[Union[int, str]] = 1 << 26,
        exist_ok: bool = True,
        filename_provider: Optional[FilenameProvider] = None,
    ):
        self.path = path
        self.columns = columns
        self.keep_local = keep_local
        self.compression = compression
        self.hashes = hashes
        self.size_limit = size_limit
        self.exist_ok = exist_ok

        self.filename_provider = filename_provider or _DefaultFilenameProvider()
        self._data_context = DataContext.get_current()

    def on_write_start(self, schema: pa.Schema | None = None) -> None:
        if self.columns is not None:  # user-provided columns
            return

        # if schema is None, it means non-tabular dataset / object dataset, use pickle format
        if schema is None:
            self.columns = {"item": "pkl"}
            return

        # infer columns from pyarrow schema
        self.columns = infer_mds_columns_from_pyarrow_schema(schema)

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> None:
        builder = DelegatingBlockBuilder()
        for block in blocks:
            builder.add_block(block)
        block = builder.build()
        block_accessor = BlockAccessor.for_block(block)

        if block_accessor.num_rows() == 0:
            logger.warning(f"Skipped writing empty block to {self.path}")
            return

        self.write_block(block_accessor, 0, ctx)

    def write_block(self, block: BlockAccessor, block_index: int, ctx: TaskContext):
        partname = self.filename_provider.get_filename_for_block(
            block, ctx.kwargs[WRITE_UUID_KWARG_NAME], ctx.task_idx, block_index
        )
        write_path = posixpath.join(self.path, partname)

        def write_block_to_path():
            with MDSWriter(
                columns=self.columns,
                out=write_path,
                compression=self.compression,
                size_limit=self.size_limit,
                keep_local=self.keep_local,
                exist_ok=True,  # overwrite existing files for retries
            ) as out:
                self.write_block_with_mdswriter(block, out)

        call_with_retry(
            write_block_to_path,
            description=f"write '{write_path}'",
            match=self._data_context.retried_io_errors,
        )

    def on_write_complete(self, write_result: WriteResult[None]):
        merge_index(self.path, keep_local=self.keep_local)

    def write_block_with_mdswriter(self, block: BlockAccessor, out: MDSWriter):
        names = list(self.columns.keys())
        data = block.to_numpy()

        if not isinstance(data, dict):
            if len(names) != 1:
                raise ValueError(
                    f"Expected 1 column for object dataset, got {len(names)}"
                )
            data = {names[0]: data}

        for i in range(block.num_rows()):
            out.write({name: data[name][i] for name in names})


def infer_mds_columns_from_pyarrow_schema(schema: pa.Schema) -> dict[str, str]:
    out = {}

    for name, t in zip(schema.names, schema.types):
        # tensor -> ndarray:dtype:shape
        if hasattr(t, "scalar_type") and hasattr(t, "shape"):
            dt = np.dtype(t.scalar_type.to_pandas_dtype()).name
            out[name] = f"ndarray:{dt}:{",".join(map(str, t.shape))}"
            continue

        # string / bytes
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            out[name] = "str"
            continue
        if pa.types.is_binary(t) or pa.types.is_large_binary(t):
            out[name] = "bytes"
            continue

        # (optional) nested -> json
        if (
            pa.types.is_struct(t)
            or pa.types.is_list(t)
            or pa.types.is_large_list(t)
            or pa.types.is_map(t)
        ):
            out[name] = "json"
            continue

        # explicit numeric mapping (MDS-supported)
        if pa.types.is_uint8(t):
            out[name] = "uint8"
        elif pa.types.is_uint16(t):
            out[name] = "uint16"
        elif pa.types.is_uint32(t):
            out[name] = "uint32"
        elif pa.types.is_uint64(t):
            out[name] = "uint64"
        elif pa.types.is_int8(t):
            out[name] = "int8"
        elif pa.types.is_int16(t):
            out[name] = "int16"
        elif pa.types.is_int32(t):
            out[name] = "int32"
        elif pa.types.is_int64(t):
            out[name] = "int64"
        elif pa.types.is_float16(t):
            out[name] = "float16"
        elif pa.types.is_float32(t):
            out[name] = "float32"
        elif pa.types.is_float64(t):
            out[name] = "float64"
        else:
            out[name] = "pkl"

    return out
