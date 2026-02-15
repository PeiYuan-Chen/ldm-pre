from typing import Any
from ldm_pre.ops.image import AssignBucket
from ldm_pre.schema import Cols


class BucketAssigner:
    def __init__(self, buckets: list[tuple[int, int]], cols: Cols) -> None:
        self.bucket_assigner = AssignBucket(buckets)
        self.cols = cols

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        bucket_id, target_height, target_width = self.bucket_assigner(
            row[self.cols.height], row[self.cols.width]
        )
        row[self.cols.bucket_id] = bucket_id
        row[self.cols.target_height] = target_height
        row[self.cols.target_width] = target_width
        return row
