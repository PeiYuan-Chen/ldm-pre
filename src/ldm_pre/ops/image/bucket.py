import numpy as np


class AssignBucket:
    def __init__(self, buckets: list[tuple[int, int]]) -> None:
        # (h, w)
        self.buckets = np.asarray(buckets, dtype=np.int32)
        self.aspect_ratio_buckets = self.buckets[:, 0] / self.buckets[:, 1]
        self.log_aspect_ratio_buckets = np.log(self.aspect_ratio_buckets)

    def __call__(
        self,
        image_height: int,
        image_width: int,
    ) -> tuple[int, int, int]:
        image_aspect_ratio = image_height / image_width
        bucket_id = np.argmin(
            np.abs(self.log_aspect_ratio_buckets - np.log(image_aspect_ratio))
        )
        target_height, target_width = self.buckets[bucket_id]
        return int(bucket_id), int(target_height), int(target_width)
