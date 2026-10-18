from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# Keypoint names follow COCO ("left_shoulder", "right_wrist", ...) and MediaPipe for
# the points COCO lacks ("left_index", "left_heel", ...): lowercase, side first.
# Code downstream refers to keypoints only by these names, never by a model's indices.


@dataclass(frozen=True)
class Keypoint:
    x: float  # pixels
    y: float
    visibility: float = 1.0  # [0, 1]; 1.0 when the model does not estimate it

    @property
    def xy(self):
        return self.x, self.y


@dataclass
class Pose:
    """
    One person: named keypoints plus an optional uint8 (0/255) mask of the body.
    """
    keypoints: dict = field(default_factory=dict)
    mask: np.ndarray | None = None

    def __getitem__(self, name):
        return self.keypoints[name]

    def __contains__(self, name):
        return name in self.keypoints

    def get(self, name, default=None):
        return self.keypoints.get(name, default)


@runtime_checkable
class PoseEstimator(Protocol):
    """Finds the people in an image and returns their keypoints."""

    keypoint_names: tuple

    def __call__(self, image: np.ndarray) -> list[Pose]: ...

    def close(self) -> None: ...
