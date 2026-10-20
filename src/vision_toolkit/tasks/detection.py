from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """
    One object found in an image.

    box: (x0, y0, x1, y1) in pixels, x1 and y1 exclusive, clipped to the image.
    keypoints: optional named points (tasks.pose.Keypoint), with the names of
    tasks.pose: a face detector gives "left_eye", "nose", "mouth_left", ...
    """
    box: tuple
    score: float
    label: str
    keypoints: dict = field(default_factory=dict)

    def __getitem__(self, name):
        return self.keypoints[name]

    def __contains__(self, name):
        return name in self.keypoints

    def get(self, name, default=None):
        return self.keypoints.get(name, default)


@runtime_checkable
class Detector(Protocol):
    """Finds the objects of its labels in an image and returns their boxes."""

    labels: tuple

    def __call__(self, image: np.ndarray) -> list[Detection]: ...

    def close(self) -> None: ...
