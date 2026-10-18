from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Segmentation:
    """
    Semantic segmentation of one image.

    category_mask: (H, W) uint8, each pixel is an index into labels.
    labels: class names as the model defines them (e.g. "background", "person").
    confidence: optional (H, W, len(labels)) float32 in [0, 1], one channel per label.
    """
    category_mask: np.ndarray
    labels: tuple
    confidence: np.ndarray | None = None

    def mask_of(self, *names):
        """uint8 (0/255) mask of the pixels whose class is one of `names`."""
        ids = [self._index(name) for name in names]
        return np.isin(self.category_mask, ids).astype(np.uint8) * 255

    def confidence_of(self, name):
        """(H, W) float32 confidence of one class; needs a model run with confidence enabled."""
        if self.confidence is None:
            raise ValueError("This segmentation has no confidence: create the model with confidence=True.")
        return self.confidence[..., self._index(name)]

    def _index(self, name):
        try:
            return self.labels.index(name)
        except ValueError:
            raise KeyError(f"Label {name!r} not in {list(self.labels)}") from None


@runtime_checkable
class Segmenter(Protocol):
    """Assigns a class to every pixel of an image."""

    labels: tuple

    def __call__(self, image: np.ndarray) -> Segmentation: ...

    def close(self) -> None: ...
