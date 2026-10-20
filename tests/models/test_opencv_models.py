"""
Contract of the OpenCV models on synthetic images (no people, no private data).
Skipped without the model files.
"""
import numpy as np
import pytest

from vision_toolkit.models import catalog
from vision_toolkit.tasks.detection import Detector


def needs(name):
    missing = catalog.missing_files(catalog.get(name))
    return pytest.mark.skipif(bool(missing), reason=f"run `vision-toolkit models download {name}`")


IMAGE = np.dstack([np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (200, 1))] * 3)


@needs("yunet-face")
def test_yunet_contract():
    with catalog.create("yunet-face") as model:
        assert isinstance(model, Detector)
        assert model.labels == ("face",)
        assert model(IMAGE) == []
        assert model(np.zeros((480, 640, 3), np.uint8)) == []  # the input size follows each image
