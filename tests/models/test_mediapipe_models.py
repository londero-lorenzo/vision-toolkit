"""
Contract of the MediaPipe models on synthetic images (no people, no private data).
Skipped without the mediapipe extra or the model files.
"""
import numpy as np
import pytest

from vision_toolkit.models import catalog
from vision_toolkit.tasks.pose import PoseEstimator
from vision_toolkit.tasks.segmentation import Segmenter

pytest.importorskip("mediapipe")


def needs(name):
    missing = catalog.missing_files(catalog.get(name))
    return pytest.mark.skipif(bool(missing), reason=f"run `vision-toolkit models download {name}`")


IMAGE = np.dstack([np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (200, 1))] * 3)


@needs("deeplab-v3")
def test_deeplab_contract():
    with catalog.create("deeplab-v3", confidence=True) as model:
        assert isinstance(model, Segmenter)
        seg = model(IMAGE)
    assert "person" in seg.labels and len(seg.labels) == 21
    assert seg.category_mask.shape == (200, 300) and seg.category_mask.dtype == np.uint8
    assert seg.confidence.shape == (200, 300, 21)
    assert 0 <= seg.confidence.min() and seg.confidence.max() <= 1
    assert set(np.unique(seg.mask_of("person"))) <= {0, 255}


@needs("selfie-multiclass")
def test_selfie_labels():
    with catalog.create("selfie-multiclass") as model:
        assert model.labels == ("background", "hair", "body-skin", "face-skin", "clothes", "others")
        assert model(IMAGE).confidence is None


@needs("mediapipe-pose-heavy")
def test_pose_contract():
    with catalog.create("mediapipe-pose-heavy") as model:
        assert isinstance(model, PoseEstimator)
        assert len(model.keypoint_names) == 33
        assert {"left_shoulder", "left_index", "right_hip"} <= set(model.keypoint_names)
        assert model(IMAGE) == []
