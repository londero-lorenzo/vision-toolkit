import numpy as np
from mediapipe.tasks.python import vision

from vision_toolkit.models.mediapipe._common import base_options, read_mask, to_mp_image
from vision_toolkit.tasks.base import Model
from vision_toolkit.tasks.segmentation import Segmentation


class ImageSegmenter(Model):
    """
    Any MediaPipe ImageSegmenter model (DeepLab V3, selfie multiclass, ...).

    The labels come from the model file's metadata. Satisfies tasks.segmentation.Segmenter.
    With confidence=True the result also carries one confidence channel per label.
    """

    def __init__(self, model_path, confidence=False):
        opts = vision.ImageSegmenterOptions(
            base_options=base_options(model_path),
            output_category_mask=True,
            output_confidence_masks=confidence,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(opts)
        self.labels = tuple(self._segmenter.labels)

    def close(self):
        self._segmenter.close()

    def __call__(self, image):
        result = self._segmenter.segment(to_mp_image(image))
        category = read_mask(result.category_mask)
        confidence = None
        if result.confidence_masks:
            confidence = np.stack([read_mask(m) for m in result.confidence_masks], axis=-1)
        return Segmentation(category, self.labels, confidence)
