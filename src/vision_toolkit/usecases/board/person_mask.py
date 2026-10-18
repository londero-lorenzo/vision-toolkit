from dataclasses import dataclass

import cv2
import numpy as np

from vision_toolkit.models import catalog
from vision_toolkit.ops.crop_refine import segment_on_crop
from vision_toolkit.ops.pointing import pointing_hand
from vision_toolkit.ops.skeleton import ARMS, draw_limbs
from vision_toolkit.tasks.pose import Pose

DEFAULT_POSE_MODEL = "mediapipe-pose-heavy"


@dataclass
class PersonMask:
    mask: np.ndarray            # uint8 0/255, already dilated; empty if nobody is in view
    pose: Pose | None
    pointer: tuple | None       # (x, y) fingertip of the more raised arm
    arm_angle: float            # degrees between arm and torso: ~0 at rest, ~90 stretched out


class PersonMasker:
    """
    Lecturer mask and pointed spot, composed from catalog models and ops.

    mask = body mask of the pose model + arms drawn as a skeleton
           (+ a segmenter run on the crop around the person, if refine_model is set)
           + dilation.

    The default pose model (PoseLandmarker) works in two stages, so it finds the
    lecturer in the full frame; a small-input segmenter such as DeepLab V3 does
    not, but it does on the crop. The pose mask sometimes loses a stretch of the
    raised arm: the skeleton covers it, refine_model can add what is left.
    """

    def __init__(self, pose_model=DEFAULT_POSE_MODEL, refine_model=None, refine_labels=("person",),
                 dilate=15, models_dir=None):
        self.pose = catalog.create(pose_model, task="pose", directory=models_dir)
        self.refiner = None
        self.refine_labels = tuple(refine_labels)
        if refine_model:
            self.refiner = catalog.create(refine_model, task="segmentation", directory=models_dir)
            unknown = set(self.refine_labels) - set(self.refiner.labels)
            if unknown:
                self.close()
                raise catalog.CatalogError(f"Model {refine_model!r} has no label(s) {sorted(unknown)}; "
                                           f"its labels are {list(self.refiner.labels)}")
        self.kernel = np.ones((dilate, dilate), np.uint8) if dilate else None

    def close(self):
        self.pose.close()
        if self.refiner:
            self.refiner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, frame):
        mask = np.zeros(frame.shape[:2], np.uint8)
        poses = self.pose(frame)
        if not poses:
            return PersonMask(mask, None, None, 0.0)

        pose = poses[0]
        if pose.mask is not None:
            mask[pose.mask > 0] = 255
        draw_limbs(mask, pose, ARMS)
        if self.refiner is not None:
            mask |= segment_on_crop(self.refiner, frame, mask, self.refine_labels)
        if self.kernel is not None:
            mask = cv2.dilate(mask, self.kernel)

        pointing = pointing_hand(pose)
        if pointing is None:
            return PersonMask(mask, pose, None, 0.0)
        return PersonMask(mask, pose, pointing.tip, pointing.arm_angle)
