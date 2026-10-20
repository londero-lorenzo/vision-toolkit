from dataclasses import dataclass

import cv2
import numpy as np

from vision_toolkit.core.foreground import foreground_mask
from vision_toolkit.models import catalog
from vision_toolkit.ops.crop_refine import segment_on_crop
from vision_toolkit.ops.pointing import pointing_hand
from vision_toolkit.ops.skeleton import ARMS, draw_limbs
from vision_toolkit.tasks.pose import Pose

DEFAULT_POSE_MODEL = "mediapipe-pose-heavy"
POSE = "pose"
FOREGROUND = "foreground"


@dataclass
class PersonMask:
    mask: np.ndarray            # uint8 0/255, already dilated; empty if nobody is in view
    pose: Pose | None
    pointer: tuple | None       # (x, y) fingertip of the more raised arm
    arm_angle: float            # degrees between arm and torso: ~0 at rest, ~90 stretched out
    source: str | None = None   # POSE, FOREGROUND, or None when nobody was found


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

    When the pose finds nobody and a background is given, the lecturer may still be
    there, seen from the side or in poor light: the mask is then the box of the
    largest foreground blob shaped like a standing person (person_blob). It has no
    pose, so no pointed spot.
    """

    def __init__(self, pose_model=DEFAULT_POSE_MODEL, refine_model=None, refine_labels=("person",),
                 dilate=15, models_dir=None, pose_options=None):
        self.pose = catalog.create(pose_model, task="pose", directory=models_dir, **(pose_options or {}))
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

    def __call__(self, frame, background=None, floor=None):
        """
        frame: what the models see, with the audience already blacked out.
        background: the same view without the lecturer (a median), for the fallback.
        floor: the row where the lecturer's blob ends, the audience line; the frame
        bottom by default.
        """
        mask = np.zeros(frame.shape[:2], np.uint8)
        poses = self.pose(frame)
        if not poses:
            if background is None:
                return PersonMask(mask, None, None, 0.0)
            mask = person_blob(foreground_mask(frame, background), floor)
            if not mask.any():
                return PersonMask(mask, None, None, 0.0)
            return PersonMask(self._dilate(mask), None, None, 0.0, FOREGROUND)

        pose = poses[0]
        if pose.mask is not None:
            mask[pose.mask > 0] = 255
        draw_limbs(mask, pose, ARMS)
        if self.refiner is not None:
            mask |= segment_on_crop(self.refiner, frame, mask, self.refine_labels)
        mask = self._dilate(mask)

        pointing = pointing_hand(pose)
        if pointing is None:
            return PersonMask(mask, pose, None, 0.0, POSE)
        return PersonMask(mask, pose, pointing.tip, pointing.arm_angle, POSE)

    def _dilate(self, mask):
        return cv2.dilate(mask, self.kernel) if self.kernel is not None else mask


def person_blob(foreground, floor=None, min_height=0.15, reach=0.03):
    """
    Filled box (uint8, 0/255) of the largest foreground blob shaped like a standing
    person: at least `min_height` of the frame tall, and reaching down to within
    `reach` of the frame height from `floor` (the frame bottom by default), where a
    standing person is cut by the audience line or the frame. Changes on the board
    itself, such as a large erased area, never reach that low. Empty if there is none.
    """
    h = foreground.shape[0]
    floor = h if floor is None else min(h, floor)
    out = np.zeros_like(foreground)
    _, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    people = [s for s in stats[1:] if s[3] >= min_height * h and s[1] + s[3] >= floor - reach * h]
    if people:
        x, y, w, bh, _ = max(people, key=lambda s: s[4])
        out[y:y + bh, x:x + w] = 255
    return out
