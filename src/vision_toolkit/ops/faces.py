"""
Heads and faces from task results: where they are and which pixels to hide.

Every function takes either a Pose (its head keypoints) or a face Detection (its
box and keypoints), so any pose model and any face detector work alike.
"""
import math

import cv2
import numpy as np

from vision_toolkit.ops.crop_refine import segment_on_crop
from vision_toolkit.ops.skeleton import shoulder_width
from vision_toolkit.tasks.detection import Detection

# Keypoints on the head, as named by tasks.pose (MediaPipe names for the face points)
HEAD_KEYPOINTS = ("nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
                  "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right")
HEAD_PER_SHOULDERS = 0.45  # head width over shoulder width, for heads whose keypoints bunch up in profile


def head_width(subject):
    """
    Width of the head in pixels; None for a pose without head keypoints.

    A detection: the width of its box. A pose: the spread of its head keypoints, but
    at least HEAD_PER_SHOULDERS times the shoulder width, since in profile the head
    keypoints bunch up.
    """
    if isinstance(subject, Detection):
        return float(subject.box[2] - subject.box[0])
    points = _head_points(subject)
    if len(points) == 0:
        return None
    spread = float(np.ptp(points, axis=0).max())
    return max(spread, HEAD_PER_SHOULDERS * (shoulder_width(subject) or 0), 1.0)


def head_box(subject, scale=2.0):
    """
    Square box (x0, y0, x1, y1) centred on the face, with a side of `scale` head widths:
    at the default it holds the whole head with its hair. Not clipped to the image.
    None for a pose without head keypoints.
    """
    width = head_width(subject)
    if width is None:
        return None
    cx, cy = _face_centre(subject, width)
    side = max(1, round(scale * width))
    x0, y0 = round(cx - side / 2), round(cy - side / 2)
    return x0, y0, x0 + side, y0 + side


def face_mask(image, subject, segmenter=None, labels=("face-skin",), dilate=0.1):
    """
    uint8 (0/255) mask of the face of `subject` in `image`, empty without a head.

    The mask joins two parts and is then dilated by `dilate` head widths:
    - an ellipse over eyes, nose and mouth, which covers the features even when
      the segmenter misses a face that is small, blurred or in profile;
    - the `labels` pixels that `segmenter` finds on the crop around the head
      (selfie-multiclass: "face-skin"), which follow the visible skin and leave the
      hair out.
    Seen from behind, the ellipse covers part of the hair: an accepted cost.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    width = head_width(subject)
    if width is None:
        return mask
    (cx, cy), (a, b), angle = _face_ellipse(subject, width)
    cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(a)), max(1, round(b))), angle, 0, 360, 255, -1)
    if segmenter is not None:
        x0, y0, x1, y1 = head_box(subject)
        region = np.zeros_like(mask)
        region[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
        mask |= segment_on_crop(segmenter, image, region, labels, margin=0)
    k = max(3, int(dilate * width) | 1)
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def _head_points(pose):
    return np.array([pose[name].xy for name in HEAD_KEYPOINTS if name in pose], np.float64).reshape(-1, 2)


def _face_centre(subject, width):
    if isinstance(subject, Detection):
        x0, y0, x1, y1 = subject.box
        return (x0 + x1) / 2, (y0 + y1) / 2
    nose = subject.get("nose")
    if nose is not None:
        return nose.x, nose.y + 0.05 * width
    return tuple(_head_points(subject).mean(axis=0))


def _face_ellipse(subject, width):
    """Centre, semi-axes and angle (degrees) of an ellipse from the eyebrows to the chin."""
    centre = _face_centre(subject, width)
    if isinstance(subject, Detection):
        x0, y0, x1, y1 = subject.box
        return centre, (0.6 * (x1 - x0), 0.6 * (y1 - y0)), 0.0
    left, right = subject.get("left_eye"), subject.get("right_eye")
    angle = math.degrees(math.atan2(left.y - right.y, left.x - right.x)) if left and right else 0.0
    return centre, (0.5 * width, 0.65 * width), angle
