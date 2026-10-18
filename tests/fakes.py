"""
Fake models and synthetic scenes for tests: they satisfy the task interfaces
without any backend, model file or private data.
"""
import cv2
import numpy as np

from vision_toolkit.tasks.base import Model
from vision_toolkit.tasks.pose import Keypoint, Pose
from vision_toolkit.tasks.segmentation import Segmentation

PERSON_BGR = (0, 140, 255)  # orange: appears nowhere else in the synthetic scenes


class ColorPose(Model):
    """
    PoseEstimator that sees a person as the pixels close to PERSON_BGR, drawn by
    draw_person: shoulders at the top of the body, hips at the bottom, the right
    fingertip at the end of the stretched arm and the left one along the side.
    """

    keypoint_names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                      "right_wrist", "left_index", "right_index", "left_hip", "right_hip")

    def __init__(self, tolerance=60):
        self.tolerance = tolerance

    def __call__(self, image):
        diff = np.abs(image.astype(np.int16) - np.int16(PERSON_BGR)).max(axis=2)
        mask = (diff < self.tolerance).astype(np.uint8) * 255
        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            return []
        top, bottom = ys.min(), ys.max()
        body = xs[ys > top + 40]
        left, right = float(np.percentile(body, 2)), float(np.percentile(body, 98))
        shoulder_y = top + 26
        tip_x = float(xs[np.abs(ys - shoulder_y) <= 2].max())
        kp = {
            "left_shoulder": (left, shoulder_y), "right_shoulder": (right, shoulder_y),
            "left_elbow": (left, top + 80), "right_elbow": ((right + tip_x) / 2, shoulder_y),
            "left_wrist": (left, top + 130), "right_wrist": (tip_x - 10, shoulder_y),
            "left_index": (left, top + 140), "right_index": (tip_x, shoulder_y),
            "left_hip": (left, bottom - 10), "right_hip": (right, bottom - 10),
        }
        return [Pose({name: Keypoint(float(x), float(y)) for name, (x, y) in kp.items()}, mask)]


class SmallInputSegmenter(Model):
    """
    Segmenter with a tiny input, like DeepLab V3 at 257x257 but worse: the image is
    shrunk to `size` x `size`, bright pixels are "person", the labels are scaled back.
    A small subject averages away in the full image and is found only on a crop.
    """

    labels = ("background", "person")

    def __init__(self, size=32):
        self.size = size

    def __call__(self, image):
        h, w = image.shape[:2]
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (self.size, self.size),
                           interpolation=cv2.INTER_AREA)
        category = (small > 128).astype(np.uint8)
        return Segmentation(cv2.resize(category, (w, h), interpolation=cv2.INTER_NEAREST), self.labels)


def draw_person(image, cx, top=120, bottom=380, half_width=30, arm_length=100):
    """Orange body from top to bottom, centred on cx, with the right arm stretched out."""
    cv2.rectangle(image, (cx - half_width, top), (cx + half_width, bottom), PERSON_BGR, -1)
    cv2.rectangle(image, (cx + half_width, top + 20), (cx + half_width + arm_length, top + 32), PERSON_BGR, -1)
    return image


# Synthetic lecture room: light wall, dark board seen in perspective, chalk lines
BOARD_QUAD = np.float32([[60, 50], [580, 70], [570, 300], [70, 290]])


def draw_room(size=(640, 400), writing=False):
    w, h = size
    room = np.full((h, w, 3), 190, np.uint8)
    cv2.fillPoly(room, [np.int32(BOARD_QUAD)], (45, 50, 45))
    for i, y in enumerate((100, 140, 180)):
        cv2.line(room, (120, y + i * 3), (480 - i * 60, y + i * 5), (230, 230, 230), 4)
    if writing:
        cv2.line(room, (110, 245), (260, 250), (230, 230, 230), 6)
    return room
