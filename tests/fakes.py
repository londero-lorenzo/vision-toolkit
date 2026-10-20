"""
Fake models and synthetic scenes for tests: they satisfy the task interfaces
without any backend, model file or private data.
"""
import cv2
import numpy as np

from vision_toolkit.tasks.base import Model
from vision_toolkit.tasks.detection import Detection
from vision_toolkit.tasks.pose import Keypoint, Pose
from vision_toolkit.tasks.segmentation import Segmentation

PERSON_BGR = (0, 140, 255)  # orange: appears nowhere else in the synthetic scenes
SKIN_BGR = (120, 160, 220)  # the skin of the synthetic faces: appears nowhere else either
FACE_RADIUS = 18


class ColorPose(Model):
    """
    PoseEstimator that sees a person as the pixels close to PERSON_BGR, drawn by
    draw_person: shoulders at the top of the body, hips at the bottom, the right
    fingertip at the end of the stretched arm and the left one along the side.
    """

    keypoint_names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                      "right_wrist", "left_index", "right_index", "left_hip", "right_hip")

    def __init__(self, tolerance=60, min_confidence=0.5):  # min_confidence: accepted as a real model does, unused
        self.tolerance = tolerance

    def __call__(self, image):
        mask = person_pixels(image, self.tolerance)
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


def person_pixels(image, tolerance=60):
    """uint8 (0/255) mask of the pixels close to PERSON_BGR."""
    diff = np.abs(image.astype(np.int16) - np.int16(PERSON_BGR)).max(axis=2)
    return (diff < tolerance).astype(np.uint8) * 255


class LargestPersonPose(ColorPose):
    """
    ColorPose that sees only the largest orange blob, as a pose model looking for
    one person picks one among several: a student in front, for instance.
    """

    def __call__(self, image):
        mask = person_pixels(image, self.tolerance)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []
        largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        only = image.copy()
        only[(mask > 0) & (labels != largest)] = 128
        return super().__call__(only)


class BlindSpotPose(ColorPose):
    """ColorPose that finds nobody while the person's centre is within `blind` (x0, x1): seen from the side, say."""

    def __init__(self, blind=(0, 0), **options):
        super().__init__(**options)
        self.blind = blind

    def __call__(self, image):
        poses = super().__call__(image)
        if poses:
            centre = (poses[0]["left_shoulder"].x + poses[0]["right_shoulder"].x) / 2
            if self.blind[0] <= centre < self.blind[1]:
                return []
        return poses


class HeadColorPose(ColorPose):
    """
    ColorPose with a head: the face keypoints of head_keypoints, on the face that
    draw_face puts just above the body (FACE_RADIUS + 2 px above its top).
    """

    keypoint_names = ColorPose.keypoint_names + ("nose", "left_eye_inner", "left_eye", "left_eye_outer",
                                                 "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
                                                 "right_ear", "mouth_left", "mouth_right")

    def __init__(self, tolerance=60, num_poses=1):
        super().__init__(tolerance)
        self.num_poses = num_poses

    def __call__(self, image):
        poses = super().__call__(image)
        for pose in poses:
            cx = (pose["left_shoulder"].x + pose["right_shoulder"].x) / 2
            top = pose["left_shoulder"].y - 26
            pose.keypoints.update(head_keypoints(cx, top - FACE_RADIUS - 2))
        return poses


def head_keypoints(cx, cy):
    """Face keypoints of a person looking at the camera, face centred on (cx, cy): ears 34 px apart."""
    points = {
        "nose": (0, 2), "left_eye_inner": (4, -5), "left_eye": (7, -5), "left_eye_outer": (10, -5),
        "right_eye_inner": (-4, -5), "right_eye": (-7, -5), "right_eye_outer": (-10, -5),
        "left_ear": (17, -2), "right_ear": (-17, -2), "mouth_left": (6, 9), "mouth_right": (-6, 9),
    }
    return {name: Keypoint(float(cx + dx), float(cy + dy)) for name, (dx, dy) in points.items()}


class SkinSegmenter(Model):
    """Segmenter that labels "face-skin" the pixels close to SKIN_BGR."""

    labels = ("background", "face-skin")

    def __init__(self, tolerance=40):
        self.tolerance = tolerance

    def __call__(self, image):
        diff = np.abs(image.astype(np.int16) - np.int16(SKIN_BGR)).max(axis=2)
        return Segmentation((diff < self.tolerance).astype(np.uint8), self.labels)


class FakeFaceDetector(Model):
    """Detector that always finds faces in the given boxes, with keypoints placed as on a frontal face."""

    labels = ("face",)

    def __init__(self, boxes=()):
        self.boxes = [tuple(box) for box in boxes]

    def __call__(self, image):
        detections = []
        for x0, y0, x1, y1 in self.boxes:
            w, h = x1 - x0, y1 - y0
            points = {"right_eye": (0.3, 0.4), "left_eye": (0.7, 0.4), "nose": (0.5, 0.6),
                      "mouth_right": (0.35, 0.8), "mouth_left": (0.65, 0.8)}
            keypoints = {name: Keypoint(x0 + fx * w, y0 + fy * h) for name, (fx, fy) in points.items()}
            detections.append(Detection((x0, y0, x1, y1), 0.9, "face", keypoints))
        return detections


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


def draw_face(image, cx, cy, radius=FACE_RADIUS, cell=4):
    """Skin-coloured face centred on (cx, cy), with a black and white checkerboard for its features."""
    cv2.circle(image, (cx, cy), radius, SKIN_BGR, -1)
    for y in range(cy - 10, cy + 10, cell):
        for x in range(cx - 10, cx + 10, cell):
            colour = 255 if ((x - cx) // cell + (y - cy) // cell) % 2 else 0
            image[y:y + cell, x:x + cell] = colour
    return image


# Synthetic lecture room: light wall, dark board seen in perspective, chalk lines
BOARD_QUAD = np.float32([[60, 50], [580, 70], [570, 300], [70, 290]])


def draw_room(size=(640, 400), writing=False, wall_marks=False):
    """The room; wall_marks adds light marks high on the wall, which show where the camera points."""
    w, h = size
    room = np.full((h, w, 3), 190, np.uint8)
    if wall_marks:
        for x in range(20, w - 30, 70):
            cv2.rectangle(room, (x, 3 + x % 3 * 2), (x + 26, 16 + x % 3 * 2), (240, 240, 235), -1)
    cv2.fillPoly(room, [np.int32(BOARD_QUAD)], (45, 50, 45))
    for i, y in enumerate((100, 140, 180)):
        cv2.line(room, (120, y + i * 3), (480 - i * 60, y + i * 5), (230, 230, 230), 4)
    if writing:
        cv2.line(room, (110, 245), (260, 250), (230, 230, 230), 6)
    return room
