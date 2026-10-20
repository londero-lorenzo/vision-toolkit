import cv2
import numpy as np

from fakes import SKIN_BGR, SkinSegmenter, head_keypoints
from vision_toolkit.ops.faces import face_mask, head_box, head_width
from vision_toolkit.tasks.detection import Detection
from vision_toolkit.tasks.pose import Keypoint, Pose

FEATURES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


def face_pose(cx=100, cy=80, shoulders=None):
    keypoints = head_keypoints(cx, cy)
    if shoulders:
        keypoints["left_shoulder"] = Keypoint(cx + shoulders / 2, cy + 60)
        keypoints["right_shoulder"] = Keypoint(cx - shoulders / 2, cy + 60)
    return Pose(keypoints)


def test_head_width_from_keypoints_shoulders_and_boxes():
    assert head_width(face_pose()) == 34  # ear to ear
    assert head_width(face_pose(shoulders=200)) == 90  # bunched-up head keypoints: 0.45 x shoulders
    assert head_width(Detection((10, 20, 40, 60), 0.9, "face")) == 30
    assert head_width(Pose({"left_shoulder": Keypoint(0, 0)})) is None


def test_head_box_is_centred_on_the_face():
    x0, y0, x1, y1 = head_box(face_pose(100, 80))
    assert x1 - x0 == y1 - y0 == 68
    assert abs((x0 + x1) / 2 - 100) <= 1 and abs((y0 + y1) / 2 - 80) <= 4
    assert head_box(Detection((10, 20, 40, 50), 0.9, "face"), scale=1.0) == (10, 20, 40, 50)
    assert head_box(Pose()) is None


def test_ellipse_covers_the_features_without_a_segmenter():
    image = np.full((200, 200, 3), 190, np.uint8)
    pose = face_pose(100, 80)
    mask = face_mask(image, pose)
    assert all(mask[int(pose[name].y), int(pose[name].x)] == 255 for name in FEATURES)
    assert mask[80, 160] == 0 and mask[150, 100] == 0  # nothing far from the face
    assert face_mask(image, Pose()).max() == 0


def test_segmenter_adds_the_skin_around_the_head_only():
    image = np.full((200, 200, 3), 190, np.uint8)
    pose = face_pose(100, 80)
    cv2.rectangle(image, (68, 70), (78, 90), SKIN_BGR, -1)    # skin beside the face, inside the head box
    cv2.rectangle(image, (170, 70), (190, 90), SKIN_BGR, -1)  # skin far away, outside the head box
    alone = face_mask(image, pose)
    joined = face_mask(image, pose, SkinSegmenter())
    assert alone[80, 72] == 0 and joined[80, 72] == 255
    assert joined[80, 180] == 0
    assert (joined >= alone).all()

    empty = face_mask(np.full((200, 200, 3), 190, np.uint8), pose, SkinSegmenter())
    np.testing.assert_array_equal(empty, alone)  # a segmenter that finds nothing leaves the ellipse


def test_detection_mask_covers_its_box():
    image = np.full((100, 100, 3), 190, np.uint8)
    mask = face_mask(image, Detection((40, 30, 60, 56), 0.9, "face"))
    assert mask[43, 50] == 255 and mask[52, 42] == 255
    assert mask[90, 90] == 0
