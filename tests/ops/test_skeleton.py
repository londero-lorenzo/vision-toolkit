import numpy as np

from vision_toolkit.ops.skeleton import draw_limbs, shoulder_width
from vision_toolkit.tasks.pose import Keypoint, Pose


def arm_pose(visibility=1.0):
    return Pose({
        "left_shoulder": Keypoint(100, 100), "right_shoulder": Keypoint(160, 100),
        "right_elbow": Keypoint(220, 100), "right_wrist": Keypoint(280, 100, visibility),
        "right_index": Keypoint(300, 100, visibility),
    })


def test_draws_arm_with_thickness_from_shoulders():
    pose = arm_pose()
    assert shoulder_width(pose) == 60
    mask = draw_limbs(np.zeros((200, 400), np.uint8), pose)
    assert mask[100, 190] == 255 and mask[100, 295] == 255
    thickness = (mask[:, 190] > 0).sum()
    assert 17 <= thickness <= 19  # 30% of the shoulder width


def test_skips_barely_visible_and_missing_keypoints():
    mask = draw_limbs(np.zeros((200, 400), np.uint8), arm_pose(visibility=0.2))
    assert mask[100, 190] == 255       # shoulder -> elbow drawn
    assert mask[100, 260] == 0         # elbow -> wrist skipped: wrist barely visible
    assert not mask[:, :150].any()     # left arm has no elbow: nothing drawn
