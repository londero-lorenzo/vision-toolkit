import pytest

from vision_toolkit.ops.pointing import arm_angle, pointing_hand
from vision_toolkit.tasks.pose import Keypoint, Pose


def person(right_tip, left_tip=(80, 200), fingers=True):
    """Front view: shoulders at y=100, hips at y=250; tips given in pixels."""
    kp = {
        "left_shoulder": Keypoint(80, 100), "right_shoulder": Keypoint(140, 100),
        "left_hip": Keypoint(85, 250), "right_hip": Keypoint(135, 250),
    }
    tip = "index" if fingers else "wrist"
    kp[f"left_{tip}"] = Keypoint(*left_tip)
    kp[f"right_{tip}"] = Keypoint(*right_tip)
    return Pose(kp)


def test_stretched_arm_wins_over_hanging_arm():
    pose = person(right_tip=(300, 100))
    assert arm_angle(pose, "right") == pytest.approx(90, abs=3)
    assert arm_angle(pose, "left") < 5
    p = pointing_hand(pose)
    assert p.side == "right" and p.tip == (300, 100) and p.arm_angle == pytest.approx(90, abs=3)


def test_falls_back_to_wrist_without_fingers():
    # COCO-style models have wrists but no fingertips
    p = pointing_hand(person(right_tip=(250, 40), fingers=False))
    assert p.side == "right" and p.tip == (250, 40) and p.arm_angle > 100


def test_incomplete_pose_gives_none():
    assert pointing_hand(Pose({"left_shoulder": Keypoint(0, 0)})) is None
