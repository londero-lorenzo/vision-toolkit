from dataclasses import dataclass

import numpy as np


@dataclass
class Pointing:
    tip: tuple            # (x, y) pixels of the fingertip (the wrist if the model has no fingers)
    arm_angle: float      # degrees between arm and torso: ~0 along the side, ~90 stretched out
    side: str             # "left" or "right"


def arm_angle(pose, side):
    """
    Angle in degrees between the arm (shoulder -> fingertip) and the torso
    (shoulder -> hip) on one side, or None if a keypoint is missing.
    """
    shoulder, hip, tip = pose.get(f"{side}_shoulder"), pose.get(f"{side}_hip"), _tip(pose, side)
    if shoulder is None or hip is None or tip is None:
        return None
    arm = np.subtract(tip.xy, shoulder.xy)
    torso = np.subtract(hip.xy, shoulder.xy)
    cos = arm @ torso / (np.linalg.norm(arm) * np.linalg.norm(torso) or 1.0)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def pointing_hand(pose):
    """
    The more raised arm and its fingertip, or None if neither arm is complete.

    The arm length is not enough: an arm along the side is as long as a stretched
    one. The angle with the torso tells them apart.
    """
    angles = {side: arm_angle(pose, side) for side in ("left", "right")}
    angles = {side: a for side, a in angles.items() if a is not None}
    if not angles:
        return None
    side = max(angles, key=angles.get)
    x, y = _tip(pose, side).xy
    return Pointing((int(x), int(y)), angles[side], side)


def _tip(pose, side):
    return pose.get(f"{side}_index") or pose.get(f"{side}_wrist")
