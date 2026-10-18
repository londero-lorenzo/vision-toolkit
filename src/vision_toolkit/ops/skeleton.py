import cv2
import numpy as np

# Keypoint chains drawn as limbs: shoulder, elbow, wrist, index fingertip
ARMS = (
    ("left_shoulder", "left_elbow", "left_wrist", "left_index"),
    ("right_shoulder", "right_elbow", "right_wrist", "right_index"),
)


def shoulder_width(pose):
    """Distance between the shoulders in pixels, or None if one is missing."""
    left, right = pose.get("left_shoulder"), pose.get("right_shoulder")
    if left is None or right is None:
        return None
    return float(np.linalg.norm(np.subtract(_px(left), _px(right))))


def draw_limbs(mask, pose, chains=ARMS, thickness=None, min_visibility=0.5):
    """
    Draw limbs as thick segments between named keypoints onto a uint8 mask, in place.

    A body mask sometimes loses part of a stretched arm; the skeleton does not
    depend on appearance and covers it. thickness defaults to 30% of the shoulder
    width (at least 3 px). Segments with a missing or barely visible end are skipped.
    """
    if thickness is None:
        thickness = max(3, int(0.3 * (shoulder_width(pose) or 0)))
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            ka, kb = pose.get(a), pose.get(b)
            if ka and kb and ka.visibility > min_visibility and kb.visibility > min_visibility:
                cv2.line(mask, _px(ka), _px(kb), 255, thickness)
    return mask


def _px(keypoint):
    return int(keypoint.x), int(keypoint.y)
