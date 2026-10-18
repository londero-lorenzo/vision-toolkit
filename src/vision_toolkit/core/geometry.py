import json

import cv2
import numpy as np


def order_corners(pts):
    """Order 4 points as tl, tr, br, bl."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def polygon_mask(shape, polygon):
    """uint8 (0/255) mask of the polygon area in an image of the given shape."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(polygon)], 255)
    return mask


def rectify_homography(quad, width=None):
    """
    Homography from the frame to the rectified plane of a quadrilateral (tl, tr, br, bl).

    If width is not given, it is the mean length of the top and bottom sides;
    the height follows the mean aspect ratio of the quadrilateral's sides.

    Returns:
        H, (width, height)
    """
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width is None:
        width = int(round(w))
    height = int(round(width * h / w))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(np.float32(quad), dst), (width, height)


def to_plane(point, H, size):
    """
    Map a frame point to integer coordinates on the rectified plane.

    Returns None if the point is None or falls outside the plane.
    """
    if point is None:
        return None
    p = cv2.perspectiveTransform(np.float32([[point]]), H)[0, 0]
    w, h = size
    if 0 <= p[0] < w and 0 <= p[1] < h:
        return int(p[0]), int(p[1])
    return None


def save_quad(path, quad):
    """Save a quadrilateral (manual or automatic calibration) as JSON {"quad": [[x, y], ...]}."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"quad": np.asarray(quad).tolist()}, f, indent=2)


def load_quad(path):
    with open(path, encoding="utf-8") as f:
        return np.float32(json.load(f)["quad"])
