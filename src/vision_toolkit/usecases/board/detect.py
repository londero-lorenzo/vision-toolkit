import cv2
import numpy as np

from vision_toolkit.core.geometry import order_corners


def detect_board(frame, person_mask=None):
    """
    Find the board as a quadrilateral in the frame.

    Assumes a dark board on a light wall (Otsu threshold on the gray image).
    If the person mask is given, its pixels are removed before looking for the
    contour: otherwise the legs below the board distort the quadrilateral.

    Returns:
        quad: (4, 2) float32 array ordered tl, tr, br, bl, or None
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if person_mask is not None:
        dark[person_mask > 0] = 0

    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))

    # Raise the tolerance until the contour shrinks to 4 vertices
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            return order_corners(quad.reshape(4, 2))
    return None
