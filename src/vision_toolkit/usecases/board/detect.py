import cv2
import numpy as np

from vision_toolkit.core.geometry import order_corners


def find_board(background, floor=None):
    """
    The board in a background frame (a median): detect_board, then refine_quad.

    floor: the audience line. Only the rows above it are searched: below it the frame
    may be blacked out, and a black band would look like a board.

    Returns:
        quad: (4, 2) float32 array ordered tl, tr, br, bl, or None
    """
    region = background if floor is None else background[:floor]
    quad = detect_board(region)
    return None if quad is None else refine_quad(region, quad)


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
    # Padded with copies of the edge, so that a board near the edge is not glued to it
    k = 61
    dark = cv2.copyMakeBorder(dark, k, k, k, k, cv2.BORDER_REPLICATE)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))[k:-k, k:-k]

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


def refine_quad(image, quad, search=None, light_strip=15.0, samples=48):
    """
    Move each side of a board quadrilateral (tl, tr, br, bl) to the board's real edge.

    Otsu leaves out whatever of the board looks light: the glow of a light strip
    over it, a light frame. Along the outward normal of each side, at `samples`
    points, the edge is the strongest dark-to-light step within `search` pixels
    outward (default 3.5% of the image width) and a quarter of that inward. On the
    top side a light strip may run along the board: where the profile peaks more
    than `light_strip` levels above the wall, the edge is the peak. A line is
    fitted through each side's points (Huber, so stray points weigh little), and
    the corners are where the lines meet. A quad with a side shorter than a pixel,
    or whose fitted sides do not meet, is returned as it is.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    search = search or max(10, round(0.035 * image.shape[1]))
    tl, tr, br, bl = np.float32(quad)
    if min(np.linalg.norm(b - a) for a, b in ((tl, tr), (tr, br), (bl, br), (tl, bl))) < 1:
        return np.float32(quad)
    centre = (tl + tr + br + bl) / 4
    top = _side(gray, tl, tr, centre, search, samples, light_strip)
    right = _side(gray, tr, br, centre, search, samples)
    bottom = _side(gray, bl, br, centre, search, samples)
    left = _side(gray, tl, bl, centre, search, samples)
    try:
        return np.float32([_meet(top, left), _meet(top, right), _meet(bottom, right), _meet(bottom, left)])
    except np.linalg.LinAlgError:
        return np.float32(quad)


def _side(gray, a, b, centre, search, samples, light_strip=None):
    """Line (point, direction) through the edge points found along the side a-b."""
    direction = (b - a) / np.linalg.norm(b - a)
    normal = np.float32([-direction[1], direction[0]])
    if np.dot((a + b) / 2 - centre, normal) < 0:
        normal = -normal  # outward
    offsets = np.arange(-(search // 4), search + 1, dtype=np.float32)
    points = []
    for f in np.linspace(0.06, 0.94, samples):
        p = a + f * (b - a)
        profile = _profile(gray, p, normal, offsets)
        i = int(np.diff(profile).argmax())
        if light_strip is not None:
            wall = np.median(profile[-max(3, len(profile) // 4):])
            if profile.max() - wall > light_strip:
                i = int(profile.argmax())
        points.append(p + offsets[i] * normal)
    vx, vy, x0, y0 = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.float32([x0, y0]), np.float32([vx, vy])


def _profile(gray, point, normal, offsets):
    """Gray levels along the normal through point, slightly smoothed."""
    xs = np.clip(point[0] + offsets * normal[0], 0, gray.shape[1] - 1).astype(np.float32)
    ys = np.clip(point[1] + offsets * normal[1], 0, gray.shape[0] - 1).astype(np.float32)
    profile = cv2.remap(gray, xs[None, :], ys[None, :], cv2.INTER_LINEAR)
    return cv2.GaussianBlur(profile, (5, 1), 0).ravel()


def _meet(line1, line2):
    (p, d), (q, e) = line1, line2
    s = np.linalg.solve(np.array([d, -e]).T, q - p)
    return p + s[0] * d
