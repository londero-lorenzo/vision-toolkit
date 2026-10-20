"""
When a frame is doubtful, and which doubtful frames go to the person.

A frame is doubtful when it may show a face that the pipeline did not hide:
foreground that no pose with a head explains, a face away from every pose, a
head cut by the frame edge. Doubtful frames close in time show the same scene,
so they are grouped into episodes and one frame per episode is reviewed.
"""
import cv2
import numpy as np

PERSON_WITHOUT_HEAD = "person-without-head"
FACE_OUTSIDE_POSE = "face-outside-pose"
HEAD_AT_EDGE = "head-at-edge"


def foreground_mask(frame, background, threshold=30, open_fraction=0.006):
    """
    uint8 (0/255) mask of what differs from the background.

    The frame is first divided by its global exposure gain against the background
    (lecture videos drift by a few percent), then pixels differing by more than
    `threshold` are kept. An opening of `open_fraction` times the frame width removes
    chalk strokes and other thin changes, and keeps person-sized ones. Pixels black
    in the background (a redacted band) never count.
    """
    valid = background.max(axis=2) > 10
    if not valid.any():
        return np.zeros(frame.shape[:2], np.uint8)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[valid]
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)[valid]
    gain = max(float(np.median(gray_frame)), 1.0) / max(float(np.median(gray_background)), 1.0)
    diff = np.abs(frame.astype(np.float32) / gain - background.astype(np.float32)).max(axis=2)
    mask = ((diff > threshold) & valid).astype(np.uint8) * 255
    k = max(3, int(open_fraction * frame.shape[1]))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def unexplained_blobs(foreground, explained, min_area):
    """Boxes (x0, y0, x1, y1) of the foreground blobs outside `explained` with at least `min_area` pixels."""
    rest = cv2.bitwise_and(foreground, cv2.bitwise_not(explained))
    _, _, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, area in stats[1:] if area >= min_area]


def episodes(items, gap, key=lambda item: item):
    """Split items into episodes: sorted by `key`, consecutive items of an episode are at most `gap` apart."""
    groups = []
    for item in sorted(items, key=key):
        if groups and key(item) - key(groups[-1][-1]) <= gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def select(groups, budget=None):
    """
    The episodes to review: all of them without a budget, otherwise at most `budget`
    of them, evenly spread over time (the first and the last are kept).
    """
    if budget is None or len(groups) <= budget:
        return list(groups)
    if budget <= 0:
        return []
    if budget == 1:
        return [groups[len(groups) // 2]]
    picks = np.floor(np.linspace(0, len(groups) - 1, budget) + 0.5).astype(int)
    return [groups[i] for i in picks]


def representative(group):
    """The frame of an episode that goes to review: the middle one."""
    return group[len(group) // 2]
