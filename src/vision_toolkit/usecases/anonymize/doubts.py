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
