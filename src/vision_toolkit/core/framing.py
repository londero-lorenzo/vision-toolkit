"""
Camera framing over a video: the stretches where the camera stays still.

A fixed camera can still be moved by hand during a recording, and a recording can
open on a title card. Frames are aligned (rotation and shift, ECC) on a mask that
only static parts of the scene reach, such as the wall above a board. Each stretch
with one framing is a segment; the frames where the camera is in motion belong to none.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision_toolkit.core.io import FrameReader

_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)


@dataclass(frozen=True)
class Motion:
    dx: float       # pixels
    dy: float
    angle: float    # degrees

    @property
    def shift(self):
        return math.hypot(self.dx, self.dy)


@dataclass(frozen=True)
class Framing:
    start: float        # seconds; the segment covers [start, end)
    end: float
    reference: float    # time of the frame the segment is aligned to

    @property
    def duration(self):
        return self.end - self.start


def camera_motion(reference, image, mask=None):
    """
    Rotation and shift that align `image` to `reference` (ECC, euclidean motion),
    measured on `mask` (uint8, 0/255). None if the alignment does not converge.
    """
    return _align(_prepare(reference), _prepare(image), mask)


def framing_segments(path, mask=None, step=15.0, start=0.0, end=None, crop=None,
                     max_shift=2.0, max_angle=0.2, persist=2, precision=1.0):
    """
    Split [start, end) of a video into segments with one camera framing each.

    A frame every `step` seconds is aligned to the reference of the current segment,
    its first frame. The camera moved when the alignment leaves the reference by more
    than `max_shift` pixels or `max_angle` degrees, or fails, for `persist` frames in a
    row: one odd frame is not a move. The last of those frames becomes the reference
    of the new segment. Each boundary is then narrowed to `precision` seconds by
    bisection: the old segment ends where the frames stop matching its reference, the
    new one starts where they match the new reference, and the frames in between are
    the camera in motion.

    Without texture in the mask (a flat wall) a move cannot be seen: one segment.
    Returns a list of Framing in time order.
    """
    with FrameReader(path, crop) as reader:
        end = reader.duration if end is None else min(end, reader.duration)

        def frame(t):
            image = reader.at(t)
            return None if image is None else _prepare(image)

        def same(reference, t):
            image = frame(t)
            motion = None if image is None else _align(reference, image, mask)
            return motion is not None and motion.shift <= max_shift and abs(motion.angle) <= max_angle

        ref_t, ref = start, frame(start)
        if ref is None or not _has_texture(ref, mask):
            return [Framing(start, end, start)]
        segments, seg_start, last_same, pending = [], start, ref_t, []
        for t in np.arange(start + step, end, step):
            t = float(t)
            if frame(t) is None:
                continue
            if same(ref, t):
                last_same, pending = t, []
                continue
            pending.append(t)
            if len(pending) < persist:
                continue
            new_t = pending[-1]
            new = frame(new_t)
            left = _boundary(last_same, pending[0], lambda s: same(ref, s), precision)
            arrived = max(left, _boundary(last_same, new_t, lambda s: not same(new, s), precision))
            segments.append(Framing(seg_start, left, ref_t))
            seg_start, ref_t, ref, last_same, pending = arrived, new_t, new, new_t, []
        segments.append(Framing(seg_start, end, ref_t))
    return segments


def _boundary(before, after, holds, precision):
    """
    The time where `holds` stops holding, between `before` (where it holds) and
    `after` (where it does not), to `precision` seconds: the first time known not to hold.
    """
    while after - before > precision:
        middle = (before + after) / 2
        if holds(middle):
            before = middle
        else:
            after = middle
    return round(after, 2)


def _prepare(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)


def _align(reference, image, mask):
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(reference, image, warp, cv2.MOTION_EUCLIDEAN, _CRITERIA, mask, 5)
    except cv2.error:
        return None
    return Motion(float(warp[0, 2]), float(warp[1, 2]), math.degrees(math.atan2(warp[1, 0], warp[0, 0])))


def _has_texture(gray, mask, edge=8.0, fraction=0.005):
    """Whether enough of the mask has edges for an alignment: a gradient above `edge` levels per pixel."""
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)) / 8
    inside = magnitude if mask is None else magnitude[mask > 0]
    return inside.size > 0 and float((inside > edge).mean()) >= fraction
