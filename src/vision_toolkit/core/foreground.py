"""
A frame against a static background of the same scene: how much brighter or darker
the frame is overall, and which parts of it differ.
"""
import cv2
import numpy as np


def exposure_gain(image, reference, mask=None, min_level=20, samples=50_000, limits=(0.5, 2.0)):
    """
    Exposure gain of `image` against `reference`: the median ratio of their gray levels.

    Only pixels in `mask` (every pixel by default) where the reference is brighter
    than `min_level` count, at most `samples` of them, evenly picked. The median
    ignores the minority of pixels whose content changed. The gain is clipped to
    `limits`: a larger jump is not an exposure change. 1.0 if no pixel qualifies.
    """
    gray, ref = _gray(image).ravel(), _gray(reference).ravel()
    usable = ref > min_level
    if mask is not None:
        usable &= np.asarray(mask).ravel() > 0
    index = np.flatnonzero(usable)
    if not len(index):
        return 1.0
    index = index[::max(1, len(index) // samples)]
    return float(np.clip(np.median(gray[index] / ref[index]), *limits))


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


def _gray(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)
