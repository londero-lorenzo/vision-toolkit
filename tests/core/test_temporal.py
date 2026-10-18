import cv2
import numpy as np

from vision_toolkit.core.geometry import rectify_homography
from vision_toolkit.core.temporal import PlaneMedian

QUAD = np.float32([[40, 30], [280, 40], [270, 200], [50, 190]])


def scene():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


def occlude(background, x):
    frame, mask = background.copy(), np.zeros(background.shape[:2], np.uint8)
    cv2.rectangle(frame, (x, 20), (x + 60, 220), (0, 0, 255), -1)
    cv2.rectangle(mask, (x, 20), (x + 60, 220), 255, -1)
    return frame, mask


def test_recovers_background_behind_moving_occluder():
    background = scene()
    H, size = rectify_homography(QUAD)
    rec = PlaneMedian(H, size, window=5)
    for x in (30, 90, 150, 210, 150):
        rec.update(*occlude(background, x))

    truth = cv2.warpPerspective(background, H, size).astype(np.float32)
    single = cv2.warpPerspective(occlude(background, 150)[0], H, size).astype(np.float32)
    assert rec.seen.all()
    assert (np.abs(rec.background - truth).max(axis=2) > 2).mean() < 0.01
    assert (np.abs(single - truth).max(axis=2) > 2).mean() > 0.1


def test_unseen_pixels_get_unseen_color():
    background = scene()
    H, size = rectify_homography(QUAD)
    rec = PlaneMedian(H, size, window=3)
    occluded = rec.update(*occlude(background, 100))
    assert occluded.any() and not rec.seen.all()
    assert (rec.image()[~rec.seen] == (255, 0, 255)).all()


def test_change_since_snapshot():
    background = scene()
    H, size = rectify_homography(QUAD)
    rec = PlaneMedian(H, size, window=1)
    empty = np.zeros(background.shape[:2], np.uint8)
    rec.update(background, empty)
    assert rec.change_since_snapshot() == 1.0
    rec.take_snapshot()
    assert rec.change_since_snapshot() == 0.0

    written = background.copy()
    cv2.rectangle(written, (100, 80), (180, 120), (255, 255, 255), -1)
    rec.update(written, empty)
    assert 0.02 < rec.change_since_snapshot() < 0.5
