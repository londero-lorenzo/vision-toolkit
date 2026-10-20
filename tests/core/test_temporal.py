import warnings

import cv2
import numpy as np

from vision_toolkit.core.geometry import rectify_homography
from vision_toolkit.core.temporal import PlaneMedian, _nan_median

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


def test_median_ignores_nan_like_numpy():
    rng = np.random.default_rng(1)
    buffer = rng.uniform(0, 255, (9, 40, 70, 3)).astype(np.float32)
    buffer[rng.uniform(size=(9, 40, 70)) < 0.4] = np.nan
    buffer[:, :, :5] = np.nan                                # never seen
    median, count = _nan_median(buffer, rows=16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the pixels never seen
        expected = np.nanmedian(buffer, axis=0)
    seen = count > 0
    assert not seen[:, :5].any() and seen[:, 5:].mean() > 0.99
    np.testing.assert_allclose(median[seen], expected[seen], rtol=1e-6)


def test_exposure_change_leaves_the_background_alone():
    background = (scene() * 0.7 + 20).astype(np.uint8)  # no level clips once 10% brighter
    H, size = rectify_homography(QUAD)
    empty = np.zeros(background.shape[:2], np.uint8)
    reference = cv2.warpPerspective(background, H, size)
    for match, expected in ((True, False), (False, True)):
        rec = PlaneMedian(H, size, window=3, change_threshold=8, exposure_reference=reference if match else None)
        rec.update(background, empty)
        rec.take_snapshot()
        brighter = np.clip(background * 1.1, 0, 255).astype(np.uint8)
        for x in (30, 150, 210):
            rec.update(*occlude(brighter, x))
        assert (rec.change_since_snapshot() > 0.05) == expected
        assert abs(rec.gain - 1.1) < 0.02 if match else rec.gain == 1.0


def test_homography_can_change_between_updates():
    # The camera moves: the same plane is seen through a new homography
    background = scene()
    H, size = rectify_homography(QUAD)
    rec = PlaneMedian(H, size, window=3)
    rec.update(background, np.zeros(background.shape[:2], np.uint8))
    rec.take_snapshot()
    moved = cv2.warpAffine(background, np.float32([[1, 0, 7], [0, 1, 3]]), background.shape[1::-1])
    rec.H, _ = rectify_homography(QUAD + [7, 3])
    rec.update(moved, np.zeros(background.shape[:2], np.uint8))
    rec.update(moved, np.zeros(background.shape[:2], np.uint8))
    assert rec.change_since_snapshot() < 0.01
