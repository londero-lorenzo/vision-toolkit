import numpy as np

from vision_toolkit.core.foreground import exposure_gain, foreground_mask


def test_foreground_ignores_exposure_drift_and_thin_strokes():
    background = np.full((200, 300, 3), 150, np.uint8)
    frame = (background * 0.94).astype(np.uint8)  # the video got 6% darker
    frame[50:90, 100:140] = 20                    # something person-sized
    frame[150:152, 20:280] = 255                  # a chalk stroke
    mask = foreground_mask(frame, background)
    assert mask[70, 120] == 255
    assert mask[151, 150] == 0 and mask[20, 20] == 0

    background[180:] = 0                          # a redacted band never counts
    frame[180:] = 0
    assert foreground_mask(frame, background)[190, 150] == 0


def test_exposure_gain_is_the_ratio_of_the_unchanged_pixels():
    rng = np.random.default_rng(0)
    reference = rng.integers(40, 220, (100, 120, 3)).astype(np.uint8)
    image = (reference * 1.06).clip(0, 255).astype(np.float32)
    image[:20] = 255                                  # a minority of pixels changed content
    assert abs(exposure_gain(image, reference) - 1.06) < 0.01

    mask = np.zeros((100, 120), bool)
    mask[50:] = True
    image[50:] = reference[50:] * 0.9
    assert abs(exposure_gain(image, reference, mask) - 0.9) < 0.01


def test_exposure_gain_without_usable_pixels_or_with_a_jump():
    reference = np.zeros((10, 10, 3), np.uint8)       # too dark to measure
    assert exposure_gain(np.full((10, 10, 3), 50, np.uint8), reference) == 1.0
    reference[:] = 100
    assert exposure_gain(np.full((10, 10, 3), 250, np.uint8), reference) == 2.0  # clipped: not an exposure change
