import numpy as np

from vision_toolkit.core.redact import pixelate


def noisy(shape=(60, 80, 3)):
    return np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)


def test_only_masked_pixels_change():
    image = noisy()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:30, 20:50] = 255
    out = pixelate(image, mask, 5)
    np.testing.assert_array_equal(image, noisy())  # the input is untouched
    np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
    assert (out[mask > 0] != image[mask > 0]).any()


def test_one_colour_per_block_and_mean_kept():
    image = noisy()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:30, 20:50] = 255
    out = pixelate(image, mask, 5)
    for y in range(10, 30, 5):
        for x in range(20, 50, 5):
            assert len(np.unique(out[y:y + 5, x:x + 5].reshape(-1, 3), axis=0)) == 1
    np.testing.assert_allclose(out[mask > 0].mean(axis=0), image[mask > 0].mean(axis=0), atol=0.5)


def test_colours_outside_the_mask_do_not_bleed_in():
    image = np.zeros((20, 20, 3), np.uint8)
    image[:, 10:] = 200
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 4:10] = 255  # all masked pixels are black, their blocks reach the white half
    out = pixelate(image, mask, 8)
    assert out[mask > 0].max() == 0


def test_gray_image_and_empty_mask():
    gray = noisy((30, 30))
    mask = np.zeros((30, 30), np.uint8)
    np.testing.assert_array_equal(pixelate(gray, mask, 4), gray)
    mask[0:8, 0:8] = 255
    assert len(np.unique(pixelate(gray, mask, 8)[0:8, 0:8])) == 1
