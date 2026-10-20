import numpy as np


def pixelate(image, mask, block):
    """
    Replace the pixels under `mask` with the mean colour of their block.

    The grid of `block` x `block` cells starts at the top-left corner of the mask.
    Only masked pixels change, and only masked pixels enter each mean: the colours
    around the mask do not bleed in, and the mean colour of the region is kept.
    Returns a copy; the input is untouched.
    """
    out = image.copy()
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return out
    block = max(1, int(block))
    rows, cols = (ys - ys.min()) // block, (xs - xs.min()) // block
    cell = rows * (cols.max() + 1) + cols
    counts = np.maximum(np.bincount(cell), 1)
    values = image[ys, xs].astype(np.float64).reshape(len(xs), -1)
    means = np.stack([np.bincount(cell, values[:, c]) / counts for c in range(values.shape[1])], axis=1)
    out[ys, xs] = np.round(means[cell]).astype(image.dtype).reshape(image[ys, xs].shape)
    return out


def black_out_below(image, row):
    """A copy of the image with the rows from `row` down set to black; the image itself if row is None."""
    if row is None:
        return image
    out = image.copy()
    out[max(0, row):] = 0
    return out
