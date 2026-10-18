import numpy as np


def bounding_box(mask, margin=0.0):
    """
    Box (x0, y0, x1, y1) around the non-zero pixels, enlarged by `margin` times its
    size on each side and clipped to the mask. x1, y1 are exclusive. None if empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    h, w = mask.shape[:2]
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + 1 + mx), min(h, y1 + 1 + my)


def segment_on_crop(segmenter, image, region_mask, labels=("person",), margin=0.25):
    """
    Run any Segmenter on the crop around `region_mask` and paste its `labels` back.

    Models with a small input (DeepLab V3 runs at 257x257) lose subjects that are
    small in the full frame: on a crop the subject fills the input and is found.
    Returns a uint8 (0/255) mask the size of the image; empty if the region is.
    """
    out = np.zeros(image.shape[:2], np.uint8)
    box = bounding_box(region_mask, margin)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = segmenter(image[y0:y1, x0:x1]).mask_of(*labels)
    return out
