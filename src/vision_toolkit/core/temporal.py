import cv2
import numpy as np

from vision_toolkit.core.foreground import exposure_gain
from vision_toolkit.core.geometry import to_plane


class PlaneMedian:
    """
    Static background on a rectified plane, rebuilt by accumulating frames over time.

    Every frame is warped onto the plane with the homography H; the pixels covered
    by an occluder (for example a person mask) become NaN. The background is the
    per-pixel median of the last `window` valid samples. The median absorbs bits of
    the occluder that escaped the mask, as long as they stay in the same place for
    less than half the valid samples.

    A pixel changes only when it was seen in at least `min_samples` of the window,
    or when it had never been seen: otherwise it keeps its last known value. Next to
    an occluder that stays, a pixel seen in one or two frames would take whatever
    those frames showed, a bit of the occluder the mask missed included.

    With an exposure_reference (the plane rectified at the wanted exposure, such as a
    median of the frames), each frame is first divided by its exposure gain against it,
    measured on the visible plane: a camera that brightens or darkens the whole picture
    changes no pixel of the background. The reference must stay fixed. Measured against
    the background being rebuilt, the gain drifts: every small error enters the
    background, and the next frames are matched to it.

    H and exposure_reference may be replaced between two updates, when the camera moved
    and the same plane is seen through a new homography.
    """

    def __init__(self, H, size, window=7, change_threshold=25, exposure_reference=None, min_samples=1):
        self.H = H
        self.size = size  # (width, height)
        self.window = window
        self.change_threshold = change_threshold
        self.exposure_reference = exposure_reference
        self.min_samples = min_samples
        w, h = size
        self.buffer = np.full((window, h, w, 3), np.nan, np.float32)
        self.index = 0
        self.background = np.zeros((h, w, 3), np.float32)
        self.seen = np.zeros((h, w), bool)
        self.last_snapshot = None
        self.gain = 1.0  # exposure gain of the last frame

    def update(self, frame, occluder_mask):
        """
        Add a frame. Returns the mask of the occluded plane pixels
        (in rectified coordinates).
        """
        warped = cv2.warpPerspective(frame, self.H, self.size).astype(np.float32)
        occluded = cv2.warpPerspective(occluder_mask, self.H, self.size, flags=cv2.INTER_NEAREST) > 0

        self.gain = 1.0
        if self.exposure_reference is not None:
            self.gain = exposure_gain(warped, self.exposure_reference, ~occluded)
            warped /= self.gain
        warped[occluded] = np.nan
        self.buffer[self.index % self.window] = warped
        self.index += 1

        median, count = _nan_median(self.buffer)
        valid = count > 0
        changed = valid & ((count >= self.min_samples) | ~self.seen)
        self.background[changed] = median[changed]
        self.seen |= valid
        return occluded

    def image(self, unseen_color=(255, 0, 255)):
        """Current background as uint8; pixels never seen get unseen_color."""
        out = np.clip(self.background, 0, 255).astype(np.uint8)
        out[~self.seen] = unseen_color
        return out

    def change_since_snapshot(self):
        """
        Fraction of pixels that changed (for example writing or erasing)
        since the last snapshot taken.
        """
        if self.last_snapshot is None:
            return 1.0
        diff = np.abs(self.background - self.last_snapshot).max(axis=2)
        return float((diff[self.seen] > self.change_threshold).mean())

    def take_snapshot(self):
        self.last_snapshot = self.background.copy()
        return self.image()

    def to_plane(self, point):
        """Map a frame point to rectified plane coordinates (None if outside)."""
        return to_plane(point, self.H, self.size)


def _nan_median(buffer, rows=32):
    """
    Per-pixel median along the first axis, ignoring NaN, and the count of valid samples.

    Sorting puts NaN last, so the valid samples of a pixel come first: a few times
    faster than np.nanmedian. Pixels with no valid sample get NaN. Runs in bands of
    rows to keep the sorted copy small.
    """
    count = np.count_nonzero(~np.isnan(buffer[..., 0]), axis=0)
    median = np.empty(buffer.shape[1:], np.float32)
    for r in range(0, buffer.shape[1], rows):
        ordered = np.sort(buffer[:, r:r + rows], axis=0)
        n = count[r:r + rows][None, ..., None]
        shape = (1,) + ordered.shape[1:]
        low = np.take_along_axis(ordered, np.broadcast_to(((n - 1) // 2).clip(0), shape), 0)[0]
        high = np.take_along_axis(ordered, np.broadcast_to(n // 2, shape), 0)[0]
        median[r:r + rows] = (low + high) / 2
    return median, count
