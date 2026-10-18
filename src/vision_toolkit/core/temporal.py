import cv2
import numpy as np

from vision_toolkit.core.geometry import to_plane


class PlaneMedian:
    """
    Static background on a rectified plane, rebuilt by accumulating frames over time.

    Every frame is warped onto the plane with the homography H; the pixels covered
    by an occluder (for example a person mask) become NaN. The background is the
    per-pixel median of the last `window` valid samples: a pixel covered for the
    whole window keeps its last known value. The median also absorbs bits of the
    occluder that escaped the mask, as long as they stay in the same place for less
    than half the window.
    """

    def __init__(self, H, size, window=7, change_threshold=25):
        self.H = H
        self.size = size  # (width, height)
        self.window = window
        self.change_threshold = change_threshold
        w, h = size
        self.buffer = np.full((window, h, w, 3), np.nan, np.float32)
        self.index = 0
        self.background = np.zeros((h, w, 3), np.float32)
        self.seen = np.zeros((h, w), bool)
        self.last_snapshot = None

    def update(self, frame, occluder_mask):
        """
        Add a frame. Returns the mask of the occluded plane pixels
        (in rectified coordinates).
        """
        warped = cv2.warpPerspective(frame, self.H, self.size).astype(np.float32)
        occluded = cv2.warpPerspective(occluder_mask, self.H, self.size, flags=cv2.INTER_NEAREST) > 0

        warped[occluded] = np.nan
        self.buffer[self.index % self.window] = warped
        self.index += 1

        valid = ~np.isnan(self.buffer[..., 0])
        any_valid = valid.any(axis=0)
        with np.errstate(all="ignore"):
            median = np.nanmedian(self.buffer[:, any_valid], axis=0)
        self.background[any_valid] = median
        self.seen |= any_valid
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
