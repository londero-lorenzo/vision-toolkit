import cv2
import numpy as np


def iter_frames(path, step=1.0, start=0.0, end=None, crop=None):
    """
    Walk through the video, one frame every `step` seconds.

    crop: optional (x, y, w, h), to cut away the player interface when the
    video is a screen recording.

    Yields:
        (t_seconds, frame_bgr)
    """
    cap = _open(path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    end = duration if end is None else min(end, duration)

    t = start
    try:
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            yield t, apply_crop(frame, crop)
            t += step
    finally:
        cap.release()


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


class FrameReader:
    """
    Random access to the frames of a video by time, for code that jumps back and
    forth (iter_frames walks forward only). A context manager.
    """

    def __init__(self, path, crop=None):
        self._cap = _open(path)
        self.crop = crop
        self.duration = self._cap.get(cv2.CAP_PROP_FRAME_COUNT) / (self._cap.get(cv2.CAP_PROP_FPS) or 25)

    def at(self, t):
        """The frame at `t` seconds, cropped; None outside the video or if it cannot be read."""
        if not 0 <= t < self.duration:
            return None
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self._cap.read()
        return apply_crop(frame, self.crop) if ok else None

    def close(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def content_box(frame, threshold=16):
    """
    (x, y, w, h) of the area that is not black, ready for `crop`: the video inside a
    screen recording, whose player draws black bars around it.

    A row or column is content when more than half of its pixels are brighter than
    `threshold`. The box spans the longest run of content rows, then the longest run
    of content columns within them, so thin controls and labels drawn on the bars
    (a progress bar, a name) stay out. None if the frame is all black.
    """
    bright = (frame if frame.ndim == 2 else frame.max(axis=2)) > threshold
    rows = _longest_run(bright.mean(axis=1) > 0.5)
    if rows is None:
        return None
    y0, y1 = rows
    x0, x1 = _longest_run(bright[y0:y1].mean(axis=0) > 0.5)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _longest_run(flags):
    """(start, stop) of the longest run of True values, None if there is none."""
    best, start = None, None
    for i, flag in enumerate(np.append(flags, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def median_frame(path, samples=30, crop=None, start=0.0, end=None):
    """
    Temporal median of `samples` frames spread between `start` and `end` seconds
    (the whole video by default).

    Whatever moves (a person walking in front of the scene) disappears from
    the median, which leaves the static background.
    """
    cap = _open(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    first = min(n - 1, max(0, int(round(start * fps))))
    last = n - 1 if end is None else min(n - 1, max(first, int(round(end * fps)) - 1))
    frames = []
    for i in np.linspace(first, last, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(apply_crop(frame, crop))
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def format_time(t):
    """Seconds as "HHhMMmSSs", the timestamp used in output file names."""
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap
