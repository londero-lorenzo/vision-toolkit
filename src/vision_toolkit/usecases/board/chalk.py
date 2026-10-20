"""
How strong the chalk is on a board, to set the change that counts as writing.

On a washed-out board the strokes stand only 10-20 levels above the slate, on a
well-lit one 30-40: a fixed threshold misses the faint writing or sees noise on
the strong one. The snapshot threshold therefore follows the chalk of the video.
"""
import cv2
import numpy as np


def stroke_contrast(board, hidden=None, margin=0.013, min_strokes=0.001, min_contrast=6.0):
    """
    Median contrast of the chalk strokes on a rectified board image (uint8 BGR).

    Strokes are thin and light: a white top-hat (15 px) picks them and Otsu splits
    them from the slate. Their contrast is the largest channel difference from the
    slate around them (a 21 px median). `hidden` (bool, board size) marks pixels to
    leave out, such as the lecturer; `margin`, as a fraction of the width, trims the
    board's frame.

    None when the board has no chalk: strokes cover less than `min_strokes` of it, or
    stand less than `min_contrast` above the slate. On an empty board Otsu still splits
    the slate's own texture, whose "strokes" measure 1-4 levels on the lectures seen.
    """
    h, w = board.shape[:2]
    m = max(1, int(margin * w))
    board = board[m:h - m, m:w - m]
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    _, otsu = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = (otsu > 0) & (tophat >= 6)
    if hidden is not None:
        strokes &= ~hidden[m:h - m, m:w - m]
    if strokes.mean() < min_strokes:
        return None
    slate = cv2.medianBlur(board, 21).astype(np.float32)
    contrast = float(np.median((board.astype(np.float32) - slate).max(axis=2)[strokes]))
    return contrast if contrast >= min_contrast else None


def change_threshold(contrasts, fraction=0.5, floor=6.0, ceiling=25.0):
    """
    Snapshot threshold from the stroke contrasts measured on frames of a video:
    `fraction` of their median, within [floor, ceiling]. The floor stays above the
    noise left in a temporal median; the ceiling, the fixed threshold used before,
    is also the answer when no frame has strokes.
    """
    values = [c for c in contrasts if c is not None]
    if not values:
        return ceiling
    return float(np.clip(fraction * np.median(values), floor, ceiling))
