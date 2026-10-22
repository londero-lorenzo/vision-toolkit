# 0005: The Otsu closing runs on a padded image

Date: 2026-10-20. Use case: board (`usecases/board/detect.py`).
Commit `e638f6d`. Note written after the fact, on 2026-10-22.

## Problem

During the calibration of 2026-10-20, the closing that fills the dark mask of `detect_board` glued a
board near the image edge to the edge: the quad took the image edge as one of its sides. The commit
message and a code comment are the only record. Which lecture showed it, and with which figures, was
not written down.

## Cause

The closing is a dilation followed by an erosion, with a 61 px kernel. OpenCV's default border never
erodes. When the dilation reaches the image edge, the erosion does not pull it back, and the dark area
stays joined to the edge. This happens to any board closer than half the kernel to the edge.

## Fix

The dark mask is padded by the kernel size with copies of its edge (`BORDER_REPLICATE`), closed, then
cropped back. The gap between board and edge is then background on both sides of the border, and the
erosion removes it.

## Verification

- **Synthetic check, made while writing this note.** On a 600×400 mask, a board 20 px from the left
  edge (x 20–499):
  - the plain closing gives x 0–499, 4,000 pixels more than the board;
  - the padded closing gives x 20–499, as the board.
- **No test of its own.** A test with a board near the image edge is still to add.

## Files

`src/vision_toolkit/usecases/board/detect.py`.
