# 0004: The board's sides are moved to its real edge

Date: 2026-10-20. Use case: board (`usecases/board/detect.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

In the blind calibration of 2026-10-20, the corners found by Otsu on the median of each lecture were
25–50 px short at the top. On 15 October the bottom-right corner was cut as well. The check was made
on the medians of released frames. The rectified board lost a strip of its top.

## Cause

Otsu leaves out whatever of the board looks light. A light strip runs along the top of the board, and
Otsu takes its glow for wall.

## Fix

- **`refine_quad`.** Each side is moved along its outward normal, at 48 points. The edge is the
  strongest dark-to-light step within 3.5% of the image width outward and a quarter of that inward.
- **Top side.** Where the profile peaks more than 15 levels above the wall, the edge is the peak: the
  light strip.
- **Corners.** A line is fitted through each side's points (Huber, so stray points weigh little), and
  the corners are where the lines meet.
- **`find_board`** = `detect_board` above the audience line, then `refine_quad`.

## Verification

- **On the originals.** The corners found by `find_board` on the medians agree within 1.3 px with the
  ones checked on released frames.
- **Test.** In `test_the_light_strip_moves_the_top_side_to_the_board_edge`
  (`tests/usecases/test_board_parts.py`), a synthetic room with a light strip: Otsu is more than 8 px
  short at the top, `find_board` is within 3 px.
- **Found later.** On single frames of the 1 October lectures `refine_quad` raised an error
  ([0016](0016-degenerate-quad-kept.md)).

## Files

`src/vision_toolkit/usecases/board/detect.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`tests/usecases/test_board_parts.py`, `tests/fakes.py`.
