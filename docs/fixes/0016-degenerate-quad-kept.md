# 0016: `refine_quad` keeps a quad whose sides do not meet

Date: 2026-10-22. Use case: board (`usecases/board/detect.py`).
Commit `2bfeec1`. Note written after the fact, on 2026-10-22.

## Problem

On single frames of the two 1 October lectures, filmed by a camera moved by hand, `refine_quad` raised
`LinAlgError` instead of returning a quad.

## Cause

On those frames Otsu returned a quad with a side of no length, or two of the fitted sides
([0004](0004-board-sides-moved-to-the-real-edge.md)) came out parallel. The corners are where the
lines meet: with parallel lines the system has no solution.

## Fix

A quad with a side shorter than a pixel, or whose fitted sides do not meet, is returned as it is.

## Verification

- **Test.** `test_refine_quad_keeps_a_degenerate_quad` (`tests/usecases/test_board_parts.py`): a quad
  with two equal corners comes back unchanged.
- **Still to do.** The 1 October lectures stay out of the board runs until a moving camera is handled.

## Files

`src/vision_toolkit/usecases/board/detect.py`, `tests/usecases/test_board_parts.py`.
