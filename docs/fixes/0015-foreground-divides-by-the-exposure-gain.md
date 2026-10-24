# 0015: `foreground_mask` divides by `exposure_gain`

Date: 2026-10-21. Use cases: board and anonymize (`core/foreground.py`).
Commit `99f0cce`. Note written after the fact, on 2026-10-22.

## Problem

While the fixes 0012–0014 were measured, the foreground of single samples jumped: on frames where wall
and board fill about half the frame each, most of the frame came out as foreground. The commit message
is the only record. The samples affected were not counted.

## Cause

`foreground_mask` estimated the frame's gain as the ratio of the median gray levels of frame and
background. With a light wall and a dark board in equal parts, the median sits between the two, and
a person who covers a little of one half moves it into the other. The ratio then jumps far from 1, and
the frame divided by it differs everywhere from the background.

The mask is shared: in `anonymize` it feeds the `person-without-head` doubt.

## Fix

`foreground_mask` uses `exposure_gain`, the median of the per-pixel ratios, on the pixels that are not
black in the background. A person changes a minority of pixels, so it barely moves that median.

## Verification

- **Synthetic check, made while writing this note.** On the scene of the test (wall at 190, board at
  45, a person on the board):
  - the old ratio gives a gain of 3.51, and 100% of the frame is foreground;
  - `exposure_gain` gives 1.00, and the foreground is the person, 13.3% of the frame.
- **Test.** `test_foreground_on_a_half_dark_scene` (`tests/core/test_foreground.py`) checks that only
  the person is foreground.

## Files

`src/vision_toolkit/core/foreground.py`, `tests/core/test_foreground.py`.
