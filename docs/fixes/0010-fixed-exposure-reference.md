# 0010: The exposure reference is the segment's median

Date: 2026-10-20. Use case: board (`core/temporal.py`, `usecases/board/pipeline.py`).
Commit `e638f6d`. Note written after the fact, on 2026-10-22.

## Problem

In the first run on the real lectures, before the commit, the exposure gain of
[0006](0006-exposure-gain-per-frame.md) was measured against the board being rebuilt. It drifted: on
the 13 October lecture it reached 1.10–1.25, where the board's real gain stays within 0.99–1.05. The
commit message gives only the behaviour after the fix.

## Cause

Every small error of the gain enters the rebuilt board, and the next frames are matched to it: the
error feeds itself.

## Fix

The reference is the median of the framing segment, rectified, and stays fixed while the segment
lasts. It changes only with the segment ([0009](0009-framing-segments.md)). The `PlaneMedian`
docstring says why it must not be the board being rebuilt.

## Verification

- **On the real lectures.** The gain stays within 0.91–1.09 (5th to 95th percentile).
- **Test.** `test_exposure_change_leaves_the_background_alone` (`tests/core/test_temporal.py`) checks
  that a fixed reference absorbs a 10% step. No test reproduces the drift against a moving reference.

## Files

`src/vision_toolkit/core/temporal.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`tests/core/test_temporal.py`.
