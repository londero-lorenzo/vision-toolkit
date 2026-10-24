# 0008: The temporal median holds 21 samples 2 s apart

Date: 2026-10-20. Use case: board (`core/temporal.py`, `usecases/board/pipeline.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

In the blind calibration of 2026-10-20, the lecturer stood still for 6–8 s at the 90th percentile and
for 14–20 s at the longest. The median held 7 samples 1 s apart, a 7 s window.

## Cause

The median absorbs an occluder only while it stays in one place for less than half the window. A
lecturer standing still for longer than 3–4 s, with a bit the mask missed, enters the board.

## Fix

- **Window.** 21 samples 2 s apart by default (`--window`, `--step`): 42 s.
- **Median by sorting.** `_nan_median` sorts each pixel's samples, NaN last, instead of calling
  `np.nanmedian`. The median costs 0.55 s per sample with 21 samples, against 0.97 s with
  `np.nanmedian` on 7.

## Verification

- **No test of its own.** No synthetic test has a lecturer standing still for longer than the old
  window. The snapshots judged by hand (L5) all ran with this window.
- **Still to verify.** A test with a still lecturer, failing with 7 samples.

## Files

`src/vision_toolkit/core/temporal.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/core/test_temporal.py`.
