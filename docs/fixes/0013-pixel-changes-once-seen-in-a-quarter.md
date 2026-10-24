# 0013: A pixel changes only once seen in a quarter of the window

Date: 2026-10-21. Use case: board (`core/temporal.py`, `usecases/board/pipeline.py`).
Commit `99f0cce`. Note written after the fact, on 2026-10-22.

## Problem

In the judgement of the 189 snapshots (see [0011](0011-no-snapshot-before-half-a-window.md)), the three
artefacts of 6 October at 40 min showed a bit of the lecturer next to where the lecturer stood.

## Cause

Next to a lecturer who stays, a pixel is visible in only one or two samples of the window. The median
of one or two samples is whatever they show, and one of them showed a bit of the lecturer the mask
missed.

## Fix

- **`PlaneMedian(min_samples)`.** A pixel changes only when it was seen in at least `min_samples`
  samples of the window, or when it had never been seen. Otherwise it keeps its last value.
- **`--min-seen`.** The fraction of the window, 0.25 by default: 5 of 21 samples.

## Verification

- **On the new run.** No snapshot matches a rejected one (see 0011).
- **Test.** In `test_pixel_seen_in_few_frames_keeps_its_value` (`tests/core/test_temporal.py`), the
  occluder stays in one place and its mask misses one frame: with one sample the background takes that
  frame, with three it keeps its value.
- **Found later.** In front of the board, 5 samples with the upper body outside the mask were enough
  to pass this rule ([0017](0017-upper-body-cut-off-from-the-legs.md)).

## Files

`src/vision_toolkit/core/temporal.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/core/test_temporal.py`, `docs/usecases/board.md`.
