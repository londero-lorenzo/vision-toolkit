# 0006: Each frame is divided by its exposure gain

Date: 2026-10-20. Use case: board (`core/foreground.py`, `core/temporal.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

In the blind calibration of 2026-10-20, the gain of the wall drifted by ±3–7% over each lecture, with
jumps of 2–4% between samples 2 s apart: 58–149 jumps above 2% per video. The drift does not depend on
where the lecturer stands (|r| < 0.26). The board's own gain differed from the wall's by up to ±7%.
With the snapshot threshold lowered to the chalk ([0007](0007-snapshot-threshold-from-the-chalk.md)),
a change of the whole picture would count as writing, and frames at different exposures mix in the
median.

## Cause

The camera's automatic exposure.

## Fix

- **`core.foreground.exposure_gain`.** The median ratio of the gray levels of a frame against a
  reference. It uses only the pixels brighter than 20 levels in the reference, at most 50,000 of them,
  and clips the result to 0.5–2.
- **`PlaneMedian(exposure_reference)`.** Each rectified frame is divided by its gain, measured on the
  visible board (the pixels outside the lecturer's mask), not on the wall: the ratio of the two varies.
- **`--no-match-exposure`** turns it off. `run.json` records the 5th, 50th and 95th percentiles of the
  gain.

## Verification

- **On the real lectures.** With the reference fixed by [0010](0010-fixed-exposure-reference.md), the
  gain stays within 0.91–1.09 (5th to 95th percentile).
- **Tests.** In `test_exposure_jump_changes_no_pixel` (`tests/usecases/test_board_pipeline.py`), the
  camera brightens by 12% halfway with nothing written: with the gain, less than 0.4% of the board
  changes, without it more than 4%. The gain measured spans the 12%.
  `test_exposure_change_leaves_the_background_alone` (`tests/core/test_temporal.py`) checks the same
  on `PlaneMedian` with a 10% step.

## Files

`src/vision_toolkit/core/foreground.py`, `src/vision_toolkit/core/temporal.py`,
`src/vision_toolkit/usecases/board/pipeline.py`, `src/vision_toolkit/usecases/board/cli.py`,
`tests/usecases/test_board_pipeline.py`, `tests/core/test_temporal.py`, `tests/core/test_foreground.py`.
