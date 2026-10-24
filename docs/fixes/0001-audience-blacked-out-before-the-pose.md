# 0001: The audience is blacked out before any model sees the frame

Date: 2026-10-20. Use case: board (`core/redact.py`, `core/layout.py`, `usecases/board/pipeline.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

The blind calibration of 2026-10-20 ran scripts on the four lectures of 6, 8, 13 and 15 October and
printed only numbers, one sample every 2 s. The pose model, looking for one person on the full frame,
locked onto a student: the nose of the pose found was below the audience line in 17%, 1%, 38% and
11% of the poses. On those samples the mask covered a student. The lecturer was left unmasked, and
while standing still entered the median of the board.

## Cause

PoseLandmarker runs with `num_poses = 1`. The students of the front row are nearer the camera and
larger than the lecturer, so the model often picks one of them.

## Fix

- **`core.redact.black_out_below`.** The rows from the audience line down are set to black before
  the pose model, the foreground and the board detection see the frame.
- **Audience line.** It comes from `--redact-below`, or from the `setup.json` confirmed for
  `anonymize` (`--setup`, `core.layout`). Crop and line are thus confirmed once (L1) and shared by
  the two use cases.
- **`find_board`** searches only the rows above the line: a black band would look like a board.

## Verification

- **On the same samples.** The share of samples with the pose on a student went from 12%, 1%, 28% and
  10% to 0.
- **Tests.** In `test_audience_is_blacked_out_before_the_pose` (`tests/usecases/test_board_pipeline.py`),
  a student larger than the visible part of the lecturer sits in front. Without the line more than 2%
  of the final board is wrong, with it less than 1%. `test_find_board_ignores_what_lies_below_the_floor`
  (`tests/usecases/test_board_parts.py`) checks that the black band is not taken for the board.
- **Still to verify.** The line 690 holds for the framing of these lectures. A new framing needs a
  new confirmation (L1).

## Files

`src/vision_toolkit/core/redact.py`, `src/vision_toolkit/core/layout.py`,
`src/vision_toolkit/usecases/board/pipeline.py`, `src/vision_toolkit/usecases/board/cli.py`,
`src/vision_toolkit/usecases/board/detect.py`, `tests/usecases/test_board_pipeline.py`,
`tests/usecases/test_board_parts.py`, `docs/usecases/board.md`.
