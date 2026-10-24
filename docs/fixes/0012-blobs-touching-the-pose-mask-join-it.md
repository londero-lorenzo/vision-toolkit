# 0012: The foreground blobs touching the pose mask join it

Date: 2026-10-21. Use case: board (`core/foreground.py`, `usecases/board/person_mask.py`).
Commit `99f0cce`. Note written after the fact, on 2026-10-22.

## Problem

In the judgement of the 189 snapshots (see [0011](0011-no-snapshot-before-half-a-window.md)), the
artefacts of 8 October at 37–40 min and of 13 October at 25 min 20 s and 48 min 12 s showed parts of
the lecturer on the board.

## Cause

The pose mask missed part of the lecturer, such as a sleeve, the side turned away or a shadow. In the
samples around the artefacts, 1–6 thousand foreground pixels hung on to the mask for 20–50 s, long
enough to enter the median.

## Fix

- **`core.foreground.blobs_touching`.** The connected foreground blobs that overlap a mask.
- **`PersonMasker(grow)`.** The foreground blobs touching the pose mask join it, dilated like the
  mask. The standing shape of [0003](0003-standing-shape-when-the-pose-misses.md) grows in the same way.
- **`--no-grow-mask`** turns it off.

## Verification

- **On the new run.** No snapshot matches a rejected one (see 0011).
- **Test.** In `test_pose_mask_grows_over_what_the_pose_missed` (`tests/usecases/test_board_pipeline.py`),
  a sleeve the pose model does not see hangs from a lecturer standing still: without the growth more
  than 1% of the final board is wrong, with it less than 0.2%.

## Files

`src/vision_toolkit/core/foreground.py`, `src/vision_toolkit/usecases/board/person_mask.py`,
`src/vision_toolkit/usecases/board/pipeline.py`, `src/vision_toolkit/usecases/board/cli.py`,
`tests/core/test_foreground.py`, `tests/usecases/test_board_pipeline.py`, `docs/usecases/board.md`.
