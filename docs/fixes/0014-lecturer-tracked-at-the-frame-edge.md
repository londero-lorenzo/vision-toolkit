# 0014: The lecturer is followed through the foreground at the edge of the frame

Date: 2026-10-21. Use case: board (`usecases/board/person_mask.py`, `usecases/board/pipeline.py`).
Commit `7ecc290`. Note written after the fact, on 2026-10-22.

## Problem

In the judgement of the 189 snapshots (see [0011](0011-no-snapshot-before-half-a-window.md)), the
artefacts of 8 October at 12 and 24 min showed the lecturer at the edge of the frame.

## Cause

At the edge of the frame the lecturer is partly out of view. The pose model finds nobody, and what
shows is no standing shape ([0003](0003-standing-shape-when-the-pose-misses.md)), so the mask was empty.

## Fix

- **`PersonMasker(previous)`.** When neither the pose nor the standing shape finds anybody, the
  foreground blobs touching the previous frame's mask, dilated by 3% of the frame width
  (`track_reach`), are the mask. Its source is `tracked`.
- **`--track`.** For up to 20 s. The budget restarted at every frame found by the pose or the standing
  shape. [0017](0017-upper-body-cut-off-from-the-legs.md) later changed both rules.
- **`run.json`** counts the frames found by tracking.

## Verification

- **On the new run.** No snapshot matches a rejected one (see 0011). The frames where nobody is found
  drop from 12%, 22%, 5% and 10% to 8%, 7%, 1% and 6%, and the lecturer is followed in 4–15% of the
  frames.
- **Test.** In `test_lecturer_followed_through_the_foreground_once_lost`
  (`tests/usecases/test_board_pipeline.py`), a lecturer with the legs behind the desk walks into the
  pose's blind spot for 5 s: with `track=0` more than 2% of the final board is wrong, by default less
  than 1%, and the 5 frames come from tracking.

## Files

`src/vision_toolkit/usecases/board/person_mask.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/usecases/test_board_pipeline.py`,
`docs/usecases/board.md`.
