# 0003: A foreground blob shaped like a standing person masks the lecturer when the pose misses

Date: 2026-10-20. Use case: board (`usecases/board/person_mask.py`, `core/foreground.py`).
Commit `e638f6d`. Note written after the fact, on 2026-10-22.

## Problem

The same as in [0002](0002-pose-confidence-0.1.md). The pose model found nobody in 18%, 1%, 21% and 6%
of the samples while the lecturer was in the frame. Even at confidence 0.1, part of those samples
stays without a pose.

## Cause

Without a pose the mask was empty. Nothing else told the lecturer apart from the board.

## Fix

- **`core.foreground.foreground_mask`.** The difference from the median of the framing segment,
  after the exposure gain, with thin changes such as chalk strokes opened away. It moves to core,
  shared with the doubts of `anonymize`.
- **`person_blob`.** When the pose finds nobody, the mask is the box of the largest foreground blob
  shaped like a standing person: at least 15% of the frame tall, and reaching within 3% of the frame
  height from the audience line. Changes on the board itself, such as a large erased area, never
  reach that low.
- **No pointing.** The mask has no pose, so no line goes to `pointing.jsonl`. The mask's source
  (`foreground`) is counted in `run.json`.
- **`--no-fallback`** turns it off.

## Verification

- **On the same samples.** With 0001 and 0002, the samples with the lecturer present and no mask went
  from 18%, 1%, 21% and 6% to 0.
- **On the second run.** The lecturer was found by the pose in 79%, 77%, 92% and 89% of the frames, and
  by the standing shape in 8%, 1%, 3% and 1%.
- **Test.** In `test_foreground_masks_the_lecturer_when_the_pose_misses`
  (`tests/usecases/test_board_pipeline.py`), a pose model that finds nobody: without the fallback more
  than 2% of the final board is wrong, with it less than 1%, and the 5 frames come from the foreground.
- **Limits found later.** At the edge of the frame the lecturer is no standing shape
  ([0014](0014-lecturer-tracked-at-the-frame-edge.md)), and in front of the board the shape may hold
  only the legs ([0017](0017-upper-body-cut-off-from-the-legs.md)).

## Files

`src/vision_toolkit/usecases/board/person_mask.py`, `src/vision_toolkit/core/foreground.py`,
`src/vision_toolkit/usecases/board/pipeline.py`, `src/vision_toolkit/usecases/board/cli.py`,
`src/vision_toolkit/usecases/anonymize/doubts.py`, `tests/usecases/test_board_pipeline.py`,
`tests/core/test_foreground.py`.
