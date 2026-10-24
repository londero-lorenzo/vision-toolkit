# 0002: The pose model runs at confidence 0.1

Date: 2026-10-20. Use case: board (`models/mediapipe/pose_landmarker.py`, `usecases/board/pipeline.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

In the blind calibration of 2026-10-20 (see [0001](0001-audience-blacked-out-before-the-pose.md)),
the pose model found nobody while the lecturer was in the frame in 18%, 1%, 21% and 6% of the samples
of the 6, 8, 13 and 15 October lectures. The longest gap lasted 38 s, on 6 October. With no mask, the
lecturer entered the median of the board.

## Cause

MediaPipe's three thresholds (person detection, pose presence, tracking) default to 0.5. A lecturer
seen from the side, with the back turned or in poor light, scores below them.

## Fix

- **`PoseLandmarker(min_confidence)`.** One option sets the three thresholds. The default stays
  MediaPipe's 0.5.
- **Board.** The pipeline passes 0.1 (`--pose-confidence`; `none` keeps the model's own).

## Verification

- **On the frames where the pose missed the lecturer.** With the audience blacked out and confidence
  0.1, the lecturer is found in 65%, 53%, 94% and 88% of them. In 4 of 188 empty frames a person is
  invented. The frames still missed are covered by the standing shape
  ([0003](0003-standing-shape-when-the-pose-misses.md)).
- **Test.** `test_pose_confidence_is_an_option` (`tests/models/test_mediapipe_models.py`) checks only
  that the model accepts the option. No synthetic test shows the gain, which depends on the real model
  on real frames.

## Files

`src/vision_toolkit/models/mediapipe/pose_landmarker.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/models/test_mediapipe_models.py`, `tests/fakes.py`.
