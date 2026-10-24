# 0009: The video is split into framing segments

Date: 2026-10-20. Use case: board (`core/framing.py`, `usecases/board/pipeline.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

In the blind calibration of 2026-10-20, the 6 October lecture had one camera jump of 31 px at about
18 s, then stayed still. The other three lectures kept one framing. Every recording opens on a Teams
title card, from 0 to about 3 s. One median and one quad for the whole video rectify the board wrongly
before a jump, and the title card enters the median.

## Cause

The pipeline assumed one camera framing per video.

## Fix

- **`core.framing.framing_segments`.** Every 15 s (`--framing-step`) a frame is aligned (ECC,
  rotation and shift) to the segment's reference. The alignment uses the band above the board's top
  side (`wall_band`), which the writing never reaches and the lecturer seldom does.
- **A move.** More than 2 px or 0.2° away from the reference, for 2 samples in a row. Each boundary is
  narrowed to 1 s by bisection. The frames while the camera moves belong to no segment.
- **Segments.** Each one gets its own median and its own corners, rectified onto the plane of the
  longest one, so the board carries on across a move. Segments shorter than `--min-segment` (60 s) are
  skipped. `run.json` lists them, with why a segment was skipped.

## Verification

- **On the 6 October lecture.** The title card [0, 2.8 s) and the move up to 18.75 s are skipped.
- **Tests.** In `test_camera_moved_by_hand` (`tests/usecases/test_board_pipeline.py`), the camera moves
  by 6 px at 10 s: two segments, a quad within 3 px in each, and one board rebuilt from both, less
  than 1% wrong. `tests/core/test_framing.py` covers the measured motion, a single odd frame that is no
  move, a still camera and a flat wall.
- **Found later.** The two lectures of 1 October are filmed by a camera moved by hand. They are kept
  for when a moving camera is handled.

## Files

`src/vision_toolkit/core/framing.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/core/test_framing.py`,
`tests/usecases/test_board_pipeline.py`, `tests/fakes.py`.
