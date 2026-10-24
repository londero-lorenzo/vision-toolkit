# 0007: The snapshot threshold follows the chalk of the video

Date: 2026-10-20. Use case: board (`usecases/board/chalk.py`, `usecases/board/pipeline.py`).
Commit `910b474`. Note written after the fact, on 2026-10-22.

## Problem

A pixel counted as changed when it moved by more than 25 levels. In the blind calibration of
2026-10-20, the chalk contrast measured on released single frames had a median of 17, 14, 42 and 30
levels in the 6, 8, 13 and 15 October lectures. The threshold of 25 saw 27%, 19%, 86% and 59% of the
stroke pixels: faint writing triggered no snapshot.

## Cause

A fixed threshold. On a washed-out board the strokes stand 10–20 levels above the slate, on a
well-lit one 30–40.

## Fix

- **`stroke_contrast`.** A white top-hat (15 px) and Otsu pick the strokes. Their contrast is the
  largest channel difference from the slate around them (a 21 px median). The answer is none when the
  strokes cover less than 0.1% of the board or stand less than 6 levels above it: on an empty board
  Otsu still splits the slate's texture, whose "strokes" measure 1–4 levels.
- **`change_threshold`.** Half the median contrast of 12 frames spread over the main framing segment,
  with the lecturer left out, within 6–25 levels. 25 stays the answer when no frame has chalk.
- **`--change-threshold`** sets it by hand. `run.json` records the threshold and the contrast it came from.

## Verification

- **On the second run.** The thresholds came out at 9.8, 9.0, 21.2 and 19.0.
- **Test.** `test_stroke_contrast_and_threshold` (`tests/usecases/test_board_parts.py`): faint chalk
  at 16 levels measures 16 ± 2, a plain slate and a textured slate measure none, and the threshold
  follows the clipping.

## Files

`src/vision_toolkit/usecases/board/chalk.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `tests/usecases/test_board_parts.py`.
