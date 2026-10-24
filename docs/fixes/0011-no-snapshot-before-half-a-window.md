# 0011: No snapshot before the median holds half a window

Date: 2026-10-21. Use case: board (`usecases/board/pipeline.py`).
Commit `99f0cce`. Note written after the fact, on 2026-10-22.

## Problem

The 189 snapshots of the 6, 8, 13 and 15 October lectures were judged by hand (L5): no person, and 21
artefacts. Two of them were the first snapshot of the 8 and 15 October lectures, at 4 s. Each had
15–18 thousand pixels never seen.

## Cause

The first snapshot came from 2 samples. A median of two samples shows whatever they both missed, and
the pixels the lecturer covered in both had never been seen.

## Fix

A snapshot is taken only once the median holds half a window of samples (`window // 2`: 10 samples,
20 s, by default). `board_final.png` is unchanged.

## Verification

- **On the new run.** No snapshot matches a rejected one within 50 pixels. The 18 that match none
  judged before were judged by hand and all approved.
- **Test.** `test_board_pipeline_outputs` (`tests/usecases/test_board_pipeline.py`) checks that the
  first snapshot comes after half a window.

## Files

`src/vision_toolkit/usecases/board/pipeline.py`, `tests/usecases/test_board_pipeline.py`,
`docs/usecases/board.md`.
