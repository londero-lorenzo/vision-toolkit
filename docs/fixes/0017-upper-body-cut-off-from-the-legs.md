# 0017: The upper body is followed when only the legs are a standing shape

Date: 2026-10-22. Use case: board (`usecases/board/person_mask.py`, `usecases/board/pipeline.py`).

## Problem

The 245 snapshots of six more lectures were judged by hand (L5), and one was rejected: `board_00h18m08s.png`
of the 6 October lecture at 9:46. Around x 248, y 302 it shows a light patch: no face, and the chalk
stays readable. The patch is about 90×90 pixels (x 203–291, y 262–350 on the rectified board). Its
mean gray is 154, against 105 for the board around it. The snapshots at 17:58 and 18:30 do not have
it.

## Cause

The masker was replayed on the samples from 1000 to 1112 s, following that area:
- **The pose loses the lecturer.** In 5 of the 21 samples of the window (1054, 1066, 1076, 1080 and 1090 s),
  the pose model finds nobody, and the mask falls back on the standing shape (`person_blob`).
- **The standing shape holds only the legs.** It is the box of the foreground blob that reaches the
  audience line, here rows 524–697 of the frame. The upper body, light against the dark board, is a
  blob of its own, cut off from the legs at the board's lower edge. It does not touch that box, so
  the foreground blobs joined to the mask (`blobs_touching`) leave it out.
- **The median takes it.** In those 5 samples, 81–96% of the area is foreground left outside the
  mask, at gray 176–183. In the other samples the mask covers it, so it is seen only in those 5. That
  is enough for `--min-seen` (a quarter of 21 samples).

The tracking added for the edge of the frame (`--track`) did follow the foreground touching the
previous mask, but only when the standing shape found nothing.

## Fix

- **`PersonMasker`.** When the pose finds nobody and the previous frame's mask is given, the
  foreground blobs touching it, within `track_reach` of the frame width, join the standing shape.
  Without a standing shape they are the whole mask, as before (source `tracked`).
- **`pipeline.run`.** The `--track` budget counts the samples since the pose lost the lecturer, whether the
  mask came from the foreground or from tracking. Before, a foreground sample reset it. Now a blob
  that stays, such as an erased area touching the lecturer, cannot stay joined to the mask for more than 20 s.
- **Documentation.** Step 4 of `docs/usecases/board.md` and the help of `--track`.

A second remedy was tried and dropped. It grew the standing shape upward by 8% of the frame height,
to reach the blob above it, and covered none of the area.

## Verification

- **On the artefact's window.** In the five samples the area is now 96–99% covered. The foreground
  left outside the mask on the board drops from 86.5 to 0.1 thousand pixels.
- **On two whole lectures.** The figures below cover the samples where the pose finds nobody, in
  frame pixels on the board summed over the samples. They were computed with the previous masks of
  the old code, and so approximate the new run.

  | Lecture | Samples | Foreground outside the mask | Mask beyond the foreground |
  |---|---|---|---|
  | 6 October, 9:46 | 69 | 279.9k → 2.9k | 602.6k → 705.4k |
  | 8 October, 12:55 | 14 | 14.6k → 6.0k | 431.0k → 440.0k |

  The mask beyond the foreground grows by about 1.5 thousand pixels per sample, against about 600
  thousand on the board. Those pixels wait for the next sample, they are not lost.
- **Test.** In `test_upper_body_cut_off_from_the_legs_is_followed`
  (`tests/usecases/test_board_pipeline.py`), a belt the colour of the board cuts the synthetic
  lecturer in two while the pose misses the lecturer. Without the fix, 8.1% of the final board is wrong; with
  it, less than 1%. All 135 tests pass.
- **On the ten lectures.** The pipeline ran again with the same setup on the ten lectures, and gave
  406 snapshots, `board_final` included, against 407 before. 401 match, within 50 pixels, one that
  was approved, and none matches the rejected one. The other 5 were judged by hand and all approved:
  2 in the 13 October lecture at 8:41 and 3 in the 20 October lecture at 8:39, 63–1232 pixels from
  the closest approved one. The mask's source changes in 7 of the 12,042 samples, between tracking
  and nobody.
- **The artefact.** In the 6 October lecture at 9:46, `board_00h18m08s.png` is gone. The snapshots
  taken in its place, at 18:22, 19:16 and 20:00, match the approved one at 17:58 with no pixel
  differing, and differ from the rejected one in about 4,300 pixels.

## Files

`src/vision_toolkit/usecases/board/person_mask.py`, `src/vision_toolkit/usecases/board/pipeline.py`,
`src/vision_toolkit/usecases/board/cli.py`, `docs/usecases/board.md`,
`tests/usecases/test_board_pipeline.py`.
