# Fixes

Every correction to the code gets a note in this folder. The note records:
- what went wrong and how it showed;
- the cause;
- what was changed;
- how the change was verified.

Larger choices still get a decision record in `../decisions/`.

## How to write a note

- **Name.** `NNNN-<slug>.md`: the next number, and a slug that says what is fixed.
- **Sections.** Follow [0001](0001-upper-body-cut-off-from-the-legs.md):
  - *Problem*: the symptom and who found it;
  - *Cause*: from the measurements, with figures;
  - *Fix*: each change, and the remedies tried and dropped;
  - *Verification*: figures before and after, the test that fails without the fix, and what is still
    to verify;
  - *Files*: the files changed.
- **Same commit.** The note goes in the same commit as the fix, together with its row in the table
  below. The commit message names the note. The note's commit is therefore the one that adds it:
  `git log -- docs/fixes/NNNN-*.md`.
- **Use case page.** If the fix changes behaviour that a use case page describes, the page changes
  in the same commit and links to the note.

## Notes

| Note | Date | Use case | Fix |
|---|---|---|---|
| [0001](0001-upper-body-cut-off-from-the-legs.md) | 2026-10-22 | board | the upper body is followed when only the legs are a standing shape |

## Corrections made before the notes

The notes started on 2026-10-22. The 16 earlier corrections were recorded in two places: the commit
messages and, for the board, `docs/usecases/board.md`. None has a note. In the page:
- what was wrong and why, with figures, is in the tables of *Blind calibration on four real
  lectures* and *First judgement of the snapshots*;
- the result is in *How it works*, which describes the behaviour after the fix.

The anonymize use case has had no code correction. The audience line moved from 770 to 690 before
its first run. That is a value in `setup.json`, not code.

| # | Date | Commit | Correction | Why | Where it is written | Test |
|---|---|---|---|---|---|---|
| 1 | 2026-10-20 | `e638f6d` | audience blacked out before any model | the pose locked onto a student in 1–38% of the poses | commit; board.md calibration table, step 1 | `test_audience_is_blacked_out_before_the_pose` |
| 2 | 2026-10-20 | `e638f6d` | pose confidence 0.1 | the pose missed the lecturer in 1–21% of the samples | commit; board.md calibration table, step 4 | `tests/models/test_mediapipe_models.py` (the option is accepted) |
| 3 | 2026-10-20 | `e638f6d` | foreground blob shaped like a standing person when the pose finds nobody | same | commit; board.md calibration table, step 4 | `test_foreground_masks_the_lecturer_when_the_pose_misses` |
| 4 | 2026-10-20 | `e638f6d` | board sides moved to the real edge (`refine_quad`) | Otsu cut 25–50 px under the light strip | commit; board.md calibration table, step 2 | `test_the_light_strip_moves_the_top_side_to_the_board_edge` |
| 5 | 2026-10-20 | `e638f6d` | Otsu closing on a padded image | a board near the image edge was glued to it | **commit and a code comment only** | — |
| 6 | 2026-10-20 | `e638f6d` | exposure gain per frame | the gain drifts by ±3–7% and jumps by 2–4% | commit; board.md calibration table, step 5 | `test_exposure_jump_changes_no_pixel` |
| 7 | 2026-10-20 | `e638f6d` | snapshot threshold from the chalk contrast, within 6–25 | the threshold of 25 missed 14–81% of the chalk | commit; board.md calibration table, step 6 | `test_stroke_contrast_and_threshold` |
| 8 | 2026-10-20 | `e638f6d` | 21 samples 2 s apart in the median | the lecturer stands still for up to 20 s, the window was 7 s | commit; board.md calibration table, step 5 | — |
| 9 | 2026-10-20 | `e638f6d` | framing segments, short ones skipped | the camera jumps once; the title card | commit; board.md calibration table, step 3 | `test_camera_moved_by_hand`, `tests/core/test_framing.py` |
| 10 | 2026-10-20 | `e638f6d` | exposure reference fixed on the segment's median | against the board being rebuilt the gain drifted to 1.10–1.25 | board.md step 5; the commit gives only the behaviour after the fix | `test_exposure_change_leaves_the_background_alone` |
| 11 | 2026-10-21 | `7ecc290` | no snapshot before half a window | the first snapshot came from 2 samples | commit; board.md judgement table, step 6 | `test_board_pipeline_outputs` |
| 12 | 2026-10-21 | `7ecc290` | foreground blobs touching the pose mask join it | the pose mask missed a sleeve or a side | commit; board.md judgement table, step 4 | `test_pose_mask_grows_over_what_the_pose_missed` |
| 13 | 2026-10-21 | `7ecc290` | a pixel changes only once seen in a quarter of the window | a pixel seen in 1–2 samples took a bit of the lecturer | commit; board.md judgement table, step 5 | `test_pixel_seen_in_few_frames_keeps_its_value` |
| 14 | 2026-10-21 | `7ecc290` | tracking through the foreground at the frame edge | neither pose nor standing shape found her there | commit; board.md judgement table, step 4 | `test_lecturer_followed_through_the_foreground_once_lost` |
| 15 | 2026-10-21 | `7ecc290` | `foreground_mask` divides by `exposure_gain` | the ratio of the gray medians jumped with wall and board half and half | **commit only** | `test_foreground_on_a_half_dark_scene` |
| 16 | 2026-10-22 | `2bfeec1` | `refine_quad` keeps a quad whose sides do not meet | `LinAlgError` on single frames of the 1 October lectures | **commit and docstring only** | `test_refine_quad_keeps_a_degenerate_quad` |

To sum up:
- **13 of the 16** say what and why in board.md, with figures. Their before-and-after figures, where
  they exist, are in the page as well. For correction 10 the commit gives only the behaviour after
  the fix, without the drift that called for it.
- **3 of the 16 (5, 15 and 16)** are only in commit messages and in code comments or docstrings.
- **2 have no test of their own:** 5 (the padded closing) and 8 (the longer window).
- **None** has cause, fix and verification together in one place, as a note does.
