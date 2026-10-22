# Fixes

Every correction to the code gets a note in this folder. The note records:
- what went wrong and how it showed;
- the cause;
- what was changed;
- how the change was verified.

Larger choices still get a decision record in `../decisions/`.

## How to write a note

- **Name.** `NNNN-<slug>.md`: the next number, and a slug that says what is fixed.
- **Sections.** Follow [0017](0017-upper-body-cut-off-from-the-legs.md):
  - *Problem*: the symptom and how it was found;
  - *Cause*: from the measurements, with figures;
  - *Fix*: each change, and the remedies tried and dropped;
  - *Verification*: figures before and after, the test that fails without the fix, and what is still
    to verify;
  - *Files*: the files changed.
- **Same commit.** The note goes in the same commit as the fix, together with its row in the table
  below. The commit message names the note. The note's commit is therefore the one that adds it:
  `git log --follow -- docs/fixes/NNNN-*.md`. `--follow` is needed because the notes were renumbered
  once (see below).
- **Use case page.** If the fix changes behaviour that a use case page describes, the page changes
  in the same commit and links to the note.

## Notes

| Note | Date | Commit | Use case | Fix | Test |
|---|---|---|---|---|---|
| [0001](0001-audience-blacked-out-before-the-pose.md) | 2026-10-20 | `e638f6d` | board | the audience is blacked out before any model | `test_audience_is_blacked_out_before_the_pose` |
| [0002](0002-pose-confidence-0.1.md) | 2026-10-20 | `e638f6d` | board | the pose model runs at confidence 0.1 | `test_pose_confidence_is_an_option` (the option is accepted) |
| [0003](0003-standing-shape-when-the-pose-misses.md) | 2026-10-20 | `e638f6d` | board | a standing shape masks the lecturer when the pose misses | `test_foreground_masks_the_lecturer_when_the_pose_misses` |
| [0004](0004-board-sides-moved-to-the-real-edge.md) | 2026-10-20 | `e638f6d` | board | the board's sides are moved to its real edge | `test_the_light_strip_moves_the_top_side_to_the_board_edge` |
| [0005](0005-otsu-closing-on-a-padded-image.md) | 2026-10-20 | `e638f6d` | board | the Otsu closing runs on a padded image | — |
| [0006](0006-exposure-gain-per-frame.md) | 2026-10-20 | `e638f6d` | board | each frame is divided by its exposure gain | `test_exposure_jump_changes_no_pixel` |
| [0007](0007-snapshot-threshold-from-the-chalk.md) | 2026-10-20 | `e638f6d` | board | the snapshot threshold follows the chalk | `test_stroke_contrast_and_threshold` |
| [0008](0008-longer-median-window.md) | 2026-10-20 | `e638f6d` | board | the median holds 21 samples 2 s apart | — |
| [0009](0009-framing-segments.md) | 2026-10-20 | `e638f6d` | board | the video is split into framing segments | `test_camera_moved_by_hand`, `tests/core/test_framing.py` |
| [0010](0010-fixed-exposure-reference.md) | 2026-10-20 | `e638f6d` | board | the exposure reference is the segment's median | `test_exposure_change_leaves_the_background_alone` |
| [0011](0011-no-snapshot-before-half-a-window.md) | 2026-10-21 | `7ecc290` | board | no snapshot before half a window | `test_board_pipeline_outputs` |
| [0012](0012-blobs-touching-the-pose-mask-join-it.md) | 2026-10-21 | `7ecc290` | board | the foreground blobs touching the pose mask join it | `test_pose_mask_grows_over_what_the_pose_missed` |
| [0013](0013-pixel-changes-once-seen-in-a-quarter.md) | 2026-10-21 | `7ecc290` | board | a pixel changes only once seen in a quarter of the window | `test_pixel_seen_in_few_frames_keeps_its_value` |
| [0014](0014-lecturer-tracked-at-the-frame-edge.md) | 2026-10-21 | `7ecc290` | board | the lecturer is followed through the foreground at the frame edge | `test_lecturer_followed_through_the_foreground_once_lost` |
| [0015](0015-foreground-divides-by-the-exposure-gain.md) | 2026-10-21 | `7ecc290` | board, anonymize | `foreground_mask` divides by `exposure_gain` | `test_foreground_on_a_half_dark_scene` |
| [0016](0016-degenerate-quad-kept.md) | 2026-10-22 | `2bfeec1` | board | `refine_quad` keeps a quad whose sides do not meet | `test_refine_quad_keeps_a_degenerate_quad` |
| [0017](0017-upper-body-cut-off-from-the-legs.md) | 2026-10-22 | `17232eb` | board | the upper body is followed when only the legs are a standing shape | `test_upper_body_cut_off_from_the_legs_is_followed` |

## Notes written after the fact

The notes started on 2026-10-22 with the fix now numbered 0017. Notes 0001–0016 were written the same
day for the corrections made before, and the numbers were reassigned in time order. The commit message
of `17232eb` therefore names its note as 0001. The messages are corrected, and the commits before
the notes name theirs, in a rebase before the first push ("Before the first push" in `CLAUDE.md`).
The hashes in this folder change with it.

The earlier notes draw on the commit messages, on `docs/usecases/board.md` and on the figures of the
calibration runs:
- **0005 and 0015** had only a commit message and a code comment. Their verification is a synthetic
  check made while the note was written.
- **0005 and 0008** have no test of their own. A test for each is still to add.
- **0010**: the commit gives only the behaviour after the fix. The drift that called for it is in
  board.md, step 5.

The anonymize use case has had no correction of its own. 0015 changes the foreground it shares with
board. The audience line moved from 770 to 690 before its first run: that is a value in `setup.json`,
not code.
