# Board without lecturer

Remove the lecturer from the board in lecture videos and know, moment by moment, what they are
pointing at. Command: `vision-toolkit board`. Code: `src/vision_toolkit/usecases/board/`.

## Why it exists

A screenshot of the full board at the end of a lecture loses the context. While explaining, the
lecturer points at the parts being discussed, and that link is needed to align the board with the
transcript in the study notes.

The use case produces:
- the board rebuilt **without the lecturer**, with a snapshot every time the content changes
  (writing or erasing);
- a timestamped log of the **spot pointed at** by the lecturer, in board coordinates, to match against
  the timestamps of the transcript.

## How it works

1. **Layout.** The crop and the audience line come from `--crop` and `--redact-below`, or from the
   `setup.json` confirmed for `anonymize` (L1, `--setup`). The rows below the audience line
   are blacked out before any model sees the frame. A pose model looking for one person otherwise
   picks a student in the front row.
   (`core.layout`, `core.redact.black_out_below`)
2. **Board.** A dark quadrilateral found with Otsu on the temporal median of the frames, above the
   audience line. The lecturer moves, so they disappear from the median. Each side is then moved
   outward to the board's real edge: the strongest dark-to-light step, or, on the top side, a light
   strip running along the board, whose glow Otsu takes for wall. The board is rectified with a
   homography. Alternatively the 4 corners are given by hand (`--quad`).
   (`core.io.median_frame`, `usecases/board/detect.py`, `core.geometry.rectify_homography`)
3. **Camera framing.** A frame every 15 s is aligned (ECC, rotation and shift) to a reference on the
   band above the board. A move that lasts splits the video into segments:
   - each segment gets its own median and its own corners, rectified onto the same plane, so the
     board carries on across the move;
   - segments shorter than `--min-segment` are skipped, such as the title card or the seconds while
     the camera is being set;
   - so are the frames while the camera is moving.

   With `--quad`, the same hand-set corners serve every segment. (`core.framing`)
4. **Lecturer.** The mask from MediaPipe PoseLandmarker, at confidence 0.1 instead of 0.5. The arms
   are added as segments between the shoulder, elbow, wrist and index keypoints. When the pose finds
   nobody, the mask is the box of the foreground blob shaped like a standing person: tall, and down
   to the audience line. The foreground blobs touching the mask join it: on the real lectures the
   pose mask misses a sleeve, the side turned away or a shadow for tens of seconds, long enough to
   enter the board. For up to 20 s after the pose loses the lecturer (`--track`), the foreground
   blobs touching the previous frame's mask are taken for the lecturer too. In front of the board
   the standing shape may hold only the legs: the upper body is a blob of its own, cut off at the
   board's lower edge. At the edge of the frame the lecturer is partly out of view and no standing
   shape at all. The
   foreground is the difference from the segment's median. Optional: DeepLab V3 on the crop around the person (`--refine-model deeplab-v3`).
   (`usecases/board/person_mask.py`, `ops.skeleton`, `ops.crop_refine`, `core.foreground`)
5. **Reconstruction.** For every pixel of the rectified board, the median of the last N samples in
   which it was not covered, 21 samples 2 s apart by default. A pixel covered now was visible shortly
   before or will be shortly after. A pixel changes only once it was seen in a quarter of the window
   (`--min-seen`), otherwise it keeps its last value: next to a lecturer who stays, a pixel seen in
   one or two samples would take whatever those showed, a bit of the lecturer the mask missed included.
   Each frame is first divided by its exposure gain, measured on the visible board against the
   segment's median: the camera's exposure drifts by a few percent and jumps between frames. The
   reference stays fixed on purpose. Against the board being rebuilt the gain drifted: on the
   13 October lecture it reached 1.10–1.25, where the board's real gain stays within 0.99–1.05.
   (`core.temporal.PlaneMedian`, `core.foreground.exposure_gain`)
6. **Snapshots.** A pixel has changed when it moved by more than half the chalk contrast measured on
   the video, between 6 and 25 levels. A snapshot is taken when enough pixels have changed
   (`--min-change`) and the change has settled, and not before the median holds half a window: from
   the first two samples, the first snapshots of two lectures had 15–18 thousand pixels never seen.
   (`usecases/board/chalk.py`)
7. **Pointed spot.** The index fingertip of the arm raised highest relative to the torso, mapped to
   board coordinates. (`ops.pointing`)

The pose and refine models are picked by name from the catalog (`vision-toolkit models list`), so a
different model is a CLI option, not a code change.

## Remove, don't blur

The lecturer is **removed** from the board, not obscured. Obscuring them (blur, silhouette, average
colour) is worse on both counts:
- **privacy:** the silhouette stays and, depending on the blur, the person can still be recognised;
- **content:** the body covers exactly the board that is needed. In the test frame (CIT, 11:33) the
  lecturer points at `H_2(P_1/2) = 1/2 log2 2 + 1/2 log2 2 = 1`, and the body covers the column with
  the definition of `H_D(P)`.

The gesture matters, but only as data. The pointed spot is extracted while the person is in the frame
and saved as metadata in `pointing.jsonl`: coordinates only, no pixels of the person. Two separate
products come out, the clean board and the log of the pointed spot. Overlaying them gives the clean
board with the pointed area highlighted.

Rebuilding from nearby frames also solves the case where the hand covers the very formula it points at.

Only the clean snapshots and `pointing.jsonl` go into the study vault. The frames with the lecturer,
even censored ones, stay in `data/`, outside git.

## From the transcript to the formula

The case that drives the project: the transcript has "Here we are calculating the entropy" at a
certain minute, and the question is which formula is being pointed at.

1. Search the sentence in the transcript and read its timestamp.
2. Take the window of frames around that timestamp.
3. Within the window, keep only the frames with a pointing gesture (`arm_angle`, persistence over
   several samples).
4. Overlay the pointed area on the clean snapshot of that moment.
5. The assistant reads the area and says which formula it is and how it relates to what the lecturer is saying.

**Transcript granularity.** The vault transcripts (`*.audio.md`) have one timestamp per block of
speech (0:21, 1:27, 2:28…), not per sentence. A sentence therefore leads to a window of about a
minute, not to the second. There are two ways forward:
- within the window, pick the frames with a pointing gesture, as in step 3 (enough to start);
- or re-transcribe the audio with word-level timestamps (Whisper provides them), and the window
  shrinks to a few seconds.

**Required precision.** An **area** is enough (a box on the board), not a precise point: the assistant
does the fine reading of the formula on the crop. This matters most when the lecturer points from a
distance, arm stretched, instead of touching the board.

Estimated difficulty, from lowest:
- timestamp → frame (ffmpeg) and searching the sentence in the transcript: easy;
- clean board: medium, it is mostly temporal fusion; the window must stay short because the board
  changes (writing, erasing);
- pointing from a distance: the most delicate part.

## What the experiments showed

On a frame of the CIT lecture (`notebooks/01_single_frame.ipynb`):

| Approach | Result |
|---|---|
| DeepLab V3 on the full frame (MediaPipe web demo) | 0 person pixels, max confidence 0.09 |
| DeepLab V3 on a crop around the lecturer | person found, confidence 0.9997 |
| PoseLandmarker on the full frame | person found, index keypoint on the fingertips |
| Colour (skin, slate, chalk in Lab) | slate compact and separable; skin and chalk have the same lightness, 10% of the arm pixels classified as chalk |

- DeepLab's problem is **resolution** (it runs at 257×257, the lecturer becomes a few pixels), not
  colour. Averaging colours per region helped, but it needs the mask to be known already: it is circular.
- Colour tells "not board" apart well, but not skin or a white shirt from chalk: useful only to refine edges.
- The decisive step is **temporal**: on a synthetic video (`tests/usecases/test_board_synthetic.py`)
  the reconstruction has 0% wrong pixels against 3.2% for a single frame with the lecturer in front.

The figures come from a frame in which the lecturer is censored with areas of average colour: they
must be confirmed on a real video.

### Blind calibration on four real lectures

On 2026-10-20, scripts ran on the four lectures of 6, 8, 13 and 15 October and printed only numbers,
one sample every 2 s (privacy protocol in [anonymize](anonymize.md#privacy-protocol)). The images
looked at were frames released through a review queue. Figures are in the order 6, 8, 13 and 15 October.

| Measure | Finding | Change |
|---|---|---|
| Pose on the full frame, one person | locks onto a student in 17%, 1%, 38% and 11% of the poses found (nose below the audience line) | audience blacked out first ([0001](../fixes/0001-audience-blacked-out-before-the-pose.md)) |
| Pose misses the lecturer | 18%, 1%, 21% and 6% of the samples, up to 38 s in a row | confidence 0.1 ([0002](../fixes/0002-pose-confidence-0.1.md)); foreground fallback ([0003](../fixes/0003-standing-shape-when-the-pose-misses.md)) |
| Both changes together, on the missed frames | lecturer found in 65%, 53%, 94% and 88% of them; a person invented in 4 of 188 empty frames | — |
| Exposure | the wall gain drifts by ±3–7%, with 2–4% jumps between samples; the board's gain differs from the wall's by up to ±7% | gain measured on the board ([0006](../fixes/0006-exposure-gain-per-frame.md)), against the segment's median ([0010](../fixes/0010-fixed-exposure-reference.md)) |
| Chalk contrast in single frames | median 17, 14, 42 and 30 levels; the old threshold of 25 saw 27%, 19%, 86% and 59% of the stroke pixels | threshold from the chalk ([0007](../fixes/0007-snapshot-threshold-from-the-chalk.md)) |
| Lecturer standing still | 90th percentile 6–8 s, longest 14–20 s, against a 7 s window | 21 samples 2 s apart ([0008](../fixes/0008-longer-median-window.md)) |
| Otsu corners | 25–50 px short at the top, under the light strip | sides moved to the real edge ([0004](../fixes/0004-board-sides-moved-to-the-real-edge.md)) |
| Camera framing | one jump of 31 px at 18 s in the 6 October lecture, then still; one framing in the others | framing segments ([0009](../fixes/0009-framing-segments.md)) |

The corners found by `find_board` on the medians of the originals agree within 1.3 px with the ones
checked on released frames. Where the pose finds the lecturer, the final mask covers the head disk in
96–98% of the poses.

### First judgement of the snapshots

The 189 snapshots of the four lectures were then judged by hand (L5). None shows a face or a person. 21
show artefacts: 3, 15, 2 and 1 in the four lectures. They do not spoil the board, though nearby text
is sometimes readable only in the next snapshot. The samples around them point to four causes:

| Cause | Where | Change |
|---|---|---|
| The first snapshot comes from 2 samples, with 15–18 thousand pixels never seen | 8 and 15 October, at 4 s | no snapshot before half a window ([0011](../fixes/0011-no-snapshot-before-half-a-window.md)) |
| The pose mask misses part of the lecturer, 1–6 thousand foreground pixels attached to it, for 20–50 s | 8 October at 37–40 min, 13 October | foreground blobs touching the mask join it ([0012](../fixes/0012-blobs-touching-the-pose-mask-join-it.md)) |
| A pixel near a lecturer who stays is seen in one or two samples, a bad one included | 6 October at 40 min | a pixel changes only once seen in a quarter of the window ([0013](../fixes/0013-pixel-changes-once-seen-in-a-quarter.md)) |
| The lecturer at the edge of the frame, partly out of view: neither the pose nor the standing shape finds the lecturer | 8 October at 12 and 24 min | followed through the foreground touching the last mask ([0014](../fixes/0014-lecturer-tracked-at-the-frame-edge.md)) |

On the new run, 144 of the 162 snapshots match, within 50 pixels, an approved one, and none
matches a rejected one. The other 18 were judged and all approved. The frames where nobody is
found drop from 12%, 22%, 5% and 10% to 8%, 7%, 1% and 6%; the lecturer is followed through the
foreground in 4–15% of the frames.

The indicator tried first, zones present in one snapshot only, caught 6 of the 19 rejected snapshots
it could measure and flagged 8 judged clean: a leak that lasts is in several snapshots.
The judgement by hand stays the measure.

### Six more lectures

On 2026-10-21 the pipeline ran with the same setup on six more lectures, of 6, 8, 13, 15 and 20
October. Their 245 snapshots were judged by hand: no person, and one artefact. It is a light patch in the
6 October lecture at 18 min 8 s, and the chalk stays readable. In front of the board the standing
shape held only the lecturer's legs, and the upper body entered the median. Cause and fix are in
[fix 0017](../fixes/0017-upper-body-cut-off-from-the-legs.md).

The corrected pipeline then ran again on all ten lectures. 401 of the 406 snapshots match, within
50 pixels, an approved one, and none matches the rejected one. The other 5 were judged and all
approved. In place of the patch, the snapshots at 18 min 22 s, 19 min 16 s and 20 min match the
approved one at 17 min 58 s.

Three recordings stay out of these runs:
- two lectures of 1 October, filmed by the lecturer's laptop camera, which is moved by hand. They
  are kept for when a moving camera is handled;
- the minute recorded on 8 October after the lecture, where students who stand up cross the
  audience line.

## Usage

```
vision-toolkit models download                 # once: model files into models/
vision-toolkit board data/video/lezione.mp4 --setup output/anonymized/lezione/setup.json
```

Main options:
- `--step`: seconds between two analysed frames (default 2)
- `--setup setup.json`: crop and audience line, as confirmed for `anonymize`
- `--crop x,y,w,h`: video area, if it is a screen recording with the player interface (overrides the setup)
- `--redact-below`: audience line in the cropped frame (overrides the setup)
- `--quad file.json`: board corners set by hand
- `--window`: samples in the temporal median (default 21)
- `--min-seen`: fraction of the window a board pixel must be seen in to change (default 0.25)
- `--min-change`: fraction of changed pixels that triggers a new snapshot (default 0.004)
- `--change-threshold`: levels a pixel must change by to count (default: half the chalk contrast of
  the video, between 6 and 25)
- `--no-match-exposure`: keep each frame's exposure as it is
- `--framing-step`: seconds between two frames of the framing check (default 15; `none` skips it)
- `--min-segment`: framing segments shorter than this are skipped (default 60 s)
- `--pose-model`: pose model from the catalog (default `mediapipe-pose-heavy`)
- `--pose-confidence`: the pose model's `min_confidence` (default 0.1; `none` keeps the model's own)
- `--no-fallback`: no foreground blob when the pose finds nobody
- `--no-grow-mask`: keep the mask as it is, without the foreground blobs touching it
- `--track`: seconds the lecturer is followed through the foreground once the pose lost the lecturer
  (default 20; 0 never)
- `--refine-model`: segmentation model run on the crop around the lecturer, e.g. `deeplab-v3`
  (default none)
- `--refine-labels`: labels of the refine model that belong to the lecturer (default `person`)

From Python: `vision_toolkit.usecases.board.pipeline.run(video, out_dir, BoardOptions(...))`.

## Output

In `output/<video name>/`:
- `board_<hh>h<mm>m<ss>s.png`: clean board snapshots
- `board_final.png`, plus `median.png` and `quad.json` when the corners are detected (`quad.json` holds `{"quad": [[x, y], ...]}`, tl, tr, br, bl).
  With several framing segments they are those of the longest one; `median.png` has the audience
  blacked out
- `pointing.jsonl`: one line per analysed frame in which the pose found the lecturer
- `run.json`: what the run measured and decided. It has the framing segments (start, end, used or
  why skipped, corners), the snapshot threshold and the chalk contrast it came from, how the lecturer
  was found (pose, foreground, tracking, nobody), the exposure gain percentiles and the number of
  snapshots.
  The vault does not read it.

```json
{"t": 693.0, "frame_xy": [679, 415], "board_xy": [612, 251], "board_xy_norm": [0.398, 0.762],
 "arm_angle": 94.0, "occluded_fraction": 0.054, "snapshot": "board_00h11m30s.png"}
```

- `t`: seconds from the start of the video
- `frame_xy`: fingertip in frame pixels; `board_xy`: on the rectified board, `null` when off the board;
  `board_xy_norm`: the same divided by the board width and height
- `arm_angle`: angle between arm and torso, about 0 at rest and about 90 with the arm stretched out
- `occluded_fraction`: fraction of the board covered by the lecturer
- `snapshot`: the latest snapshot at that time

The vault reads these files. The format is pinned by `tests/usecases/test_board_pipeline.py`. Any change
must be agreed first (see [architecture](../architecture.md#usecases)).

## Status

The pipeline is complete. It was calibrated blind on four real lectures, and its snapshots were
judged by hand (L5 in [anonymize](anonymize.md#where-the-person-comes-in)): no person, 21 artefacts in
189, whose causes are corrected (see above). No artefact was found on the corrected run. On six
more lectures one in 245 was found, whose cause is corrected in
[fix 0017](../fixes/0017-upper-body-cut-off-from-the-legs.md). None was found on the corrected run
of all ten lectures.

Known problems:
- on the synthetic video the pointed spot is right in 7 frames out of 10. In one the hand leaves the
  frame. In the other two, PoseLandmarker reads the stretched arm as lowered, even on frames almost
  identical to one it reads correctly;
- on the real lectures the arm is raised above 60° in 51–60% of the frames, because the lecturer
  writes: the arm angle alone does not tell pointing from writing.

Countermeasures to try:
- MediaPipe's VIDEO mode (tracking between frames);
- accepting a pointing gesture only if it persists for 2 or more samples, and away from the chalk
  that appears.

Next steps:
- export for the summaries: snapshots + transcript intervals + pointed areas
- query by timestamp: given a moment (or a sentence of the transcript), return the snapshot and the
  pointed area in the window (see "From the transcript to the formula")
- evaluate re-transcribing with word-level timestamps (Whisper) to narrow the window
