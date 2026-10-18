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

1. **Board.** A quadrilateral found on the temporal median of the frames: the lecturer moves, so they
   disappear from the median. It is then rectified with a homography. Alternatively the 4 corners are
   given by hand (`--quad`), once per room.
   (`core.io.median_frame`, `usecases/board/detect.py`, `core.geometry.rectify_homography`)
2. **Lecturer.** The mask from MediaPipe PoseLandmarker, plus the arms drawn as segments between the
   shoulder, elbow, wrist and index keypoints. Optional: DeepLab V3 on the crop around the person
   (`--refine-model deeplab-v3`).
   (`usecases/board/person_mask.py`, `ops.skeleton`, `ops.crop_refine`)
3. **Reconstruction.** For every pixel of the rectified board, the median of the last N samples in
   which it was not covered. A pixel covered now was visible shortly before or will be shortly after.
   (`core.temporal.PlaneMedian`)
4. **Pointed spot.** The index fingertip of the arm raised highest relative to the torso, mapped to
   board coordinates. (`ops.pointing`)

The pose and refine models are picked by name from the catalog (`vision-toolkit models list`), so a
different model is a CLI option, not a code change.

## Remove, don't blur

The lecturer is **removed** from the board, not obscured. Obscuring them (blur, silhouette, average
colour) is worse on both counts:
- **privacy:** the silhouette stays and, depending on the blur, the person can still be recognised;
- **content:** the body covers exactly the board that is needed. In the test frame (CIT, 11:33) the
  lecturer points at `H_2(P_1/2) = 1/2 log2 2 + 1/2 log2 2 = 1`, and her body covers the column with
  the definition of `H_D(P)`.

The gesture matters, but only as data. The pointed spot is extracted while the person is in the frame
and saved as metadata in `pointing.jsonl`: coordinates only, no pixels of the person. Two separate
products come out, the clean board and the log of the pointed spot. Overlaying them gives the clean
board with the pointed area highlighted.

Rebuilding from nearby frames also solves the case where the hand covers the very formula it points at.

Only the clean snapshots and `pointing.jsonl` go into the study vault. The frames with the lecturer,
even censored ones, stay in `data/`, outside git.

## From the transcript to the formula

The case that drives the project: in the transcript you hear "Here we are calculating the entropy" at
a certain minute, and you want to know which formula is being pointed at.

1. Search the sentence in the transcript and read its timestamp.
2. Take the window of frames around that timestamp.
3. Within the window, keep only the frames with a pointing gesture (`arm_angle`, persistence over
   several samples).
4. Overlay the pointed area on the clean snapshot of that moment.
5. Claude reads the area and says which formula it is and how it relates to what the lecturer is saying.

**Transcript granularity.** The vault transcripts (`*.audio.md`) have one timestamp per block of
speech (0:21, 1:27, 2:28…), not per sentence. A sentence therefore leads to a window of about a
minute, not to the second. There are two ways forward:
- within the window, pick the frames with a pointing gesture, as in step 3 (enough to start);
- or re-transcribe the audio with word-level timestamps (Whisper provides them), and the window
  shrinks to a few seconds.

**Required precision.** An **area** is enough (a box on the board), not a precise point: Claude does
the fine reading of the formula on the crop. This matters most when the lecturer points from a
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

## Usage

```
vision-toolkit models download                 # once: model files into models/
vision-toolkit board data/video/lezione.mp4 --step 1
```

Main options:
- `--step`: seconds between two analysed frames (default 1)
- `--crop x,y,w,h`: video area, if it is a screen recording with the player interface
- `--quad file.json`: board corners set by hand
- `--window`: samples in the temporal median (default 7)
- `--min-change`: fraction of changed pixels that triggers a new snapshot (default 0.004)
- `--pose-model`: pose model from the catalog (default `mediapipe-pose-heavy`)
- `--refine-model`: segmentation model run on the crop around the lecturer, e.g. `deeplab-v3`
  (default none)
- `--refine-labels`: labels of the refine model that belong to the lecturer (default `person`)

From Python: `vision_toolkit.usecases.board.pipeline.run(video, out_dir, BoardOptions(...))`.

## Output

In `output/<video name>/`:
- `board_<hh>h<mm>m<ss>s.png`: clean board snapshots
- `board_final.png`, plus `median.png` and `quad.json` when the corners are detected (`quad.json` holds `{"quad": [[x, y], ...]}`, tl, tr, br, bl)
- `pointing.jsonl`: one line per analysed frame with the lecturer in view

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

The pipeline is complete, verified only on one frame and on a synthetic video.

Known problem: on the synthetic video the pointed spot is right in 7 frames out of 10. In one the hand
leaves the frame. In the other two, PoseLandmarker reads the stretched arm as lowered, even on frames
almost identical to one it reads correctly. Countermeasures to try on the real video:
- MediaPipe's VIDEO mode (tracking between frames);
- accepting a pointing gesture only if it persists for 2 or more samples.

Next steps:
- try it on a real video of a whole lecture
- shadows of the lecturer on the board (they are not in the mask)
- tune the snapshot threshold, and tell pointing from gesturing (angle and duration of the pose)
- export for the summaries: snapshots + transcript intervals + pointed areas
- query by timestamp: given a moment (or a sentence of the transcript), return the snapshot and the
  pointed area in the window (see "From the transcript to the formula")
- evaluate re-transcribing with word-level timestamps (Whisper) to narrow the window
