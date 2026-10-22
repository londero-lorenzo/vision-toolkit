# Anonymize

Hide the faces in frames of a video and let a person decide which frames may be shown to anyone
else. Command: `vision-toolkit anonymize`. Code: `src/vision_toolkit/usecases/anonymize/`.

## Why it exists

Calibrating the board use case means looking at real lecture frames: angles, light, saturation,
masks. The frames show the lecturer and the students, who have not agreed to be shown. Hiding faces
by hand, frame by frame, does not scale to whole lectures.

The models never see the anonymized copies: they run locally on the originals. The copies are only
for whoever may not see the originals: the assistant during calibration. Hiding the face would also hurt the
pose model, whose detector uses the face as its main signal. Measurements are therefore always taken
on the originals.

## Privacy protocol

1. **The originals in `data/video/` are processed, never viewed.** The assistant runs code on them and
   does not open them.
2. **Numbers travel freely**: `manifest.jsonl`, `vision-toolkit review status`, calibration statistics.
3. **The assistant looks only at two kinds of image:**
   - images in the `approved/` folder of a review queue, put there by the reviewer;
   - images that contain no person by construction: binary masks, skeletons on black.
4. **Every other image derived from the originals goes through a queue.** This covers the median, the
   board snapshots, `setup.png`, frames asked for by the calibration, and `data/video/sintetico.mp4`,
   which is built from real frames.

Two roles take part. The reviewer may look at the originals and is the one who releases images.
The assistant, in general an AI agent, may not: what it looks at leaves the machine.

## Where the person comes in

| # | When | What the reviewer does | Cost |
|---|---|---|---|
| L1 | Setup of a layout, once if the videos share the player frame | Opens `setup.png`, the median with the crop in green and the audience line in red, then confirms `setup.json` or corrects it (Paint.NET shows the cursor coordinates) | about 1 min |
| L2 | Doubtful frames | Opens the frame in Paint.NET, hides what the pipeline missed, saves and approves, or rejects | about 1 min per frame |
| L3 | Release | Answers yes or no on every `auto` frame, the ones the pipeline is confident about | about 2 s per frame |
| L4 | Derived images: median, snapshots, frames asked for by the calibration | Same release, after the assistant queues them with `review add` | about 2 s per image |
| L5 | Judging the board snapshots | Answers «is the lecturer absent?». A yes releases the snapshot, a no is a leak label: the ground truth of the calibration | about 2 s per snapshot |
| L6 | After the first video | Looks at how many `auto` frames needed fixing and how many doubts came out, then decides whether to keep checking every `auto` frame or only a sample, and whether doubts need a cap (`--review-budget`) | one decision |
| L7 | Already in place | Says «sì» to commits | — |

With `--step 300` a 90-minute lecture gives 18 frames: L3 and L5 take about a minute each, L2 a minute
per doubtful frame. How many doubts a lecture produces will be known after the first video.

## How it works

0. **Setup (L1).** Without a confirmed `setup.json` the run proposes one and stops. The crop is the
   non-black area of the median frame (`core.io.content_box`), so the player bars stay out, including
   the name in the Teams bar. `redact_below` starts empty. The reviewer checks `setup.png`, corrects the
   values if needed and sets `confirmed` to true. One setup can serve several videos (`--setup`), and
   `vision-toolkit board --setup` reads the same file (`core.layout`).

   Then, for each requested time (every `--step` seconds, or `--times`):
1. **Frame.** The frame is read with the crop (`core.io.FrameReader`), and the band below
   `redact_below` (the audience) is blacked out.
2. **Faces.** The heads come from the pose model (`num_poses = --max-people`), the frontal faces from
   YuNet. Each one becomes a face mask (`ops.faces.face_mask`), the union of two parts:
   - an ellipse over eyes, nose and mouth;
   - the `face-skin` pixels that `selfie-multiclass` finds on the crop around the head.

   The mask is pixelated (`core.redact.pixelate`) with `--blocks` blocks across the face. Hair, head
   outline and mean colour remain.
3. **Doubts** (`doubts.py`). A frame is doubtful when any of these holds:
   - **`person-without-head`:** foreground that no pose with a head explains. The foreground is
     the difference from the median, after compensating the exposure gain (the median ratio of
     the pixels, `core.foreground.exposure_gain`), cleared of thin changes such as chalk strokes
     (`core.foreground.foreground_mask`, shared with `board`);
   - **`face-outside-pose`:** a YuNet face far from every pose's head;
   - **`head-at-edge`:** a head touching the edge of the frame, where keypoints are unreliable.
4. **Resolving doubts.**
   1. **Substitute.** The frames within `--substitute-window` seconds are tried, nearest first; the
      first confident one takes the place of the original. The manifest records why
      (`replaced_because`).
   2. **Episodes.** The doubtful frames left that are close in time form an episode (`--episode-gap`,
      default twice the step). Only the middle frame of each episode goes to review.
   3. **Cap.** No cap by default. With `--review-budget N`, at most N episodes are queued, evenly
      spread in time.
   4. **Queue.** The chosen frames go to the `doubtful` group, already pixelated as far as the
      pipeline could, so the reviewer fixes only what is missing.
5. **Confident frames** go to the `auto` group.

## Review queue

The queue is a contract on files (`core/review.py`). It has no domain assumptions: `anonymize`, the
board snapshots and the future app all use it.

```
<queue>/
  queue.jsonl              one line per addition: name, group, question, reasons, boxes, sha256, created
  todo/<group>/<name>.png  waiting for a decision; edit the file in place
  approved/<name>.png      released: the only folder the assistant reads
  rejected/<name>.png      discarded
```

- **The folder is the state.** A decision moves the file, by hand in Explorer or from the app.
- **An approved item is `edited`** when its hash differs from the recorded `sha256`.
- **Producers never touch `approved/` or `rejected/`**, and they never replace an item in `todo/`
  that has been edited. A new run skips the names already decided.
- **`reasons` and `boxes`** say what to check and where. They are meant to be drawn over the image,
  never into it.

```
vision-toolkit review add QUEUE IMAGES... --group snapshots --question "Is the lecturer absent?"
vision-toolkit review status QUEUE
```

`review status` prints only counts, per group: todo, approved, edited, rejected, missing.

## The review section of the app

The toolkit's graphical app is designed separately: its skeleton first, then a review section. That
section must:
- **depend only on the queue contract**, work for any group and question, and know neither
  `anonymize` nor `board`;
- **show** the items to do, with `doubtful` first, and a live preview of the selected one, with its
  question, reasons and boxes drawn over it;
- **make decisions quick:**
  - open the item in the default PNG program (Paint.NET) and reload it once saved;
  - approve or reject it with one key and move to the next item;
  - undo the last decision.

  Proposed keys: Enter opens, y approves, n rejects.
- **say how much is left:** the items remaining and the minutes, at a minute per frame to fix and a
  few seconds per yes or no.

Until the app exists, the same review is done in Explorer and Paint.NET by moving files.

## Usage

```
vision-toolkit models download yunet-face selfie-multiclass mediapipe-pose-heavy
vision-toolkit anonymize data/video/lecture.mp4                # first run: proposes setup.json, stops
vision-toolkit anonymize data/video/lecture.mp4                # after confirming: fills the queue
vision-toolkit anonymize data/video/other.mp4 --setup output/anonymized/lecture/setup.json
vision-toolkit anonymize data/video/lecture.mp4 --times 693,1210.5   # frames asked for by the calibration
```

Main options:
- `--step`: seconds between two frames (default 300)
- `--times t1,t2,...`: explicit times, instead of `--step`, `--start` and `--end`
- `--setup file.json`: crop and audience line shared by videos with the same layout
- `--max-people`: people the pose model looks for (default 2)
- `--pose-model`, `--face-model` (`none` to disable), `--skin-model` (`none` to disable)
- `--blocks`: pixelation blocks across a face (default 6)
- `--review-budget`: most doubtful episodes to queue (default: no cap)
- `--substitute-window`: seconds searched for a confident substitute (default 3)
- `--episode-gap`: seconds that join doubtful frames into one episode (default twice the step, 0 with
  `--times`)
- `--min-blob`: smallest unexplained foreground, as a fraction of the frame, that is a doubt
  (default 0.002)

From Python: `vision_toolkit.usecases.anonymize.pipeline.run(video, out_dir, AnonymizeOptions(...))`.

## Output

In `output/anonymized/<video name>/`:
- `setup.json` (`crop`, `redact_below`, `frame_size`, `confirmed`) and `setup.png`;
- the review queue: `queue.jsonl`, `todo/auto/`, `todo/doubtful/`, then `approved/` and `rejected/`;
- `manifest.jsonl`: one line per requested time, rewritten at every run.

```json
{"name": "frame_00h05m00s", "t_requested": 300.0, "t": 301.0, "status": "queued", "group": "auto",
 "reasons": [], "heads_from_pose": 1, "faces_from_detector": 0,
 "replaced_because": ["person-without-head"], "skipped_because": null}
```

- `status`: `queued`, `skipped` or `decided` (already approved or rejected in an earlier run)
- `reasons`: the doubts of the queued frame, empty for `auto`
- `replaced_because`: the doubts of the requested frame when a substitute took its place
- `skipped_because`: `episode` (another frame of the episode went to review), `budget`, `unreadable`
  or `edited-in-todo`

Frame names follow `frame_<hh>h<mm>m<ss>s`, so two requested times within the same second give one
frame.

## Status

- **Tests:** implemented and tested on synthetic videos with fake models.
- **First real run, 2026-10-20:** the four lectures of 6, 8, 13 and 15 October, one frame every
  300 s, with the shared setup (crop `0,68,1674,944`, audience line 690). Every `auto` frame
  was approved without corrections. The doubts were:
  - the Teams title card in the first 3 s, now skipped with `--start 5`;
  - a stain that turned out to be the lecturer's trousers;
  - two harmless doubts;
  - new chalk on the board.
- **The doubt thresholds** (`--min-blob`, the difference threshold, the opening), the episode gap and
  a review budget are still to be decided (L6).
- **Fragile point:** the foreground is the difference from the median of the whole lecture. Writing
  that appears later is missing from that median and comes out as foreground; so can erasing the
  board, or a laptop screen that changes. A median over a time window would avoid it. A false doubt
  costs the reviewer a minute. A missed doubt is caught by the release (L3).
