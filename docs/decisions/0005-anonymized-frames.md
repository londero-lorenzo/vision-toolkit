# 0005: Frames leave the machine only through a person's review

Status: accepted (2026-10-20).

## Context

The board use case needs calibrating on real lectures: Lorenzo downloaded four whole lectures,
recorded from the Teams player, so that Claude can see the angles, the light and the saturation. The
frames show the lecturer and the students, who have not agreed to be shown, and sending a frame to
Claude means sending it out of the machine.

Lorenzo first censored 36 frames by hand (`data/video/censured`), blurring the face. That does not
scale to whole lectures. It also left the lecturer's name readable in the Teams bar. The idea of
masking faces with a black rectangle raised a doubt: the models need to see the face.

Two facts settle the doubt:
- the models never run on the anonymized copies: they run locally on the originals, and the copies
  are only for Claude's eyes;
- the pose detector (BlazePose) uses the face as its main signal, so a hidden face would make the
  measurements worse.

## Decision

1. **Measurements on the originals, numbers only.** Calibration scripts run on the original videos
   and print statistics. Claude never opens an original.
2. **Images only through a review queue.** Claude looks only at the `approved/` folder of a queue,
   where Lorenzo puts the images, and at images that contain no person by construction (binary masks,
   skeletons on black). The median, the board snapshots and `setup.png` go through the queue like the
   frames.
3. **`vision-toolkit anonymize` fills the queue.** It pixelates the skin of each face. The faces come
   from the heads of the pose model and from YuNet, and the skin from an ellipse over the features
   joined with the `face-skin` of `selfie-multiclass`. Hair, head outline and mean colour remain,
   which a calibration needs. The player bars are cropped away and the audience band is blacked out.
4. **A person in the loop where the pipeline cannot vouch for itself.** Doubtful frames are handled
   in this order:
   - a confident frame up to a few seconds away replaces the doubtful one;
   - the doubtful frames left that are close in time become one episode, of which only one frame goes
     to Lorenzo;
   - there is no cap by default.

   Lorenzo fixes doubtful frames in Paint.NET at about a minute each, and answers yes or no on the
   confident ones. He also confirms the setup and judges the board snapshots, and his answers become
   the ground truth of the calibration. The points L1–L7 are listed in
   [the use case page](../usecases/anonymize.md#where-the-person-comes-in).
5. **The queue is a contract on files, in `core/review.py`.** The folder of a file is its state, a
   decision is a move, and an edit is a changed hash. Any producer (a use case, Claude with
   `review add`) and any consumer (Explorer today, an app section tomorrow) can use it without knowing
   each other.

## Consequences

- No image of a person reaches Claude without Lorenzo's decision, and Lorenzo's time is spent where
  the pipeline is unsure.
- A new task (`detection`) and a new backend (`models/opencv/`, no extra) entered the toolkit. Canvas
  will add its OpenCV models there.
- **False doubts cost a minute each, missed doubts are caught by the release.** The doubt detection
  ranks the work and is not the only safeguard.
- Lorenzo's decisions are data: how many confident frames he had to fix measures the anonymizer's
  real miss rate.

## Not done (yet)

- **The graphical app.** Its skeleton is designed separately, and the review section is added to it
  afterwards, on this contract.
- **Ground truth by clicks** (fingertip, board corners): a candidate for another section of the app.
- **Tuning the doubt thresholds**, after the first real video.
- **Changing the release after the first video**, from every confident frame to a sample, if Lorenzo
  decides so.
