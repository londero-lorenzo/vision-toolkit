# Roadmap

The work left, split into sessions. Each session is one conversation with the assistant, on a
branch of its own, and ends with its commits merged into `main`. When a session ends, its section
is removed from this page, and what it left open moves to the session it belongs to.

## Order

1. **Sessions 2, 4 and 5 in parallel**, each on its own branch. Session 3 follows session 2.
2. **Session 6** once the skeleton of the app is designed.

Sessions 2 and 3 change the same files of the board use case, so they do not run together. The
other sessions share few files: the State in `CLAUDE.md`, the Status in `README.md` and
`docs/architecture.md`. Their merge conflicts stay there.

## 2. Board: pointing on the real lectures

Medium to long.

- **Where it stands.** The pointed spot is right in 7 frames out of 10 on the synthetic video, and
  is not measured on the real lectures. There the arm is raised above 60° in 51–60% of the frames,
  because the lecturer writes: the arm angle alone does not tell pointing from writing.
- **What to try**, from `docs/usecases/board.md`:
  - MediaPipe's VIDEO mode, which tracks the pose between frames;
  - a pointing gesture accepted only if it lasts 2 or more samples, and away from the chalk that
    appears.
- **What it needs.** Measuring it needs labels from the reviewer. They can come through a review
  queue of skeletons on black, which the assistant may see too.

## 3. Board: export for the vault

Medium. It comes after session 2: it changes the same pipeline, and the pointed areas are worth
exporting only once pointing is reliable.

- **The export** for the summaries: snapshots, transcript intervals and pointed areas.
- **Query by timestamp:** given a moment or a sentence of the transcript, the snapshot and the
  pointed area in its window.
- **Word-level timestamps:** whether to transcribe again with Whisper, to narrow the window.
- **Limits.** Any change to the output format is agreed with the maintainer first. The vault is read
  only if the maintainer asks, and never written.

## 4. Anonymize: the doubt thresholds (L6)

Short to medium. Independent of the others.

- **To decide:** the doubt thresholds (`--min-blob`, the difference threshold, the opening), the
  episode gap and a review budget (`--review-budget`).
- **The fragile point** in `docs/usecases/anonymize.md`: the foreground is the difference from the
  median of the whole lecture, so writing that appears later comes out as a doubt. A median over a
  time window would avoid it.
- **If useful,** a run on the six lectures added on 2026-10-21. Anonymize has run on the first four
  only.

## 5. Canvas: absorbing image-stitching

Long. Independent of the others, and it uses no lecture video.

- **The plan** is in "Where future work lands" in `docs/architecture.md`: `tasks/registration.py`,
  SIFT and ORB as models in `models/opencv/`, warping and compositing on a canvas, and
  `vision-toolkit canvas`.

## 6. Graphical app with a review section

To design. The maintainer designs the skeleton of the app in a session of its own. Its review
section depends only on the `core.review` contract. What it must offer is in "The review section
of the app" in `docs/usecases/anonymize.md`.

## Later: a moving camera

The two lectures of 1 October in `data/video/moving_camera/` were filmed by a laptop camera moved
by hand. The starting point is `core.framing`, which splits a video into segments at every camera
jump. This waits until the sessions above are done.

## Cleanup outside git

With the maintainer's confirmation:
- `output/board_v1/` and `output/board_v3/` hold superseded runs. `output/board/`,
  `output/board_v2/` and `output/board_v5/` stay, as the references for comparing snapshots.
