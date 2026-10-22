# CLAUDE.md

**vision-toolkit**: a Python toolkit for image and video processing. It separates subjects from the
background, rebuilds what they cover and follows gestures and positions over time. The project is
**general**: every concrete application is a use case built on reusable layers, and models are
pluggable, picked by name from a catalog.

## State

- First use case, implemented: **board without lecturer** (`vision-toolkit board`). From lecture
  videos it removes the lecturer, saves a clean board snapshot at every change and logs the pointed
  spot with its timestamp. Page: `docs/usecases/board.md`. It was calibrated blind on the four real
  lectures. Its snapshots were judged by hand (L5): no person, 21 artefacts in 189, whose causes are
  corrected; none was found on the corrected run. On six more lectures 1 in 245 was found, corrected
  in `docs/fixes/0017-upper-body-cut-off-from-the-legs.md`; none was found on the corrected run of
  all ten lectures. Pointing is right in 7 frames out of 10 on the synthetic video, and is not
  measured on the real lectures.
- Second use case, implemented: **anonymize** (`vision-toolkit anonymize`). It pixelates the faces in
  frames of a video and puts the frames in a review queue for the reviewer. Page:
  `docs/usecases/anonymize.md`, decision `docs/decisions/0005-anonymized-frames.md`. It has run on
  the four real lectures, and every frame was approved. Its doubt thresholds are still open (L6).
- Next step: absorb `D:\Programmazione\Projects\image-stitching` as the `canvas` use case. Where it
  lands (a registration task, OpenCV SIFT/ORB as models, warp and composite, `usecases/canvas/`) is
  in `docs/architecture.md`.

## Layout

```
src/vision_toolkit/
  core/       video I/O, camera framing, plane geometry, temporal reconstruction, foreground,
              redaction, recording layout, review queue (imports nothing internal)
  tasks/      interfaces per domain: Protocol + NumPy results (segmentation, pose, detection)
  models/     implementations by backend (mediapipe/, opencv/) + catalog.py (models by name, downloads)
  ops/        model-agnostic algorithms on task results (crop_refine, skeleton, pointing, faces)
  usecases/   board/ (detect, person_mask, chalk, pipeline, cli), anonymize/ (doubts, pipeline, cli)
  cli.py      `vision-toolkit models|review|board|anonymize`
```

`docs/architecture.md` is the reference. It covers the layer rules, what belongs in each layer and
how to make each kind of change. Read it before adding a model, a task, an op or a use case.
`tests/test_layers.py` enforces the import rules and checks that every module appears in the
document. A new module therefore goes into the document's module map too. Larger decisions get a
record in `docs/decisions/`, in the existing format.

Board output format: snapshots `board_<hh>h<mm>m<ss>s.png`, `board_final.png`, `median.png`,
`quad.json` (key `quad`) and `pointing.jsonl` (keys `t, frame_xy, board_xy, board_xy_norm,
arm_angle, occluded_fraction, snapshot`). The vault reads it, and `tests/usecases/test_board_pipeline.py`
pins it. Any change must be agreed with the maintainer first. `run.json` records what a run measured and
decided; the vault does not read it.

## Roles

- **The maintainer** owns the repository: says «sì» to commits, agrees changes to the output format
  and to the environment, and decides what is in scope.
- **The reviewer** may see the lecture videos and releases images from a review queue (L1–L6 in
  `docs/usecases/anonymize.md`).
- **The assistant** is, in general, an AI agent working on the repository. What it looks at leaves
  the machine, so it never sees an original frame.

For now the maintainer and the reviewer are the same person. The documents keep the roles apart.

## Data

- `data/`, `models/` and `output/`, as well as images and videos, are outside git (`.gitignore`).
  Lecture frames show a real person: they are never committed, published or attached to issues and PRs.
- The assistant processes the lecture videos but never opens them. It looks only at images in the `approved/`
  folder of a review queue, released by the reviewer, and at images with no person by construction (binary
  masks, skeletons on black). To see the median, a snapshot or `setup.png`, the assistant first puts it in a
  queue with `vision-toolkit review add` (`docs/usecases/anonymize.md`, privacy protocol).
- Notebooks are committed **without output**.
- Public tests and examples use synthetic data (`tests/fakes.py`).
- `tests/usecases/test_board_synthetic.py` needs `data/frame/lezione_prova_cit_censured.png` and is
  skipped without it.
- The university vault `D:\Uniud\ms-aics` provides the sources and is **out of scope**: read it only
  if the maintainer asks, never write to it.

## Environment

- Windows, Git Bash. `python` is not on PATH: use `.env\Scripts\python` or the `py` launcher.
- Versions: Python 3.12, OpenCV 5.0, NumPy 2.x, MediaPipe Tasks 1.1.1.
- CPU only. The GPU (GTX 1050 Ti, 4 GB) is not used.
- `mediapipe` requires `opencv-contrib-python`, while the core declares `opencv-python`. Both install
  the same `cv2` module, which is why the environment has both. Do not change this without the maintainer.

```
py -m venv .env
.env\Scripts\activate
pip install -e ".[dev,mediapipe]"
vision-toolkit models download       # model files into models/ (or $VISION_TOOLKIT_MODELS)
pytest -s tests
jupyter lab                          # from the root, then notebooks/
```

**MediaPipe:** masks are always read as `.numpy_view().squeeze().copy()`, through
`models/mediapipe/_common.py: read_mask`. The view points to the result's memory. If the result is a
temporary, the next access causes a native crash (dead Jupyter kernel, access violation), not an exception.

## Conventions

- **Language:** English everywhere: code, docstrings, comments, CLI, notebooks, docs, this file
  (`docs/decisions/0004-english.md`). Conversations with the maintainer are in Italian. Commits are in English,
  as conventional commits with a scope (`feat(models): ...`), like in `image-stitching`.
- **Impersonal form:** documents, code and comments name roles (the maintainer, the reviewer, the
  assistant), never a person, and use no "I/we". The lecturer is "the lecturer", with no pronoun.
- **Git:** no commit without the maintainer's «sì». Propose the files and the message, then wait.
  Commits carry the assistant's `Co-Authored-By` line. Add files by name, never `git add -A`. Never `push --force` or
  `--no-verify`. No remote and no `gh repo create` without confirmation.
  Nothing has been pushed yet: the history can still be analysed and its commits changed with a
  rebase, with the maintainer's agreement. Once pushed, it no longer can.
- **Before the first push:** `docs/impersonal-form` is merged into `main`, so the commit messages
  can be rewritten in one pass, with the maintainer's agreement:
  - `7ecc290` and `17232eb`: the lecturer with no pronoun;
  - `17232eb` and `6652c37`: the note they call 0001 is now 0017;
  - `e638f6d`, `7ecc290` and `2bfeec1`: name their notes (0001–0010, 0011–0015, 0016).

  After the merge, `git log` names no other person and has no "I/we". The history holds a merge,
  so only the messages are rewritten (`git filter-branch --msg-filter`): a rebase would replay the
  merge and its conflict on 0017. The rewrite changes every hash from `e638f6d` on. `docs/fixes/`
  cites them (each note and the table in its README), so the citations are updated after the
  rewrite. This item is then removed.
- **Fixes:** every correction to the code gets a note in `docs/fixes/` (`NNNN-<slug>.md`, in the format
  of 0017). The note says what went wrong and how it showed, the cause, what changed and how it was
  verified. It goes in the same commit as the fix, with its row in `docs/fixes/README.md`, and the
  commit message names it. Notes 0001–0016 were written after the fact, on 2026-10-22, for the
  corrections made before the notes started.
- The context of the session that created the repo (experiments, figures, reasons) is in the assistant's
  memory for this project.
