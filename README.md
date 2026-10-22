# vision-toolkit

Python toolkit for image and video processing: separate subjects from the background, rebuild what
they cover, follow gestures and positions over time. Each concrete application is a **use case**
built on reusable layers. Models are **pluggable**: a use case picks them by name, so a new model, or
a model for a new domain, does not change the code that uses it.

## Use cases

| Use case | Command | What it does | Docs |
|---|---|---|---|
| Board without lecturer | `vision-toolkit board` | removes the lecturer from lecture videos, saves a clean snapshot of the board at every change and logs the pointed spot with its timestamp | [docs/usecases/board.md](docs/usecases/board.md) |
| Anonymize | `vision-toolkit anonymize` | pixelates the faces in frames of a video and puts the frames in a review queue, where a person approves, fixes or rejects each one before release | [docs/usecases/anonymize.md](docs/usecases/anonymize.md) |
| Canvas stitching | `vision-toolkit canvas` (planned) | merges overlapping exports of a canvas, absorbed from `image-stitching` | [architecture](docs/architecture.md#where-future-work-lands) |

## Layout

```
src/vision_toolkit/
  core/        video I/O, plane geometry, temporal reconstruction, redaction, review queues (plain NumPy/OpenCV)
  tasks/       one interface per domain: segmentation, pose, detection (Protocol + NumPy results)
  models/      implementations by backend (mediapipe/, opencv/) + catalog.py: models by name
  ops/         model-agnostic algorithms on task results: crop_refine, skeleton, pointing, faces
  usecases/    applications: board/, anonymize/
  cli.py       `vision-toolkit <command>`
tests/         mirrors the layers; synthetic data, plus one test on a private frame
notebooks/     explorations (01 single frame, 02 temporal reconstruction)
docs/          architecture, migration guide, decisions, fix notes, one page per use case
data/          frames, videos, calibration (outside git)
models/        downloaded model files (outside git)
output/        results (outside git)
```

Each layer imports only the layers below it, and use cases reach models only through the catalog.
[docs/architecture.md](docs/architecture.md) describes every layer, what belongs there and how to
change it. `tests/test_layers.py` keeps the import rules true.

## Models

```
$ vision-toolkit models list
deeplab-v3             segmentation  mediapipe  ready
mediapipe-pose-heavy   pose          mediapipe  ready
selfie-multiclass      segmentation  mediapipe  ready
yunet-face             detection     opencv     ready
```

| Name | Task | Notes |
|---|---|---|
| `mediapipe-pose-heavy` | pose | 33 named keypoints and a body mask; two-stage, finds people that are small in the frame |
| `deeplab-v3` | segmentation | 257×257 input, 21 PASCAL VOC labels; misses small subjects in a full frame, so run it on a crop |
| `selfie-multiclass` | segmentation | hair, body-skin, face-skin, clothes; made for close-up people |
| `yunet-face` | detection | OpenCV YuNet: faces down to about 10 px with five keypoints; runs on OpenCV, no extra |

To add a model:
1. If the domain is new, add its interface in `tasks/`.
2. Implement the class in `models/<backend>/`. A new backend also gets its own pip extra.
3. Add a `ModelSpec` to `models/catalog.py`. An external package can do this through the
   `vision_toolkit.models` entry point instead.
4. Use it from a use case by name, e.g. `vision-toolkit board ... --refine-model <name>`.

The details for each kind of change are in [docs/architecture.md](docs/architecture.md#models-and-the-catalog).

## Setup

Windows, Python 3.12, CPU only.

```
py -m venv .env
.env\Scripts\activate
pip install -e ".[dev,mediapipe]"
vision-toolkit models download     # model files into models/ (or $VISION_TOOLKIT_MODELS)
pytest -s tests
jupyter lab                        # from the root, then notebooks/
```

The core needs only OpenCV and NumPy. Each model backend is a pip extra (`mediapipe` today).
Without an extra, only the models that need it fail, and the message names the extra to install.

The tests use synthetic data. `tests/usecases/test_board_synthetic.py` also needs a private lecture
frame (`data/frame/lezione_prova_cit_censured.png`) and is skipped without it. Lecture frames show a
real person: they are never committed or published.

## Lessons that generalise

From the first use case, valid beyond the board:
- Models with a small input (DeepLab V3 at 257×257) lose subjects that are small in the full frame.
  Crop first (`ops.crop_refine`), or use a two-stage model such as PoseLandmarker.
- A static background with a moving subject is rebuilt with the temporal median of the uncovered
  pixels, after bringing them onto a common plane with a homography (`core.temporal.PlaneMedian`).
- Colour separates a uniform background well, but not light subjects from light strokes: use it to
  refine, not as the main criterion.
- Pose keypoints give more than a mask, for example the pointed spot and the arm angle (`ops.pointing`).

## Documentation

- [Architecture](docs/architecture.md): layers, rules, and how to change each layer
- [Migration guide](docs/migration-guide.md): from `image-segmentation` / `lavagna` to vision-toolkit
- Decisions: [0001 umbrella toolkit](docs/decisions/0001-umbrella-toolkit.md),
  [0002 single package](docs/decisions/0002-single-package.md),
  [0003 model layer](docs/decisions/0003-model-layer.md),
  [0004 English](docs/decisions/0004-english.md),
  [0005 anonymized frames](docs/decisions/0005-anonymized-frames.md)
- Use cases: [board without lecturer](docs/usecases/board.md), [anonymize](docs/usecases/anonymize.md)
- [Fixes](docs/fixes/README.md): one note per correction to the code

## Status

- The board use case is complete and has run on four real lectures: after one round of
  corrections, no artefact was found in its snapshots. Pointing is measured only on a synthetic
  video ([status](docs/usecases/board.md#status)).
- The anonymize use case releases frames of the real lectures to the assistant only after the
  reviewer's approval
  ([status](docs/usecases/anonymize.md#status)).
- Next: absorb `image-stitching` as the canvas use case.
