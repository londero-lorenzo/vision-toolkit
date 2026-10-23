# Architecture

vision-toolkit is one Python distribution (`vision_toolkit`) organised in layers. Each layer knows
only the layers below it. Every concrete application is a **use case** on top. Models are
**pluggable**: a use case asks for a model by name and receives any implementation of the right
task interface.

This document is the reference for future changes. For each layer it says what the layer does, what
belongs there and what does not, what it may import, and how to make each kind of change.
`tests/test_layers.py` enforces the import rules and checks that every module appears in the module
map below. If a rule has to change, update this document first, then the test.

## Layers

```
cli         vision-toolkit <command>: argument parsing, error messages
 │
 ▼
usecases    board/, anonymize/ (later canvas/): pick models by name, compose ops and core, write outputs
 │    │
 │    └───► models/catalog.py ┄┄ lazy import ┄┄► models/<backend>/   mediapipe/, opencv/ (later onnx/, torch/)
 ▼                                                     │
ops         model-agnostic algorithms on task results  │ implement
 │                                                     │
 ▼                                                     │
tasks       interfaces: Protocol + NumPy results ◄─────┘
 │
 ▼
core        foundations on plain arrays: video I/O, camera framing, plane geometry, temporal
            reconstruction, foreground, redaction, recording layout, review queues, paths
            (every layer may import core; core imports nothing internal)
```

Dependency rules (arrows point only downward):

| Layer | May import | Must not import |
|---|---|---|
| `core` | other `core` modules; numpy, opencv | anything else in the package |
| `tasks` | `core` | models, ops, use cases, backend libraries |
| `models/<backend>/` | `core`, `tasks`, its own backend library | other backends, `catalog`, ops, use cases |
| `models/catalog.py` | `core` | any backend module (factories are strings, imported on demand) |
| `ops` | `core`, `tasks` | `models` (an op receives a model object, never creates one), use cases |
| `usecases/<name>/` | `core`, `tasks`, `ops`, `models.catalog` | `models/<backend>/`, other use cases |
| `cli` | everything except `models/<backend>/` | — |

Two more rules:
- Backend libraries (`mediapipe`, `onnxruntime`, `torch`, ...) are imported only inside `models/<backend>/`.
  The package therefore imports and runs without any backend installed. Only the models that need a
  missing backend fail, with a message naming the pip extra. OpenCV is a core dependency: any layer
  may import `cv2`, and `models/opencv/` needs no extra.
- Use cases do not import each other. Code that two use cases need moves down, to `ops` or `core`.

## Module map

```
src/vision_toolkit/
  __init__.py                 __version__
  cli.py                      `vision-toolkit`: `models list|download`, `review add|status`, then one
                              subcommand per use case
  core/
    paths.py                  models_dir(): repo models/ or $VISION_TOOLKIT_MODELS
    io.py                     iter_frames, FrameReader, apply_crop, content_box, median_frame, format_time
    framing.py                camera_motion, framing_segments: stretches of video with one camera framing (ECC)
    geometry.py               order_corners, polygon_mask, rectify_homography, to_plane, save_quad/load_quad
    temporal.py               PlaneMedian: masked per-pixel temporal median on a rectified plane, exposure matched
    foreground.py             exposure_gain, foreground_mask, blobs_touching: a frame against a static
                              background
    redact.py                 pixelate, black_out_below
    layout.py                 Setup, load_setup, save_setup: crop and audience line of a recording (setup.json)
    review.py                 ReviewQueue: images waiting for a person's decision, kept as plain files
  tasks/
    base.py                   Model: context-manager base class for implementations
    segmentation.py           Segmenter (Protocol), Segmentation (category mask, labels, confidence)
    pose.py                   PoseEstimator (Protocol), Pose (named keypoints + mask), Keypoint
    detection.py              Detector (Protocol), Detection (box, score, label, named keypoints)
  models/
    catalog.py                ModelSpec, BUILTIN, register, get, available, create, download, entry points
    mediapipe/
      _common.py              base_options, to_mp_image, read_mask (the only numpy_view() reader)
      image_segmenter.py      ImageSegmenter: any MediaPipe segmenter (deeplab-v3, selfie-multiclass)
      pose_landmarker.py      PoseLandmarker: 33 named keypoints + body mask (mediapipe-pose-heavy)
    opencv/
      face_detector.py        YuNetFaceDetector: faces with five named keypoints (yunet-face)
  ops/
    crop_refine.py            bounding_box, segment_on_crop: run any Segmenter on a crop, paste back
    skeleton.py               ARMS, shoulder_width, draw_limbs: limbs between named keypoints onto a mask
    pointing.py               Pointing, arm_angle, pointing_hand: the more raised arm and its fingertip
    faces.py                  HEAD_KEYPOINTS, head_width, head_box, face_mask: heads of poses and face detections
  usecases/
    board/
      detect.py               find_board, detect_board, refine_quad: dark board on a light wall (Otsu + contour),
                              sides moved to the real edge
      person_mask.py          PersonMasker, person_blob: pose mask + arm skeleton + optional refine model
                              + dilation + the foreground blobs touching it; a person-shaped foreground
                              blob when the pose finds nobody, else the blobs touching the last mask
      chalk.py                stroke_contrast, change_threshold: the snapshot threshold from the chalk
      pipeline.py             BoardOptions, run: framing segments, snapshot policy, pointing.jsonl, run.json
      cli.py                  `vision-toolkit board` options
    anonymize/
      doubts.py               unexplained_blobs, episodes, select: which frames need a person
      pipeline.py             AnonymizeOptions, Setup, run: setup, pixelated faces, review queue, manifest.jsonl
      cli.py                  `vision-toolkit anonymize` options
```

Outside `src/`: `tests/` mirrors the layers (`tests/core`, `tests/models`, `tests/ops`,
`tests/usecases`). `tests/fakes.py` holds fake models and synthetic scenes. `notebooks/` holds the
explorations, `docs/` the documentation. `models/`, `data/` and `output/` stay out of git.

## Conventions shared by every task

- Input images are `uint8` BGR arrays (the OpenCV convention). Models convert to their own format inside.
- Masks are `uint8` arrays with values 0/255, the same size as the input image.
- Coordinates are pixels of the input image (models convert from normalised coordinates).
- Classes and keypoints are **names**, never a model's indices: `seg.mask_of("person")`,
  `pose["left_index"]`. Keypoint names follow COCO (`left_shoulder`, ...) and MediaPipe where COCO
  has no such point (`left_index`, `left_heel`, ...).
- Every model is a context manager: `with catalog.create("deeplab-v3") as model: ...`.

## How a model reaches a use case

```
vision-toolkit board video.mp4 --refine-model deeplab-v3
  └─ usecases/board/person_mask.py
       catalog.create("mediapipe-pose-heavy", task="pose")
         └─ ModelSpec: task ok? files in models/? → import "vision_toolkit.models.mediapipe.pose_landmarker"
            → PoseLandmarker(models/pose_landmarker_heavy.task)          satisfies tasks.pose.PoseEstimator
       catalog.create("deeplab-v3", task="segmentation")
         └─ ImageSegmenter(models/deeplab_v3.tflite)                     satisfies tasks.segmentation.Segmenter
       per frame:
         poses = pose(frame)                       list[Pose] with named keypoints and a mask
         ops.skeleton.draw_limbs(mask, poses[0])   covers the stretched arm the mask loses
         ops.crop_refine.segment_on_crop(...)      DeepLab on the crop, where the lecturer is big enough
         ops.pointing.pointing_hand(poses[0])      fingertip + arm angle
```

`create` raises `CatalogError` with an actionable message in four cases: unknown name, wrong task,
missing files (`vision-toolkit models download <name>`) or missing backend (`pip install -e ".[extra]"`).

## Layer by layer

### core

**Responsibility.** Foundations that any layer may need, on plain NumPy and OpenCV: reading frames
from a video, telling where the camera moved, plane geometry (quadrilaterals, homographies), the
temporal reconstruction of a static background on a rectified plane, comparing a frame with a
background, and paths.

**Belongs here:** `iter_frames`, `median_frame`, `framing_segments`, `rectify_homography`, `to_plane`,
`PlaneMedian`, `exposure_gain`, `foreground_mask`, `blobs_touching`, `pixelate`, `Setup`,
`ReviewQueue`. The test is that the algorithm needs no model, no task result and no domain
assumption. The review queue qualifies: it
is a contract on files, and it knows neither the use cases that fill it nor the app that will work
through it. So does the recording layout: a crop and a row, confirmed by a person, that two use cases
read.

**Does not belong here:**
- domain assumptions such as "a dark board on a light wall" (`detect_board` lives in `usecases/board/`);
- anything that runs a model or reads a `Pose` or a `Segmentation` (that goes to `ops`);
- any dependency beyond numpy and opencv.

**May import:** other `core` modules only.

| Change | How | Blast radius | Tests |
|---|---|---|---|
| Add a utility (e.g. `write_video` in `io.py`) | add the function to the module of its kind and list it in the module map | local | `tests/core/test_<module>.py` |
| Add a module (e.g. `core/color.py`) | new file, listed in the module map | local | new `tests/core/test_<module>.py` |
| Change a signature or behaviour | search the callers in `ops`, `usecases`, notebooks; update them in the same change | every caller, possibly every layer | existing tests of all callers |
| Add a third-party dependency | only if light and needed by core: add it to `dependencies` in `pyproject.toml`. Otherwise put the code in a higher layer behind an extra | every installation | — |

### tasks

**Responsibility.** The contract of each domain: a `Protocol` the models satisfy and result
dataclasses in plain NumPy. Thanks to it, models and ops evolve independently.

**Belongs here:**
- `Segmenter` / `Segmentation` (with `mask_of`, `confidence_of`);
- `PoseEstimator` / `Pose` / `Keypoint`;
- `Detector` / `Detection`, whose keypoints use the names of `tasks.pose`;
- the `Model` base class;
- small helpers on results.

**Does not belong here:**
- implementations (`models`) and algorithms (`ops`);
- backend types: an `mp.Image` or a landmark index never crosses this layer.

**May import:** `core`.

| Change | How | Blast radius | Tests |
|---|---|---|---|
| Add a domain (detection, depth, OCR, tracking, registration) | new `tasks/<domain>.py` with a Protocol and result dataclasses that follow the conventions above. Add it when a use case needs it, together with at least one model and its `ModelSpec`. Add the task name to `test_builtin_specs_are_consistent` | new file only | a fake model in `tests/fakes.py`, used by the ops tests |
| Add an optional field to a result (as `Segmentation.confidence`) | default `None`; models fill it when they can | compatible: models and ops that ignore it keep working | a test on the model that fills it |
| Add keypoint names | COCO first, then MediaPipe naming | compatible | — |
| Change or remove a field, or change a convention | write a decision record in `docs/decisions/`, then update every model of the task and every op and use case that reads it, in one change | **breaking** for all models, ops and use cases of that task | all of them |

### models and the catalog

**Responsibility.**
- `models/<backend>/` implements task interfaces. The conversion from backend types to task results
  happens here and only here: MediaPipe landmark indices become keypoint names, and
  `numpy_view().squeeze().copy()` turns MediaPipe masks into arrays.
- `models/catalog.py` maps a **name** to a `ModelSpec` with these fields:
  - `task`, `backend`;
  - `factory` (`"module:Class"`);
  - `files` (file name → download URL);
  - `extra` (pip extra);
  - `options` (default keyword arguments);
  - `description`.

  It creates models (`create`), lists them (`available`) and downloads their files (`download`).
  External packages add specs through the entry-point group `vision_toolkit.models`.

**Belongs here:** one class per model family (`ImageSegmenter` serves every MediaPipe segmenter),
plus the specs of the models in use.

**Does not belong here:** post-processing that would help any model of the task (drawing the
skeleton, crop-and-refine), which goes to `ops`, and choosing which model a use case uses, which is
the use case's default.

**May import:** `core`, `tasks`, and its own backend library. The catalog imports only `core`.

| Change | How | Blast radius | Tests |
|---|---|---|---|
| Add a model that an existing class already handles (e.g. PoseLandmarker *full*, another MediaPipe segmenter) | add a `ModelSpec` to `BUILTIN`, and its licence to `docs/licensing.md` | none | the catalog consistency test covers it |
| Add a model that needs a new class (e.g. an ONNX SAM as a segmenter) | `models/<backend>/<model>.py` with a class satisfying the task Protocol (subclass `tasks.base.Model`), plus a `ModelSpec` and its licence in `docs/licensing.md`. Use cases pick it by name (`--refine-model sam`) | local | contract test in `tests/models/`, skipped without files or extra |
| Add a backend (ONNX Runtime, PyTorch) | `models/<backend>/` with a `_common.py` for conversions; a pip extra in `[project.optional-dependencies]`; the library in `BACKEND_LIBRARIES` in `tests/test_layers.py` | none for users without the extra | `pytest.importorskip("<library>")` |
| Upgrade a model file | change URL and file name in the spec. If labels or keypoints change, check the ops and use cases that name them | users re-run `models download` | contract test |
| Rename or remove a model | breaking for CLI users and scripts: write it in the migration guide | users of that name | — |
| Ship models from another package | in that package: `[project.entry-points."vision_toolkit.models"] my-models = "my_pkg.specs:SPECS"` (a spec, a list of specs or a callable returning them) | none in this repo | — |

A broken plugin is skipped with a warning, so it never breaks the built-in models.

The catalog holds only models whose licence allows any use, commercial use included. Models under
copyleft or non-commercial licences ship from separate packages through the entry points
([licensing](licensing.md), [0007](decisions/0007-apache-2.0-licence.md)).

### ops

**Responsibility.** Model-agnostic algorithms on task interfaces and results. The lessons that hold
for any model live here: "small-input models miss small subjects: crop first"
(`segment_on_crop`), "the skeleton covers what the body mask misses" (`draw_limbs`), "arm angle,
not arm length, tells pointing from resting" (`pointing_hand`).

**Belongs here:** functions that take a `Segmenter`, a `PoseEstimator` or their results.

**Does not belong here:**
- creating a model by name (an op receives a model object);
- domain policy, such as which labels mean "lecturer" or when to take a snapshot (that is the use case's);
- algorithms on plain arrays that need no task (those go to `core`).

**May import:** `core`, `tasks`.

| Change | How | Blast radius | Tests |
|---|---|---|---|
| Add an op | a function in the module of its kind or a new module, typed on task interfaces only | local | `tests/ops/` with a fake model or a hand-made result (`tests/fakes.py`), no model files |
| Promote use-case code to a generic op | move it from `usecases/<name>/` to `ops` (if it uses task results) or `core` (plain arrays), and turn constants into parameters with the old values as defaults | imports of that use case | move the tests and add generic ones |
| Change an op's behaviour | check every use case that calls it. If the board outputs change, state it (the vault reads them) | every use case using it | rerun the use-case comparisons |

### usecases

**Responsibility.** Concrete applications. A use case:
- chooses models by name, with defaults and CLI options;
- composes ops and core;
- owns its domain assumptions;
- defines its output files and their format;
- adds its CLI subcommand.

**Belongs here** (`board/`):
- `detect.py`: dark board on a light wall, with a light strip that may run along its top;
- `person_mask.py`: the lecturer mask recipe;
- `chalk.py`: what counts as writing;
- `pipeline.py`: the framing segments, the snapshot policy and the `pointing.jsonl` format;
- `cli.py`.

And in `anonymize/`:
- `doubts.py`: the policy that decides which frames a person must see;
- `pipeline.py`: the setup, the queue groups and the `manifest.jsonl` format;
- `cli.py`.

**Does not belong here:** code another use case would need (move it down) and backend imports.

**May import:** `core`, `tasks`, `ops`, `models.catalog`.

| Change | How | Blast radius | Tests |
|---|---|---|---|
| Add a use case | `usecases/<name>/` with a pipeline and `cli.py` exposing `add_cli(subparsers)`. Append it to `USECASES` in `cli.py`, write `docs/usecases/<name>.md`, link it from the README | local | an end-to-end test on synthetic data in `tests/usecases/` |
| Add an option | a field in the options dataclass and an argument in `cli.py`, with today's value as default | local | — |
| Change a default model | change the default constant (e.g. `DEFAULT_POSE_MODEL`). Outputs may change: state it | outputs of that use case | rerun the synthetic comparison |
| Change the output format (file names, `pointing.jsonl` keys) | the vault reads these files: agree it with the maintainer first, then update `docs/usecases/<name>.md`, the migration guide and the pinned format in `tests/usecases/test_board_pipeline.py` | vault consumers | `test_board_pipeline.py` |

### cli

**Responsibility.** The `vision-toolkit` command: parse arguments, dispatch, turn `CatalogError`
into a one-line message. No logic of its own.

| Change | How | Blast radius |
|---|---|---|
| New use-case command | the use case's `add_cli`, appended to `USECASES` | local |
| New toolkit command (like `models`, `review`) | an `add_<name>_cli` function in `cli.py` | local |
| Use cases from other packages | add an entry-point group (as pdf-toolkit does for its modules) when an external package needs it | `cli.py` only |

## Where future work lands

- **Stitching (step 2, from `image-stitching`).** Registration is a new domain:
  - `tasks/registration.py`: a `Registrar` that takes an image pair and returns a homography with its matches;
  - `models/opencv/`, which already holds YuNet: SIFT or ORB with the Lowe ratio and RANSAC as models
    of that task;
  - warping and compositing on a canvas in `core/geometry.py` (plain arrays) or `ops/` (if they use
    the registration result);
  - `usecases/canvas/` with `vision-toolkit canvas`.
- **New domains** (depth, OCR, tracking; detection arrived with `anonymize`): a task module plus a
  model, as in the table above.
- **Graphical app.** Its skeleton is designed separately. Its review section reads and writes only
  the `core.review` contract, so it works for every queue without importing a use case
  ([anonymize](usecases/anonymize.md#the-review-section-of-the-app)).
- **Pose tracking between frames** (MediaPipe VIDEO mode, against the 7/10 pointing): an option of
  `PoseLandmarker`, or a smoothing op on a sequence of `Pose`, so it works for any pose model.

## When to split into a workspace

Today a single distribution is enough. The code is about a thousand lines with one author, and the
pip extras already keep heavy backends optional. Split into a uv workspace, as in pdf-toolkit, when
one of these happens:
- a backend's dependencies conflict with another one (for example torch and mediapipe pinning
  incompatible versions), so they need separate installs;
- a layer needs its own release (for example `core` + `tasks` used by another project);
- installing or testing one part slows down the others.

The layer rules already follow the future package boundaries:
- `packages/vision-core`: `core` + `tasks`;
- `packages/vision-models-<backend>`: one package per backend, registered through the
  `vision_toolkit.models` entry points, which already exist;
- `packages/vision-ops`;
- `apps/vision-toolkit`: `cli` + `usecases`.

Because use cases reach models only by name, the split is a move of files, not a redesign.

## Decisions

- [0001](decisions/0001-umbrella-toolkit.md): one umbrella toolkit, named vision-toolkit
- [0002](decisions/0002-single-package.md): one distribution with subpackages, workspace later
- [0003](decisions/0003-model-layer.md): tasks, models, catalog and ops
- [0004](decisions/0004-english.md): English everywhere
- [0005](decisions/0005-anonymized-frames.md): frames leave the machine only through a person's review
- [0006](decisions/0006-data-names.md): English names for the local data, one scheme for data/ and output/
- [0007](decisions/0007-apache-2.0-licence.md): Apache-2.0 licence

## Fixes

Every correction to the code has a note in [fixes/](fixes/README.md), written in the same commit as
the fix: what went wrong, the cause, what changed and how it was verified. The README there also
records where the corrections made before the notes are written.
