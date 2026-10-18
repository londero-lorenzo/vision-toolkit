# Migration guide: from image-segmentation to vision-toolkit

On 2026-10-18 the repository `image-segmentation` (package `image_segmentation`, command `lavagna`)
became **vision-toolkit** (package `vision_toolkit`, command `vision-toolkit`). The board use case
works as before and writes the same files. What changed is the structure, the names, the commands and
the language. The reasons are in decisions [0001](decisions/0001-umbrella-toolkit.md) to
[0004](decisions/0004-english.md).

## 1. Reinstall

In the existing environment:

```
.env\Scripts\python -m pip uninstall -y image-segmentation
.env\Scripts\python -m pip install -e ".[dev,mediapipe]"
```

MediaPipe is now an extra (`[mediapipe]`): without it the toolkit installs, but the MediaPipe models
cannot be created. The model files already in `models/` keep their names and are reused. The variable
`VISION_TOOLKIT_MODELS` can point to another folder.

**After renaming the folder** (`image-segmentation` → `vision-toolkit`), recreate the environment. A
virtual environment stores absolute paths: the editable install points to the old `src/`, and the
`.exe` launchers in `.env\Scripts` (pip, pytest, jupyter, vision-toolkit) point to the old Python.

```
rm -rf .env                      # Git Bash; in cmd: rmdir /s /q .env
py -m venv .env
.env\Scripts\python -m pip install -e ".[dev,mediapipe]"
```

## 2. Commands

| Before | After |
|---|---|
| `lavagna VIDEO [options]` | `vision-toolkit board VIDEO [options]` |
| `--refine` | `--refine-model deeplab-v3` |
| — | `--pose-model NAME` (default `mediapipe-pose-heavy`), `--refine-labels person` |
| `python scripts/download_models.py` | `vision-toolkit models download [NAMES]` |
| — | `vision-toolkit models list [--task TASK]` |
| `pip install -e ".[dev]"` | `pip install -e ".[dev,mediapipe]"` |

The other options are unchanged: `--out --step --start --end --crop --quad --window --min-change`.
Messages are in English: "Lavagna non trovata" became "Board not found: pass --quad with the
corners set by hand."

## 3. Outputs (what the vault reads)

Unchanged, in `output/<video name>/`:
- `board_<hh>h<mm>m<ss>s.png`, `board_final.png`, `median.png`;
- `quad.json` with the key `quad`;
- `pointing.jsonl` with the keys `t, frame_xy, board_xy, board_xy_norm, arm_angle,
  occluded_fraction, snapshot`.

Checked on `data/video/sintetico.mp4`. With the default options, `vision-toolkit board` writes files
byte-identical to those of the old `lavagna`, and the same holds for `--refine-model deeplab-v3`
against `--refine`.

One deliberate change, only with `--refine-model`. The old `_deeplab_on_crop` used the last row and
column of the person box as an exclusive bound, so the crop lost them. `ops.crop_refine.bounding_box`
now includes them. On the synthetic video:
- `occluded_fraction` moves by at most 0.0001 on 5 lines out of 10;
- 0.002% of the pixels of `board_final.png` change;
- pointed spots and snapshots do not change.

`tests/usecases/test_board_pipeline.py` now pins the file names and the `pointing.jsonl` keys.

## 4. Code: where everything went

| Before (`src/image_segmentation/`) | After (`src/vision_toolkit/`) |
|---|---|
| `board.py` `detect_board` | `usecases/board/detect.py` `detect_board` |
| `board.py` `order_corners`, `rectify_homography`, `save_quad`, `load_quad` | `core/geometry.py`, same names |
| `board.py` `board_mask` | `core/geometry.py` `polygon_mask` |
| `person.py` `MODELS_DIR` | `core/paths.py` `models_dir()`, a function read at call time, overridable by `VISION_TOOLKIT_MODELS` |
| `person.py` `PersonSegmenter(refine=False, dilate=15, models_dir=...)` | `usecases/board/person_mask.py` `PersonMasker(pose_model=..., refine_model=None, refine_labels=("person",), dilate=15, models_dir=None)` |
| `person.py` `PersonResult(mask, pointer, arm_angle, landmarks)` | `PersonMask(mask, pose, pointer, arm_angle)`: `landmarks` (MediaPipe list) became `pose` (`tasks.pose.Pose`, named keypoints) |
| `person.py` MediaPipe setup of PoseLandmarker | `models/mediapipe/pose_landmarker.py` `PoseLandmarker`, catalog name `mediapipe-pose-heavy` |
| `person.py` MediaPipe setup of DeepLab | `models/mediapipe/image_segmenter.py` `ImageSegmenter`, catalog name `deeplab-v3` |
| `person.py` `_mp_image` | `models/mediapipe/_common.py` `to_mp_image` (+ `read_mask` for every mask) |
| `person.py` `_draw_arms`, `ARMS` (indices) | `ops/skeleton.py` `draw_limbs`, `ARMS` (keypoint names) |
| `person.py` `_pointing_hand` | `ops/pointing.py` `pointing_hand` → `Pointing(tip, arm_angle, side)` |
| `person.py` `_deeplab_on_crop` | `ops/crop_refine.py` `segment_on_crop(segmenter, image, region_mask, labels)`, for any segmenter |
| `person.py` `SHOULDER`, `INDEX`, `HIP`, `DEEPLAB_PERSON = 15` | names: `pose["left_shoulder"]`, `seg.mask_of("person")` |
| `reconstruction.py` `BoardReconstructor` | `core/temporal.py` `PlaneMedian` (same constructor) |
| `BoardReconstructor.board` | `PlaneMedian.background` |
| `BoardReconstructor.to_board(point)` | `PlaneMedian.to_plane(point)`, also `core.geometry.to_plane(point, H, size)` |
| `video.py` `iter_frames`, `apply_crop`, `median_frame`, `format_time` | `core/io.py`, same names |
| `main.py` `main` | `usecases/board/pipeline.py` `run(video, out_dir, BoardOptions(...))`, `usecases/board/cli.py`, `cli.py` |
| `main.py` `parse_crop` | `usecases/board/cli.py` `parse_crop` |
| `scripts/download_models.py` `MODELS` | `models/catalog.py` `BUILTIN` specs + `download()` |

Other files:

| Before | After |
|---|---|
| `tests/test_synthetic.py` `test_reconstruction_removes_teacher` | `tests/usecases/test_board_synthetic.py` `test_reconstruction_removes_lecturer` |
| `notebooks/01_frame_singolo.ipynb` | `notebooks/01_single_frame.ipynb` |
| `notebooks/02_ricostruzione_temporale.ipynb` | `notebooks/02_temporal_reconstruction.ipynb` |
| `README.md` (board) | `docs/usecases/board.md`; the README now describes the toolkit |

New tests: `tests/core/`, `tests/models/`, `tests/ops/`, `tests/usecases/test_board_pipeline.py`
(synthetic room with a fake pose model) and `tests/test_layers.py` (import rules).

## 5. Code that used the old API

```python
# before
from image_segmentation.person import PersonSegmenter
with PersonSegmenter(refine=True) as segment:
    person = segment(frame)            # person.mask, person.pointer, person.landmarks[19]

# after
from vision_toolkit.usecases.board.person_mask import PersonMasker
with PersonMasker(refine_model="deeplab-v3") as masker:
    person = masker(frame)             # person.mask, person.pointer, person.pose["left_index"]
```

```python
# before: a MediaPipe segmenter built by hand, class 15 = person
# after: any segmenter from the catalog, classes by name
from vision_toolkit.models import catalog
with catalog.create("deeplab-v3", confidence=True) as deeplab:
    seg = deeplab(frame)
    person = seg.mask_of("person")             # uint8 0/255
    confidence = seg.confidence_of("person")   # float32 [0, 1]
```

## 6. Language

Code, docstrings, comments, CLI, notebooks, docs and CLAUDE.md are in English
([0004](decisions/0004-english.md)). Commits were already in English. The names of the local files in
`data/` (for example `lezione_prova_cit_censured.png`) are Lorenzo's private data and stay as they are.
