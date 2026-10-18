# 0003: Models behind task interfaces, created by name from a catalog

Status: accepted (2026-10-18).

## Context

The board use case uses two MediaPipe models, PoseLandmarker and DeepLab V3. The old code wove them
into `PersonSegmenter`:
- MediaPipe options built inline;
- landmark indices as constants (`INDEX = {"left": 19, ...}`);
- the PASCAL VOC class `DEEPLAB_PERSON = 15`;
- `--refine` as a switch for exactly one model.

Lorenzo's requirement: DeepLab V3 must not preclude other models. The repository must host many
specialised models, for the same task (another segmenter next to DeepLab) and for other domains
(detection, depth, OCR, tracking), on other backends (ONNX Runtime, PyTorch).

## Decision

Four parts, each in its own layer ([architecture](../architecture.md)):

1. **`tasks/`: one interface per domain.** A `Protocol` plus result dataclasses in plain NumPy:
   `Segmenter` → `Segmentation(category_mask, labels, confidence)`, `PoseEstimator` → `list[Pose]`.
   Classes and keypoints are **names**: `seg.mask_of("person")`, `pose["left_index"]`. Keypoint names
   follow COCO and, for the points COCO lacks, MediaPipe. A COCO-style pose model therefore fits too.
2. **`models/<backend>/`: implementations.** The conversion from backend types happens here and only
   here (`read_mask` is the only place that calls MediaPipe's `numpy_view()`). One class can serve
   several models: `ImageSegmenter` handles both DeepLab V3 and selfie multiclass.
3. **`models/catalog.py`: one registry for creating and downloading.**
   - A `ModelSpec` gives the name, task, backend, lazy factory, files with their URLs, pip extra and
     default options.
   - `create(name, task=...)` checks the task and the files, then imports the factory. Failures raise
     `CatalogError` with the fix (`vision-toolkit models download NAME`, `pip install -e ".[extra]"`).
   - External packages add specs through the entry-point group `vision_toolkit.models`.
   - It replaces the hard-coded list of `scripts/download_models.py`.
4. **`ops/`: model-agnostic algorithms on task results:** `segment_on_crop` (any segmenter),
   `draw_limbs`, `pointing_hand`. They depend on interfaces only and are tested with fake models.

Use cases choose models by name, with defaults and CLI options (`--pose-model`, `--refine-model`).
Composition is plain Python, with no generic pipeline framework.

## Consequences

- Swapping a model is a CLI option or a parameter. Adding one is a class and a spec, or only a spec
  if an existing class handles it. Nothing downstream changes.
- Changing a task's result type is the expensive change: it reaches every model and op of that task,
  so it needs a decision record.
- `selfie-multiclass`, downloaded but unused before, is now usable
  (`--refine-model selfie-multiclass --refine-labels hair,body-skin,face-skin,clothes`).

## Not done (yet)

- No ONNX or PyTorch backend: added when a use case needs a model that MediaPipe does not have.
- No entry points for use cases (as pdf-toolkit has for its modules): added when an external package
  needs to ship one.
