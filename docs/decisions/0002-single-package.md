# 0002: One distribution with subpackages; a workspace only when needed

Status: accepted (2026-10-18).

## Context

`pdf-toolkit` is a uv workspace with several packages (`pdf-core`, `pdf-redaction`, `pdf-fonts`, an
app). vision-toolkit could start the same way. Today, though, the code is about a thousand lines, with
one author and one use case. The only heavy dependency is a model backend (MediaPipe), which pulls
TensorFlow Lite, protobuf and `opencv-contrib-python`.

## Decision

- **One distribution, `vision-toolkit`, with subpackages per layer:** `core`, `tasks`, `models`,
  `ops`, `usecases`, `cli`.
- **Heavy dependencies go behind pip extras, one per model backend:** `[mediapipe]` today, later
  `[onnx]`, `[torch]`. The core depends only on OpenCV and NumPy. Backends are imported lazily by the
  model catalog, so a missing extra breaks only the models that need it.
- **The layers already follow the future package boundaries.** `tests/test_layers.py` enforces the
  rules between them, so a split stays mechanical.

## When to split

As described in [architecture](../architecture.md#when-to-split-into-a-workspace):
- a backend whose dependencies conflict with another one;
- a layer that needs its own release;
- install or test times that slow down unrelated work.

The model entry-point group `vision_toolkit.models` already lets a backend live in another package.

## Consequences

- One `pip install -e ".[dev,mediapipe]"`, one test run, one version number.
- `requirements.txt` is the full freeze of the development environment, not the dependency list.
  The dependencies are in `pyproject.toml`.
