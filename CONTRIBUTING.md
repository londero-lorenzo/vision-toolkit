# Contributing

Contributions are welcome. A few rules matter more than usual here, because the use cases process
videos of real people.

1. **Never share images or videos of real people.** Not in issues, not in pull requests, not in test
   fixtures: no lecture frames, no screenshots of a video, no faces, even blurred. Reproduce problems
   with synthetic data (`tests/fakes.py`). If a bug only shows on a real video, describe the video
   instead: resolution, frame rate, length, whether the camera moves, the light, where the people are.
2. **Images leave the machine only through a review.** Code that publishes or uploads an image
   derived from a real video, or hands it to an AI service, goes through a review queue
   (`core.review`, [decision 0005](docs/decisions/0005-anonymized-frames.md)). No other networking
   code: the model downloads in `models/catalog.py` are the only exception.
3. **Layers.** Each layer imports only the layers below it, and use cases reach models only by name,
   through the catalog ([docs/architecture.md](docs/architecture.md)). `tests/test_layers.py`
   enforces it, and a new module goes into the module map of the architecture document.
4. **Licences.** Permissive dependencies only, and models whose licence allows any use
   ([docs/licensing.md](docs/licensing.md)). `python scripts/check_licenses.py` must pass.
5. **Tests.** `pytest tests` must pass. New behaviour comes with tests on synthetic data.
6. **Commits.** Conventional commits in English, with a scope (`feat(models): ...`). Every correction
   to the code gets a note in `docs/fixes/`, in the same commit. Larger choices get a record in
   `docs/decisions/`.
7. **Output formats.** The board outputs are read by other tools and pinned by
   `tests/usecases/test_board_pipeline.py`. A change to them is discussed in an issue first.

Setup, tests and notebooks: see "Setup" in the [README](README.md).

## Adding a model

A model that an existing class already handles needs only a `ModelSpec` in `models/catalog.py`. A new
kind of model needs a class in `models/<backend>/` that satisfies the task's Protocol. Either way, its
licence and source go in the table of [docs/licensing.md](docs/licensing.md). A model under a copyleft
or non-commercial licence goes in a separate package that declares the `vision_toolkit.models` entry
point. See "models and the catalog" in [docs/architecture.md](docs/architecture.md).

## Adding a use case

A use case lives in `usecases/<name>/` and reaches models only through the catalog. See "usecases" in
[docs/architecture.md](docs/architecture.md), and add a page in `docs/usecases/`.

## Licence of contributions

Contributions are licensed under the Apache License 2.0 (section 5 of the licence), without a CLA.
Contributors keep their copyright. See [docs/licensing.md](docs/licensing.md).
