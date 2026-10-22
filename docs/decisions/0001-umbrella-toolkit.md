# 0001: One umbrella toolkit for image and video processing, named vision-toolkit

Status: accepted (2026-10-18).

## Context

The repository was born for one task: remove the lecturer from the board in lecture videos, so that
clean snapshots can be read next to the lecture transcripts in the study vault. Other image
studies use the same building blocks. `image-stitching` merges DrawNote canvas exports
with SIFT and homographies, and more will follow.

There were two options. The repository could hold only segmentation, with each other study in its own
repository. Or it could be one toolkit with a shared core and one module per kind of processing, as
`pdf-toolkit` does for PDFs.

"Segmentation only" does not fit the code that exists. Most of the board use case is not
segmentation:
- plane geometry (finding and rectifying the board);
- temporal reconstruction (median of the uncovered pixels);
- pose (the pointed spot).

Stitching shares the same core: homographies and a common plane. Separate repositories would copy
that core or depend on each other.

## Decision

- **One toolkit** for image and video processing. Reusable layers (core, tasks, models, ops) sit
  below concrete use cases. The board is the first use case.
- **Name: vision-toolkit**, package `vision_toolkit`, command `vision-toolkit`, following the
  `pdf-toolkit` naming. The old name `image-segmentation` described one technique, not the repository.
- **`image-stitching` is absorbed in a second step**, as the `canvas` use case
  ([architecture](../architecture.md#where-future-work-lands)). Until then the old repository stays
  as it is, with its history.
- **Out of scope:** transcripts, Whisper and calls to a language model. The toolkit produces artifacts: clean
  snapshots, `pointing.jsonl`, and later a "moment → snapshot + pointed area" query. Aligning them with
  the transcripts stays on the vault side.

## Consequences

- The README describes the toolkit. Each use case has its own page in `docs/usecases/`.
- The folder and the assistant's memory folder are renamed after this change. The environment must be
  recreated after the rename ([migration guide](../migration-guide.md#1-reinstall)).
