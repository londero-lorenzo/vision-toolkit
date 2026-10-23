# 0007: Apache-2.0 licence, with the policy of pdf-toolkit

Status: accepted (2026-10-23).

## Context

The repository has no licence yet, so nobody else may use, copy or change the code. It goes public
on GitHub after the history rewrite (session 1 of the [roadmap](../roadmap.md)) and is meant to
accept contributions from the public. Every commit so far has a single author, so the licence can
be chosen now without anyone else's consent.

The sibling project pdf-toolkit is under Apache-2.0
([its decision 0005](https://github.com/londero-lorenzo/pdf-toolkit/blob/main/docs/decisions/0005-apache-2.0-licence.md)).
It has the same needs:
- a public project, open to contributions without paperwork;
- separate packages that plug in through the entry points under their own licence. Here the
  entry-point group is `vision_toolkit.models`;
- use by companies, whose legal review looks for a patent grant.

Two needs are specific to a vision toolkit:
- **Models have licences of their own**, separate from the code that runs them. The catalog
  downloads them and the repository does not ship them. Well-known detectors are under AGPL-3.0
  (Ultralytics YOLO), and some face models allow only non-commercial research.
- **The OpenCV wheels bundle LGPL libraries** as separate files: FFmpeg (LGPL-2.1) on every
  platform, and Qt 5 (LGPL-3.0) in the non-headless Linux and macOS wheels. The package metadata
  shows only Apache-2.0.

image-stitching, to be absorbed as the canvas use case (session 5), is public on GitHub without a
licence and has the same single author.

## Options considered

| | Separate packages under their own licence | Contributions | Fit |
|---|---|---|---|
| **Apache-2.0** (chosen) | yes, through the entry points | same licence, by section 5 of the licence | good, plus a patent grant; the same as pdf-toolkit |
| **MIT** | yes, through the entry points | same licence, by a note in `CONTRIBUTING.md` | good, no patent grant |
| **GPL-3.0 / AGPL-3.0** | GPL-compatible ones only | same licence | would let AGPL models into the repository, but many companies avoid it |

## Decision

### Licence

The repository is under the **Apache License 2.0**. `LICENSE` holds the unmodified text from
apache.org, `NOTICE` the copyright line, and `pyproject.toml` declares `license = "Apache-2.0"`.
The licence covers every published commit, including those made before `LICENSE` was added.

Apache-2.0 rather than MIT, for the reasons given in pdf-toolkit:
- **Patents (section 3).** Every contributor grants a licence on the patents that the
  contribution uses, and loses it if they sue over patents in the work.
- **Contributions (section 5).** Anything submitted for inclusion is under Apache-2.0 unless stated
  otherwise. The "inbound = outbound" rule is in the licence itself, so no CLA is needed.

### Dependencies

- Permissive licences (MIT, BSD, Apache-2.0, PSF, ISC, MPL-2.0) are accepted.
  `scripts/check_licenses.py` fails on any GPL, LGPL or AGPL distribution in the environment.
- **LGPL** libraries are allowed when they are used unmodified, as separate files loaded
  dynamically, so that the user can replace them. Each one is listed in
  [docs/licensing.md](../licensing.md). The ones inside the OpenCV wheels are listed there by hand,
  because the check reads package metadata.
- **GPL and AGPL** libraries only in separate optional packages that register models through the
  `vision_toolkit.models` entry points.

### Models

- The catalog lists only models whose licence allows any use, commercial use included. Each one
  gets its licence and source in [docs/licensing.md](../licensing.md), in the change that adds it.
- Models under copyleft or non-commercial licences go in separate packages, through the entry
  points, under their own terms.
- Model files are downloaded into `models/` and never committed.

### Contributions

Contributions are accepted under Apache-2.0 (section 5), without a CLA. Contributors keep their
copyright. `CONTRIBUTING.md` holds the rules, `SECURITY.md` how to report a vulnerability or a
privacy leak.

### image-stitching

image-stitching has the same author, so its code enters under Apache-2.0 when it is absorbed.

### Data

The licence covers the code and the documents. The lecture videos and frames are not part of the
repository ([0005](0005-anonymized-frames.md), [0006](0006-data-names.md)), and no licence is
granted on them.

## Consequences

- `LICENSE`, `NOTICE`, `pyproject.toml`, `README.md`, `CLAUDE.md`, `docs/licensing.md`,
  `docs/architecture.md` (adding a model), `CONTRIBUTING.md`, `SECURITY.md`, the issue templates and
  `scripts/check_licenses.py` come with this record.
- `NOTICE` names the copyright holder. A legal notice is the one place where the impersonal form
  gives way.
- `pyproject.toml` requires setuptools 77 or later, the first that reads a licence expression
  (PEP 639).
- Redistributors keep `NOTICE` and mark the files they changed (section 4).
- The repository has no CI yet: `scripts/check_licenses.py` is run by hand until it has a workflow.
- This record is not legal advice.
