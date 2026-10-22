# 0004: English everywhere

Status: accepted (2026-10-18).

## Context

The repository was written in Italian: README, docstrings, comments, CLI messages, notebooks and
CLAUDE.md. Only the commits were in English. As a general toolkit it should be understandable by
anyone.

## Decision

- English for code, identifiers, docstrings, comments, CLI help and messages, notebooks, tests, docs
  and CLAUDE.md. Commits stay in English, as conventional commits with a scope.
- The names of the local data files (`data/frame/lezione_prova_cit_censured.png`, ...) are not
  renamed. They are private, outside git, and other tools may refer to them.
- Conversations with the maintainer stay in Italian. Only the repository content is in English.

## Consequences

- The command `lavagna` became `vision-toolkit board`. The notebooks were renamed
  (`01_single_frame.ipynb`, `02_temporal_reconstruction.ipynb`).
- The board output format did not change: its names were already in English (`board_*.png`,
  `pointing.jsonl` keys).
