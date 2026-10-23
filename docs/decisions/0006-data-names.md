# 0006: English names for the local data, one scheme for data/ and output/

Status: accepted (2026-10-23). Replaces the point on data file names in [0004](0004-english.md).

## Context

[0004](0004-english.md) kept the names of the local data files as they were: they are private,
outside git, and other tools might refer to them. Those names came from three sources:
- the recordings kept the names Teams gives them, with spaces
  (`Riunione in General-<date>_<time>-Registrazione della riunione.mp4`), or a shortened copy
  (`Riunione_in_General-<date>_<time>.mp4`);
- other files were named by hand in Italian (`lezione_prova_cit_censured.png`, `calibrazione/`,
  `sintetico.mp4`, `frame_aggiuntivi_<date>_<time>/`);
- frames were stored under `data/video/`.

One calibration crop was named after the lecturer, and a committed notebook cited it. Each output
folder takes the name of its input, so `output/` carried the same names. No tool outside the
repository refers to these paths: the vault receives copies of the snapshots and of
`pointing.jsonl`.

## Decision

- Names in `data/` are in English and in lower case, with `_` between words and no spaces. They say
  what a file holds, never who appears in it.
- Folders by kind:
  - `data/video/`: the recordings, `lecture_<YYYYMMDD>_<HHMMSS>.mp4`, with the date and time at
    which Teams started the recording, as in its own name. `synthetic.mp4` is the synthetic video.
    `moving_camera/` keeps apart the recordings with a camera moved by hand;
  - `data/frame/`: single frames. `sample_*.png` is the sample frame censored by hand and the images
    derived from it. `censored/01.png`–`36.png` are the player screenshots with the face blurred,
    in time order. `lecture_<YYYYMMDD>_<HHMMSS>_extra/` holds frames saved by hand from a recording;
  - `data/calibration/`: labelled crops. `sample_arm.png` is the arm cut from the sample frame. The
    arm crops from the screenshots are named by gesture (`arm_pointing_<n>`,
    `arm_writing_on_board_<n>`, ...).
- An output folder takes the name of its input, `output/<use case>/<input name>/`, as the commands
  already do. A run kept for comparison gets a suffix on the use case folder (`output/board_v5/`),
  not on the input name.
- A new recording is renamed when it is added to `data/`.

## Consequences

- `data/` and `output/` were renamed on 2026-10-23. The review queues kept every decision: the extra
  frame's queue had its item names changed with its files. A second copy of the 6 October lecture
  at 8:45, identical byte for byte, was deleted.
- The notebooks, `tests/usecases/test_board_synthetic.py` and the documents use the new names. The
  scripts kept in `output/` with earlier runs point to the new paths. Their logs are records and
  were not changed.
- The board output format is unchanged: only folder names changed.
- The commits that cite the crop named after the lecturer cite `data/calibration/sample_arm.png`
  instead. The path is replaced in the notebook of every commit by the history rewrite before the
  first push.
