"""Fail if any installed distribution is under a GPL-family licence (GPL, LGPL, AGPL).

vision-toolkit is Apache-2.0 and must stay usable in closed-source products: only permissive
dependencies (MIT, BSD, Apache-2.0, PSF, ISC, MPL-2.0 file-level) are accepted (decision 0007).
The LGPL libraries bundled inside the OpenCV wheels (FFmpeg, Qt) do not show in the metadata and
are listed by hand in docs/licensing.md. Run with ``python scripts/check_licenses.py`` in an
environment with every extra installed (``pip install -e ".[dev,mediapipe]"``).
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import distributions

COPYLEFT = re.compile(r"\b(A|L)?GPL\b|GNU (Affero |Lesser )?General Public", re.I)
OURS = {"vision-toolkit"}


def licence_text(dist) -> str:
    meta = dist.metadata
    parts = [meta.get("License-Expression") or "", (meta.get("License") or "")[:300]]
    parts += [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return " | ".join(p for p in parts if p)


def main() -> int:
    bad = []
    for dist in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        if name.lower() in OURS:
            continue
        text = licence_text(dist)
        if COPYLEFT.search(text):
            bad.append(f"{name}: {text}")
    for line in bad:
        print(f"COPYLEFT {line}")
    print(f"licence check: {len(bad)} copyleft distribution(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
