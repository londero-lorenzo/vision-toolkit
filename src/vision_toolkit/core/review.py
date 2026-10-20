"""
Review queue: images waiting for a person's decision, kept as plain files.

    <root>/
      queue.jsonl              one line per addition: name, group, question, reasons, boxes, sha256, created
      todo/<group>/<name>.png  waiting for a decision; the person may edit the file in place
      approved/<name>.png      released
      rejected/<name>.png      discarded

The folder is the state: a decision moves the file, by hand in a file manager or
from an app. An approved item whose hash differs from the recorded one was edited.
Producers only add to todo/: they never touch approved/ or rejected/, and they do
not replace an item in todo/ that the person has already edited.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STATES = ("todo", "approved", "rejected", "missing")


@dataclass
class Item:
    name: str
    group: str
    question: str
    reasons: list
    boxes: list         # [x0, y0, x1, y1] areas to point out, drawn over the image, never into it
    state: str          # "todo", "approved", "rejected", or "missing" when the file was deleted
    path: Path | None
    edited: bool        # the file differs from the one that was added


class ReviewQueue:
    def __init__(self, root):
        self.root = Path(root)
        self.index = self.root / "queue.jsonl"

    def add(self, image, name, group, question, reasons=(), boxes=()):
        """
        Put `image` (a BGR array or the path of an image file) in todo/<group>/<name>.png.

        Returns False, writing nothing, when `name` is already decided or is in todo/
        with the person's edits. An untouched item in todo/ is replaced, also when its
        group changes, so a producer can run again with better settings.
        """
        item = self.item(name)
        if item is not None:
            if item.state in ("approved", "rejected") or (item.state == "todo" and item.edited):
                return False
            if item.path is not None:
                item.path.unlink()
        data = _png_bytes(image)
        path = self.root / "todo" / group / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record = {"name": name, "group": group, "question": question, "reasons": list(reasons),
                  "boxes": [[int(v) for v in box] for box in boxes],
                  "sha256": hashlib.sha256(data).hexdigest(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.index, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True

    def item(self, name):
        """The item `name` with its state read from the folders, None if it was never added."""
        record = self._records().get(name)
        return None if record is None else self._item(record)

    def items(self):
        """Every item by name; when a name was added more than once, the latest addition wins."""
        return {name: self._item(record) for name, record in self._records().items()}

    def decided(self, name):
        """True if `name` is approved or rejected."""
        item = self.item(name)
        return item is not None and item.state in ("approved", "rejected")

    def status(self):
        """Counts per group, {group: {"todo", "approved", "edited", "rejected", "missing"}}; edited ones are approved too."""
        counts = {}
        for item in self.items().values():
            c = counts.setdefault(item.group, dict.fromkeys(("todo", "approved", "edited", "rejected", "missing"), 0))
            c[item.state] += 1
            c["edited"] += item.state == "approved" and item.edited
        return counts

    def _records(self):
        records = {}
        if self.index.exists():
            for line in self.index.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    records[record["name"]] = record
        return records

    def _item(self, record):
        name, group = record["name"], record["group"]
        # A file left in two folders by mistake counts as the more cautious state
        candidates = [("rejected", self.root / "rejected" / f"{name}.png"),
                      ("approved", self.root / "approved" / f"{name}.png"),
                      ("todo", self.root / "todo" / group / f"{name}.png")]
        state, path = next(((s, p) for s, p in candidates if p.exists()), ("missing", None))
        edited = path is not None and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        return Item(name, group, record["question"], record["reasons"], record["boxes"], state, path, edited)


def _png_bytes(image):
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() == ".png":
            return path.read_bytes()
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode the image as PNG")
    return buffer.tobytes()
