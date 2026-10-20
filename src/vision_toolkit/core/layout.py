"""
Layout of a screen recording: where the video sits in it and where the audience
starts. A person confirms it once; every video with the same layout shares it.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Setup:
    crop: tuple | None          # (x, y, w, h) of the video area in the recording
    redact_below: int | None    # rows from here down, in the cropped frame, are blacked out (the audience)
    frame_size: tuple           # (w, h) of the videos the setup is for
    confirmed: bool = False


def load_setup(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crop = data.get("crop")
    return Setup(tuple(crop) if crop else None, data.get("redact_below"), tuple(data["frame_size"]),
                 bool(data.get("confirmed", False)))


def save_setup(path, setup):
    data = {"crop": list(setup.crop) if setup.crop else None, "redact_below": setup.redact_below,
            "frame_size": list(setup.frame_size), "confirmed": setup.confirmed}
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
