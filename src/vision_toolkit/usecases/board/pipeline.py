import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from vision_toolkit.core.geometry import load_quad, rectify_homography, save_quad
from vision_toolkit.core.io import format_time, iter_frames, median_frame
from vision_toolkit.core.temporal import PlaneMedian
from vision_toolkit.usecases.board.detect import detect_board
from vision_toolkit.usecases.board.person_mask import DEFAULT_POSE_MODEL, PersonMasker


class BoardNotFoundError(RuntimeError):
    pass


@dataclass
class BoardOptions:
    step: float = 1.0               # seconds between two analysed frames
    start: float = 0.0
    end: float | None = None
    crop: tuple | None = None       # (x, y, w, h) of the video area in a screen recording
    quad: Path | None = None        # JSON with the 4 board corners; detected when None
    window: int = 7                 # samples in the temporal median
    min_change: float = 0.004       # fraction of changed pixels that triggers a new snapshot
    pose_model: str = DEFAULT_POSE_MODEL
    refine_model: str | None = None
    refine_labels: tuple = ("person",)
    models_dir: Path | None = None


def run(video, out_dir, options=None):
    """
    Remove the lecturer from the board in a lecture video.

    Writes to out_dir/<video name>/:
    - board_<hh>h<mm>m<ss>s.png: clean board snapshot, every time its content changes
    - board_final.png: the board at the end of the video
    - median.png, quad.json: median frame and detected corners (only without options.quad)
    - pointing.jsonl: one line per analysed frame with the lecturer in view

    Returns the output folder.
    """
    options = options or BoardOptions()
    video = Path(video)
    out = Path(out_dir) / video.stem
    out.mkdir(parents=True, exist_ok=True)

    if options.quad:
        quad = load_quad(options.quad)
    else:
        background = median_frame(video, crop=options.crop)
        cv2.imwrite(str(out / "median.png"), background)
        quad = detect_board(background)
        if quad is None:
            raise BoardNotFoundError("Board not found: pass --quad with the corners set by hand.")
        save_quad(out / "quad.json", quad)

    H, size = rectify_homography(quad)
    board = PlaneMedian(H, size, window=options.window)
    masker = PersonMasker(options.pose_model, options.refine_model, options.refine_labels,
                          models_dir=options.models_dir)
    prev_change = None

    with masker, open(out / "pointing.jsonl", "w", encoding="utf-8") as log:
        snapshot = None
        for t, frame in iter_frames(video, options.step, options.start, options.end, options.crop):
            person = masker(frame)
            occluded = board.update(frame, person.mask)

            # Snapshot when the board has changed enough and has settled
            change = board.change_since_snapshot()
            if (change >= options.min_change and prev_change is not None
                    and abs(change - prev_change) < options.min_change / 4):
                snapshot = f"board_{format_time(t)}.png"
                cv2.imwrite(str(out / snapshot), board.take_snapshot())
                change = 0.0
            prev_change = change

            board_xy = board.to_plane(person.pointer)
            if person.pose is not None:
                log.write(json.dumps({
                    "t": round(t, 2),
                    "frame_xy": person.pointer,
                    "board_xy": board_xy,
                    "board_xy_norm": [round(board_xy[0] / size[0], 4), round(board_xy[1] / size[1], 4)] if board_xy else None,
                    "arm_angle": round(person.arm_angle, 1),
                    "occluded_fraction": round(float(occluded.mean()), 4),
                    "snapshot": snapshot,
                }) + "\n")

    cv2.imwrite(str(out / "board_final.png"), board.image())
    return out
