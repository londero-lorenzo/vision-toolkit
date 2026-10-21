import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision_toolkit.core.framing import Framing, framing_segments
from vision_toolkit.core.geometry import load_quad, rectify_homography, save_quad
from vision_toolkit.core.io import FrameReader, format_time, iter_frames, median_frame
from vision_toolkit.core.layout import load_setup
from vision_toolkit.core.redact import black_out_below
from vision_toolkit.core.temporal import PlaneMedian
from vision_toolkit.usecases.board.chalk import change_threshold, stroke_contrast
from vision_toolkit.usecases.board.detect import find_board
from vision_toolkit.usecases.board.person_mask import DEFAULT_POSE_MODEL, FOREGROUND, POSE, TRACKED, PersonMasker


class BoardNotFoundError(RuntimeError):
    pass


class SetupError(RuntimeError):
    """The setup.json given cannot be used for this video."""


@dataclass
class BoardOptions:
    step: float = 2.0                       # seconds between two analysed frames
    start: float = 0.0
    end: float | None = None
    crop: tuple | None = None               # (x, y, w, h) of the video area in a screen recording
    redact_below: int | None = None         # audience line: rows from here down are blacked out before any model
    setup: Path | None = None               # confirmed setup.json (core.layout) for crop and redact_below
    quad: Path | None = None                # JSON with the 4 board corners; detected when None
    window: int = 21                        # samples in the temporal median
    min_seen: float = 0.25                  # fraction of the window a board pixel must be seen in to change
    min_change: float = 0.004               # fraction of changed pixels that triggers a new snapshot
    change_threshold: float | None = None   # levels a pixel must change by to count; None measures the chalk
    match_exposure: bool = True             # divide each frame by its exposure gain against the median
    framing_step: float | None = 15.0       # seconds between two frames of the framing check; None skips it
    min_segment: float = 60.0               # seconds: shorter framing segments are skipped
    pose_model: str = DEFAULT_POSE_MODEL
    pose_confidence: float | None = 0.1     # the pose model's min_confidence; None keeps the model's default
    fallback: bool = True                   # a person-shaped foreground blob when the pose finds nobody
    grow_mask: bool = True                  # add the foreground blobs touching the pose mask
    track: float = 20.0                     # seconds the lecturer is followed through the foreground once the pose lost her
    refine_model: str | None = None
    refine_labels: tuple = ("person",)
    models_dir: Path | None = None


@dataclass
class _Segment:
    framing: Framing
    background: np.ndarray  # median of the segment's frames, audience blacked out
    quad: np.ndarray
    H: np.ndarray | None = None


def run(video, out_dir, options=None):
    """
    Remove the lecturer from the board in a lecture video.

    Writes to out_dir/<video name>/:
    - board_<hh>h<mm>m<ss>s.png: clean board snapshot, every time its content changes,
      from the moment the median holds half a window of frames
    - board_final.png: the board at the end of the video
    - median.png, quad.json: median frame and detected corners (only without options.quad),
      of the longest framing segment
    - pointing.jsonl: one line per analysed frame where the pose found the lecturer
    - run.json: what the run measured and decided (framing segments, snapshot threshold,
      how the lecturer was found)

    Returns the output folder.
    """
    options = options or BoardOptions()
    video = Path(video)
    out = Path(out_dir) / video.stem
    out.mkdir(parents=True, exist_ok=True)
    crop, floor = _layout(video, options)

    # The board, and the stretches of video with one camera framing
    hand_quad = load_quad(options.quad) if options.quad else None
    whole = black_out_below(median_frame(video, crop=crop), floor)
    first_quad = hand_quad if hand_quad is not None else find_board(whole, floor)
    if first_quad is None:
        raise BoardNotFoundError("Board not found: pass --quad with the corners set by hand.")
    framings = _framings(video, whole.shape, first_quad, crop, options)
    long_enough = [f for f in framings if f.duration >= options.min_segment] or [max(framings, key=lambda f: f.duration)]
    segments, report = [], []
    for framing in framings:
        row = {"start": round(framing.start, 2), "end": round(framing.end, 2), "used": False,
               "skipped_because": None, "quad": None}
        report.append(row)
        if not any(framing is f for f in long_enough):
            row["skipped_because"] = "short"
            continue
        background = whole if len(framings) == 1 else black_out_below(
            median_frame(video, crop=crop, start=framing.start, end=framing.end), floor)
        quad = hand_quad if hand_quad is not None else find_board(background, floor)
        if quad is None:
            row["skipped_because"] = "board-not-found"
            continue
        row.update(used=True, quad=np.round(np.float64(quad), 1).tolist())
        segments.append(_Segment(framing, background, np.float32(quad)))
    if not segments:
        raise BoardNotFoundError("Board not found in any framing segment: pass --quad with the corners set by hand.")

    # One rectified plane for every segment: the board is the same, only the view changes
    main = max(segments, key=lambda s: s.framing.duration)
    _, size = rectify_homography(main.quad)
    w, h = size
    plane = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    for segment in segments:
        segment.H = cv2.getPerspectiveTransform(segment.quad, plane)
    if hand_quad is None:
        cv2.imwrite(str(out / "median.png"), main.background)
        save_quad(out / "quad.json", np.round(np.float64(main.quad), 1))

    pose_options = {"min_confidence": options.pose_confidence} if options.pose_confidence is not None else None
    masker = PersonMasker(options.pose_model, options.refine_model, options.refine_labels,
                          models_dir=options.models_dir, pose_options=pose_options,
                          fallback=options.fallback, grow=options.grow_mask)

    def lecturer(frame, segment, previous=None):
        return masker(black_out_below(frame, floor), segment.background, floor, previous)

    found = {POSE: 0, FOREGROUND: 0, TRACKED: 0, "none": 0}
    max_tracked = int(options.track / options.step)
    gains, snapshots = [], 0
    with masker:
        threshold, contrast = options.change_threshold, None
        if threshold is None:
            contrasts = _chalk_contrasts(video, crop, main, size, lecturer)
            threshold = change_threshold(contrasts)
            measured = [c for c in contrasts if c is not None]
            contrast = round(float(np.median(measured)), 1) if measured else None
        board = PlaneMedian(main.H, size, window=options.window, change_threshold=threshold,
                            min_samples=max(1, round(options.min_seen * options.window)))
        warmup = options.window // 2  # a median of the first few frames shows whatever they missed
        prev_change = None

        with open(out / "pointing.jsonl", "w", encoding="utf-8") as log:
            snapshot = None
            for segment in sorted(segments, key=lambda s: s.framing.start):
                board.H = segment.H
                if options.match_exposure:
                    board.exposure_reference = cv2.warpPerspective(segment.background, segment.H, size)
                previous, tracked = None, 0
                for t, frame in iter_frames(video, options.step, segment.framing.start, segment.framing.end, crop):
                    person = lecturer(frame, segment, previous if tracked < max_tracked else None)
                    found[person.source or "none"] += 1
                    tracked = tracked + 1 if person.source in (FOREGROUND, TRACKED) else 0
                    previous = person.mask if person.source else None
                    occluded = board.update(frame, person.mask)
                    gains.append(board.gain)

                    # Snapshot when the board has changed enough and has settled
                    change = board.change_since_snapshot()
                    if (board.index >= warmup and change >= options.min_change and prev_change is not None
                            and abs(change - prev_change) < options.min_change / 4):
                        snapshot = f"board_{format_time(t)}.png"
                        cv2.imwrite(str(out / snapshot), board.take_snapshot())
                        snapshots += 1
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
    summary = {
        "framings": report,
        "change_threshold": round(float(threshold), 1),
        "chalk_contrast": contrast,
        "frames": {"analysed": sum(found.values()), "lecturer_by_pose": found[POSE],
                   "lecturer_by_foreground": found[FOREGROUND], "lecturer_by_tracking": found[TRACKED],
                   "nobody": found["none"]},
        "exposure_gain": {f"p{p}": round(float(np.percentile(gains, p)), 3) for p in (5, 50, 95)} if gains else None,
        "snapshots": snapshots,
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out


def _layout(video, options):
    """(crop, audience line) from the options, then from the setup file."""
    crop, floor = options.crop, options.redact_below
    if options.setup:
        setup = load_setup(options.setup)
        if not setup.confirmed:
            raise SetupError(f"{options.setup} is not confirmed yet: check it and set confirmed to true.")
        with FrameReader(video) as reader:
            frame = reader.at(0)
        if frame is not None and tuple(setup.frame_size) != (frame.shape[1], frame.shape[0]):
            raise SetupError(f"{options.setup} is for {setup.frame_size[0]}x{setup.frame_size[1]} frames, "
                             f"this video has {frame.shape[1]}x{frame.shape[0]}")
        crop = crop if crop is not None else setup.crop
        floor = floor if floor is not None else setup.redact_below
    return crop, floor


def _framings(video, shape, quad, crop, options):
    """Framing segments of [start, end), aligned on the wall band above the board."""
    if not options.framing_step:
        with FrameReader(video) as reader:
            end = reader.duration if options.end is None else min(options.end, reader.duration)
        return [Framing(options.start, end, options.start)]
    return framing_segments(video, wall_band(shape, quad), options.framing_step, options.start, options.end, crop)


def wall_band(shape, quad, below=15):
    """
    uint8 (0/255) mask of what lies above the board's top side, down to `below` pixels
    under it: the wall, a light strip and the top of the board's frame. The lecturer
    seldom reaches it and the writing stays out, so it shows the camera's framing.
    """
    h, w = shape[:2]
    (x0, y0), (x1, y1) = quad[0], quad[1]
    top = y0 + (np.arange(w) - x0) * (y1 - y0) / (x1 - x0)
    return ((np.arange(h)[:, None] < top[None, :] + below) * 255).astype(np.uint8)


def _chalk_contrasts(video, crop, segment, size, lecturer, samples=12):
    """Stroke contrast on frames spread over a segment, with the lecturer left out."""
    contrasts = []
    with FrameReader(video, crop) as reader:
        for t in np.linspace(segment.framing.start, segment.framing.end, samples + 2)[1:-1]:
            frame = reader.at(float(t))
            if frame is None:
                continue
            person = lecturer(frame, segment)
            board = cv2.warpPerspective(frame, segment.H, size)
            hidden = cv2.warpPerspective(person.mask, segment.H, size, flags=cv2.INTER_NEAREST) > 0
            contrasts.append(stroke_contrast(board, hidden))
    return contrasts
