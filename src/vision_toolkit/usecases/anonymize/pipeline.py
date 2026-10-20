import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision_toolkit.core.io import FrameReader, apply_crop, content_box, format_time, median_frame
from vision_toolkit.core.redact import pixelate
from vision_toolkit.core.review import ReviewQueue
from vision_toolkit.models import catalog
from vision_toolkit.ops.faces import face_mask, head_box, head_width
from vision_toolkit.ops.skeleton import draw_limbs
from vision_toolkit.usecases.anonymize.doubts import (FACE_OUTSIDE_POSE, HEAD_AT_EDGE, PERSON_WITHOUT_HEAD,
                                                      episodes, foreground_mask, representative, select,
                                                      unexplained_blobs)

DEFAULT_POSE_MODEL = "mediapipe-pose-heavy"
DEFAULT_FACE_MODEL = "yunet-face"
DEFAULT_SKIN_MODEL = "selfie-multiclass"
SKIN_LABELS = ("face-skin",)
QUESTION = "Are all faces hidden, with nothing else identifying in view? Approving releases the frame."


class SetupPending(RuntimeError):
    """The crop and the audience line are waiting for the person's confirmation in setup.json."""


@dataclass
class AnonymizeOptions:
    step: float = 300.0                 # seconds between two frames
    times: tuple | None = None          # explicit times in seconds, instead of step, start and end
    start: float = 0.0
    end: float | None = None
    setup: Path | None = None           # setup.json to use; default OUT/<video name>/setup.json
    max_people: int = 2                 # passed to the pose model as num_poses
    pose_model: str = DEFAULT_POSE_MODEL
    face_model: str | None = DEFAULT_FACE_MODEL
    skin_model: str | None = DEFAULT_SKIN_MODEL
    blocks: int = 6                     # pixelation blocks across a face
    review_budget: int | None = None    # most doubtful episodes to queue; None queues them all
    substitute_window: float = 3.0      # seconds around a doubtful frame searched for a confident one
    episode_gap: float | None = None    # doubtful frames this close form one episode; 2 * step, 0 with times
    min_blob: float = 0.002             # fraction of the frame: smallest unexplained foreground that is a doubt
    models_dir: Path | None = None


@dataclass
class Setup:
    crop: tuple | None          # (x, y, w, h) of the video area in the recording
    redact_below: int | None    # rows from here down, in the cropped frame, are blacked out (the audience)
    frame_size: tuple           # (w, h) of the videos the setup is for
    confirmed: bool = False


@dataclass
class Summary:
    out: Path
    requested: int = 0
    auto: int = 0               # confident frames queued
    doubtful: int = 0           # doubtful frames left after looking for substitutes
    episodes: int = 0
    doubtful_queued: int = 0
    skipped: int = 0
    decided: int = 0            # already approved or rejected in an earlier run


def run(video, out_dir, options=None):
    """
    Hide the faces in frames of a video and put the frames in a review queue.

    Writes to out_dir/<video name>/:
    - setup.json, setup.png: the proposed crop and audience line. Without a confirmed
      setup the run writes them and stops with SetupPending
    - queue.jsonl, todo/auto/, todo/doubtful/: the review queue (core.review)
    - manifest.jsonl: one line per requested time

    Returns a Summary.
    """
    options = options or AnonymizeOptions()
    video = Path(video)
    out = Path(out_dir) / video.stem
    out.mkdir(parents=True, exist_ok=True)

    median = median_frame(video)
    setup = _confirmed_setup(Path(options.setup) if options.setup else out / "setup.json", median, out)
    background = _redact(apply_crop(median, setup.crop), setup.redact_below)

    queue = ReviewQueue(out)
    summary = Summary(out)
    rows, doubtful, names = [], [], set()
    with ExitStack() as stack:
        directory = options.models_dir
        pose = stack.enter_context(catalog.create(options.pose_model, task="pose", directory=directory,
                                                  num_poses=options.max_people))
        detector = None
        if options.face_model:
            detector = stack.enter_context(catalog.create(options.face_model, task="detection", directory=directory))
        skin = None
        if options.skin_model:
            skin = stack.enter_context(catalog.create(options.skin_model, task="segmentation", directory=directory))
            if not set(SKIN_LABELS) <= set(skin.labels):
                raise catalog.CatalogError(f"Model {options.skin_model!r} has no label(s) {list(SKIN_LABELS)}; "
                                           f"its labels are {list(skin.labels)}")
        reader = stack.enter_context(FrameReader(video, setup.crop))
        anonymize = _FrameAnonymizer(reader, background, setup.redact_below, options, pose, detector, skin)

        for t_requested in _times(options, reader.duration):
            name = f"frame_{format_time(t_requested)}"
            if name in names:  # two times within the same second
                continue
            names.add(name)
            row = {"name": name, "t_requested": t_requested, "t": None, "status": "skipped", "group": None,
                   "reasons": [], "heads_from_pose": 0, "faces_from_detector": 0, "replaced_because": None,
                   "skipped_because": None}
            rows.append(row)
            if queue.decided(name):
                row["status"] = "decided"
                continue
            result, doubts = _first_confident(anonymize, t_requested, options.substitute_window)
            if result is None:
                row["skipped_because"] = "unreadable"
                continue
            row.update(t=result.t, reasons=result.reasons, heads_from_pose=result.heads,
                       faces_from_detector=result.faces)
            if result.t != t_requested:
                row["replaced_because"] = doubts
            if result.reasons:
                doubtful.append(row)
            elif queue.add(result.image, name, "auto", QUESTION):
                row.update(status="queued", group="auto")
            else:
                row["skipped_because"] = "edited-in-todo"

        # Doubtful frames close in time show the same scene: one per episode goes to review
        gap = options.episode_gap if options.episode_gap is not None else 0.0 if options.times else 2 * options.step
        groups = episodes(doubtful, gap, key=lambda row: row["t_requested"])
        chosen = select(groups, options.review_budget)
        for group in groups:
            keep = representative(group) if any(group is c for c in chosen) else None
            for row in group:
                if row is not keep:
                    row["skipped_because"] = "budget" if keep is None else "episode"
                    continue
                result = anonymize(row["t"])  # analysed again: only metadata was kept in memory
                if result is None:
                    row["skipped_because"] = "unreadable"
                elif queue.add(result.image, row["name"], "doubtful", QUESTION, result.reasons, result.boxes):
                    row.update(status="queued", group="doubtful")
                else:
                    row["skipped_because"] = "edited-in-todo"

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda r: r["t_requested"]):
            f.write(json.dumps(row) + "\n")

    summary.requested = len(rows)
    summary.auto = sum(row["group"] == "auto" for row in rows)
    summary.doubtful, summary.episodes = len(doubtful), len(groups)
    summary.doubtful_queued = sum(row["group"] == "doubtful" for row in rows)
    summary.skipped = sum(row["status"] == "skipped" for row in rows)
    summary.decided = sum(row["status"] == "decided" for row in rows)
    return summary


def load_setup(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crop = data.get("crop")
    return Setup(tuple(crop) if crop else None, data.get("redact_below"), tuple(data["frame_size"]),
                 bool(data.get("confirmed", False)))


def save_setup(path, setup):
    data = {"crop": list(setup.crop) if setup.crop else None, "redact_below": setup.redact_below,
            "frame_size": list(setup.frame_size), "confirmed": setup.confirmed}
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


@dataclass
class _Analysis:
    t: float
    image: np.ndarray   # the frame with its faces pixelated
    reasons: list       # doubts; empty when the frame is confident
    boxes: list         # where the doubts are
    heads: int
    faces: int


class _FrameAnonymizer:
    """Reads the frame at a time, pixelates the faces it finds and lists its doubts."""

    def __init__(self, reader, background, redact_below, options, pose, detector, skin):
        self.reader, self.background, self.redact_below = reader, background, redact_below
        self.options, self.pose, self.detector, self.skin = options, pose, detector, skin
        k = max(3, int(0.02 * background.shape[1]))
        self.kernel = np.ones((k, k), np.uint8)

    def __call__(self, t):
        frame = self.reader.at(t)
        if frame is None:
            return None
        frame = _redact(frame, self.redact_below)
        h, w = frame.shape[:2]
        poses = self.pose(frame)
        faces = self.detector(frame) if self.detector is not None else []
        heads = [pose for pose in poses if head_width(pose) is not None]

        image = frame
        for subject in heads + faces:
            block = max(2, round(head_width(subject) / self.options.blocks))
            image = pixelate(image, face_mask(frame, subject, self.skin, SKIN_LABELS), block)

        # What the poses with a head explain: body, arms and head
        explained = np.zeros((h, w), np.uint8)
        for pose in heads:
            if pose.mask is not None:
                explained |= cv2.dilate(pose.mask, self.kernel)
            draw_limbs(explained, pose)
            _fill(explained, head_box(pose))
        for face in faces:
            _fill(explained, _grown(face.box, 2.0))

        blobs = unexplained_blobs(foreground_mask(frame, self.background), explained, self.options.min_blob * h * w)
        blobs += [_keypoint_box(pose, w, h) for pose in poses if head_width(pose) is None]
        pose_heads = [head_box(pose) for pose in heads]
        stray = [face.box for face in faces if not any(_contains(box, _centre(face.box)) for box in pose_heads)]
        edge = [box for box in (head_box(s, scale=1.2) for s in heads + faces) if _touches_edge(box, w, h)]
        reasons = [reason for reason, found in ((PERSON_WITHOUT_HEAD, blobs), (FACE_OUTSIDE_POSE, stray),
                                                (HEAD_AT_EDGE, edge)) if found]
        boxes = [_clip(box, w, h) for box in blobs + stray + edge]
        return _Analysis(t, image, reasons, boxes, len(heads), len(faces))


def _first_confident(anonymize, t, window):
    """
    The analysis at t or, if it has doubts, the nearest one within `window` seconds
    that has none; with it, the doubts found at t (["unreadable"] if t could not be read).
    """
    first = anonymize(t)
    doubts = first.reasons if first is not None else ["unreadable"]
    if first is not None and not first.reasons:
        return first, []
    for k in range(1, int(window) + 1):
        for s in (t + k, t - k):
            result = anonymize(s)
            if result is not None and not result.reasons:
                return result, doubts
            first = first or result
    return first, doubts


def _confirmed_setup(path, median, out):
    """The setup at `path` if confirmed; otherwise propose it or draw it again, and stop."""
    h, w = median.shape[:2]
    if path.exists():
        setup = load_setup(path)
        if tuple(setup.frame_size) != (w, h):
            raise ValueError(f"{path} is for {setup.frame_size[0]}x{setup.frame_size[1]} frames, "
                             f"this video has {w}x{h}")
        if setup.confirmed:
            return setup
    else:
        setup = Setup(content_box(median), None, (w, h))
        save_setup(path, setup)
    cv2.imwrite(str(out / "setup.png"), _draw_setup(median, setup))
    raise SetupPending(f"Check {out / 'setup.png'} (video area in green, audience line in red). "
                       f"Correct crop and redact_below in {path} if needed, set confirmed to true, then run again.")


def _draw_setup(median, setup):
    image = median.copy()
    x, y, w, h = setup.crop or (0, 0, image.shape[1], image.shape[0])
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    if setup.redact_below is not None:
        cv2.line(image, (x, y + setup.redact_below), (x + w - 1, y + setup.redact_below), (0, 0, 255), 2)
    return image


def _times(options, duration):
    if options.times:
        return sorted({round(float(t), 2) for t in options.times if 0 <= t < duration})
    end = duration if options.end is None else min(options.end, duration)
    return [round(float(t), 2) for t in np.arange(options.start, end, options.step)]


def _redact(frame, redact_below):
    if redact_below is None:
        return frame
    frame = frame.copy()
    frame[max(0, redact_below):] = 0
    return frame


def _clip(box, w, h):
    x0, y0, x1, y1 = box
    return max(0, min(w, x0)), max(0, min(h, y0)), max(0, min(w, x1)), max(0, min(h, y1))


def _fill(mask, box):
    x0, y0, x1, y1 = _clip(box, mask.shape[1], mask.shape[0])
    mask[y0:y1, x0:x1] = 255


def _grown(box, scale):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = scale * (x1 - x0) / 2, scale * (y1 - y0) / 2
    return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)


def _keypoint_box(pose, w, h):
    if not pose.keypoints:
        return 0, 0, w, h
    xs, ys = zip(*(kp.xy for kp in pose.keypoints.values()))
    return int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _contains(box, point):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _touches_edge(box, w, h):
    x0, y0, x1, y1 = box
    return x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h
