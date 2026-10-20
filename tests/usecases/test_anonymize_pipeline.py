"""
The anonymize pipeline end to end on a fully synthetic room, with fake models
registered in the catalog: no private data, no model files, no MediaPipe.
"""
import json

import cv2
import numpy as np
import pytest

from fakes import draw_face, draw_person, draw_room
from vision_toolkit.core.io import FrameReader
from vision_toolkit.models import catalog
from vision_toolkit.models.catalog import ModelSpec
from vision_toolkit.usecases.anonymize.pipeline import AnonymizeOptions, SetupPending, load_setup, run, save_setup

POSITIONS = [100, 180, 260, 340, 420, 500, 420, 340, 260, 180, 100, 180]  # body centre per second
FACE_Y = 100       # face centre, just above the body (draw_person's top is 120)
BLOB = (560, 330, 600, 370)  # something the pose model does not see
MANIFEST_KEYS = {"name", "t_requested", "t", "status", "group", "reasons", "heads_from_pose",
                 "faces_from_detector", "replaced_because", "skipped_because"}


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(catalog, "_SPECS", dict(catalog._SPECS))
    catalog.register(ModelSpec(name="fake-head-pose", task="pose", backend="test", factory="fakes:HeadColorPose"))
    catalog.register(ModelSpec(name="fake-skin", task="segmentation", backend="test", factory="fakes:SkinSegmenter"))
    catalog.register(ModelSpec(name="fake-faces", task="detection", backend="test", factory="fakes:FakeFaceDetector",
                               options={"boxes": [(580, 20, 610, 50)]}))


def make_video(path, blob_times=()):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (640, 400))
    for t, cx in enumerate(POSITIONS):
        frame = draw_face(draw_person(draw_room(), cx), cx, FACE_Y)
        if t in blob_times:
            cv2.rectangle(frame, BLOB[:2], BLOB[2:], (255, 0, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def options(**changes):
    fields = dict(step=1.0, pose_model="fake-head-pose", face_model=None, skin_model="fake-skin",
                  substitute_window=0)
    return AnonymizeOptions(**{**fields, **changes})


def confirmed_run(video, out_root, opts):
    """First run proposes the setup and stops; confirm it as the person would, then run again."""
    with pytest.raises(SetupPending):
        run(video, out_root, opts)
    path = out_root / video.stem / "setup.json"
    setup = load_setup(path)
    setup.confirmed = True
    save_setup(path, setup)
    return run(video, out_root, opts)


def manifest(out):
    return {row["name"]: row for row in map(json.loads, (out / "manifest.jsonl").read_text().splitlines())}


def test_first_run_proposes_the_setup_and_stops(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4")
    with pytest.raises(SetupPending, match="setup.png"):
        run(video, tmp_path / "out", options())
    out = tmp_path / "out" / "room"
    setup = load_setup(out / "setup.json")
    assert setup.crop == (0, 0, 640, 400) and setup.redact_below is None
    assert setup.frame_size == (640, 400) and not setup.confirmed
    assert (out / "setup.png").exists() and not (out / "todo").exists()


def test_faces_are_pixelated_and_nothing_else_changes(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4")
    summary = confirmed_run(video, tmp_path / "out", options())
    out = summary.out
    assert summary.auto == len(POSITIONS) and summary.doubtful == 0

    queued = cv2.imread(str(out / "todo" / "auto" / "frame_00h00m02s.png"))
    with FrameReader(video) as reader:
        original = reader.at(2.0)
    cx = POSITIONS[2]

    def detail(image):
        face = image[FACE_Y - 10:FACE_Y + 10, cx - 10:cx + 10]
        return cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert detail(queued) < 0.1 * detail(original), "the checkerboard features must be gone"

    outside = np.ones(original.shape[:2], bool)
    outside[FACE_Y - 50:FACE_Y + 50, cx - 50:cx + 50] = False
    np.testing.assert_array_equal(queued[outside], original[outside])

    rows = manifest(out)
    assert len(rows) == len(POSITIONS) and all(set(row) == MANIFEST_KEYS for row in rows.values())
    assert all(row["status"] == "queued" and row["heads_from_pose"] == 1 for row in rows.values())


def test_close_doubtful_frames_become_one_episode(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4", blob_times=(4, 5, 6))
    summary = confirmed_run(video, tmp_path / "out", options())
    assert (summary.doubtful, summary.episodes, summary.doubtful_queued) == (3, 1, 1)
    assert summary.auto == len(POSITIONS) - 3 and summary.skipped == 2

    rows = manifest(summary.out)
    assert rows["frame_00h00m05s"]["group"] == "doubtful"
    assert rows["frame_00h00m04s"]["skipped_because"] == rows["frame_00h00m06s"]["skipped_because"] == "episode"
    record = json.loads((summary.out / "queue.jsonl").read_text().splitlines()[-1])
    assert record["name"] == "frame_00h00m05s" and record["reasons"] == ["person-without-head"]
    x0, y0, x1, y1 = record["boxes"][0]
    assert abs(x0 - BLOB[0]) <= 4 and abs(y1 - BLOB[3]) <= 4
    assert (summary.out / "todo" / "doubtful" / "frame_00h00m05s.png").exists()


def test_a_confident_neighbour_replaces_a_doubtful_frame(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4", blob_times=(4,))
    summary = confirmed_run(video, tmp_path / "out", options(step=2.0, substitute_window=3))
    assert summary.doubtful == 0 and summary.auto == 6
    row = manifest(summary.out)["frame_00h00m04s"]
    assert row["t_requested"] == 4.0 and row["t"] == 5.0 and row["group"] == "auto"
    assert row["replaced_because"] == ["person-without-head"]


def test_a_face_away_from_every_pose_is_a_doubt(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4")
    summary = confirmed_run(video, tmp_path / "out", options(step=4.0, face_model="fake-faces"))
    assert (summary.doubtful, summary.episodes, summary.doubtful_queued) == (3, 1, 1)
    row = manifest(summary.out)["frame_00h00m04s"]
    assert row["reasons"] == ["face-outside-pose"] and row["faces_from_detector"] == 1


def test_a_new_run_keeps_the_decisions(fakes, tmp_path):
    video = make_video(tmp_path / "room.mp4")
    summary = confirmed_run(video, tmp_path / "out", options(step=4.0))
    out = summary.out
    (out / "approved").mkdir()
    (out / "todo" / "auto" / "frame_00h00m00s.png").rename(out / "approved" / "frame_00h00m00s.png")
    approved = (out / "approved" / "frame_00h00m00s.png").read_bytes()

    again = run(video, tmp_path / "out", options(step=4.0))
    assert again.decided == 1 and again.auto == 2
    assert (out / "approved" / "frame_00h00m00s.png").read_bytes() == approved
    assert not (out / "todo" / "auto" / "frame_00h00m00s.png").exists()
    assert manifest(out)["frame_00h00m00s"]["status"] == "decided"
