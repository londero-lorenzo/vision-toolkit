"""
The board pipeline end to end on a fully synthetic room, with fake pose models
registered in the catalog: no private data, no model files, no MediaPipe.

It pins the output format the vault relies on (file names and pointing.jsonl keys)
and shows that the pose model is swapped by name.
"""
import json
import re

import cv2
import numpy as np
import pytest

from fakes import BOARD_QUAD, PERSON_BGR, draw_person, draw_room
from vision_toolkit.core.geometry import load_quad, rectify_homography
from vision_toolkit.core.layout import Setup, save_setup
from vision_toolkit.models import catalog
from vision_toolkit.models.catalog import ModelSpec
from vision_toolkit.usecases.board.pipeline import BoardOptions, SetupError, run

POINTING_KEYS = {"t", "frame_xy", "board_xy", "board_xy_norm", "arm_angle", "occluded_fraction", "snapshot"}
POSITIONS = [100, 180, 260, 340, 420, 500, 420, 340, 260, 180, 100, 180]  # body centre per second
WRITING_FROM = 6  # seconds: a new chalk line appears
STILL = [100, 500, 100, 500, 420, 500, 300, 300, 300, 300, 300, 500]  # stands still in front for 5 s
FAST = dict(step=1.0, window=7)


@pytest.fixture
def fake_pose(monkeypatch):
    monkeypatch.setattr(catalog, "_SPECS", dict(catalog._SPECS))
    catalog.register(ModelSpec(name="fake-pose", task="pose", backend="test", factory="fakes:ColorPose"))
    catalog.register(ModelSpec(name="one-person-pose", task="pose", backend="test", factory="fakes:LargestPersonPose"))
    catalog.register(ModelSpec(name="blind-pose", task="pose", backend="test", factory="fakes:BlindSpotPose",
                               options={"blind": (290, 310)}))


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, frames[0].shape[1::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video(tmp_path):
    frames = [draw_person(draw_room(writing=t >= WRITING_FROM), cx) for t, cx in enumerate(POSITIONS)]
    return write_video(tmp_path / "room.mp4", frames)


def board_errors(out, frame, quad=None):
    """Fraction of the final board's pixels far from the board in `frame`, rectified with the run's quad."""
    H, size = rectify_homography(quad if quad is not None else load_quad(out / "quad.json"))
    truth = cv2.warpPerspective(frame, H, size).astype(np.float32)
    final = cv2.imread(str(out / "board_final.png")).astype(np.float32)
    return float((np.abs(final - truth).max(axis=2) > 40).mean())


def test_board_pipeline_outputs(fake_pose, video, tmp_path):
    out = run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", **FAST))
    assert out == tmp_path / "out" / "room"

    names = {p.name for p in out.iterdir()}
    snapshots = sorted(n for n in names if n.startswith("board_") and n != "board_final.png")
    assert {"median.png", "quad.json", "board_final.png", "pointing.jsonl", "run.json"} <= names
    assert snapshots and all(re.fullmatch(r"board_\d{2}h\d{2}m\d{2}s\.png", n) for n in snapshots)
    assert snapshots[-1] >= f"board_00h00m{WRITING_FROM:02d}s.png", "the new writing must trigger a snapshot"

    np.testing.assert_allclose(load_quad(out / "quad.json"), BOARD_QUAD, atol=3)

    lines = [json.loads(line) for line in (out / "pointing.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(POSITIONS)
    for line in lines:
        assert set(line) == POINTING_KEYS
        assert line["snapshot"] is None or line["snapshot"] in names
    pointed = [line for line in lines if line["board_xy"] is not None]
    assert pointed and all(0 <= v <= 1 for line in pointed for v in line["board_xy_norm"])
    assert all(line["arm_angle"] > 80 for line in pointed)
    assert max(line["occluded_fraction"] for line in lines) > 0.02

    # The final board has no lecturer and has the new writing
    assert board_errors(out, draw_room(writing=True)) < 0.01

    summary = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert [f["used"] for f in summary["framings"]] == [True]
    assert summary["frames"] == {"analysed": 12, "lecturer_by_pose": 12, "lecturer_by_foreground": 0, "nobody": 0}
    assert summary["change_threshold"] == 25 and summary["chalk_contrast"] > 100  # bold synthetic chalk
    assert summary["snapshots"] == len(snapshots)


def test_wrong_model_task_is_refused(fake_pose, video, tmp_path):
    with pytest.raises(catalog.CatalogError, match="is a pose model"):
        run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", refine_model="fake-pose"))


def test_audience_is_blacked_out_before_the_pose(fake_pose, tmp_path):
    # A student in front, bigger than what shows of the lecturer: a model looking for
    # one person picks the student, and the lecturer standing still enters the board
    def frame(cx):
        room = draw_person(draw_room(), cx, bottom=310)  # legs behind the desk
        cv2.rectangle(room, (150, 330), (470, 399), PERSON_BGR, -1)
        return room

    path = write_video(tmp_path / "audience.mp4", [frame(cx) for cx in STILL])
    seen = run(path, tmp_path / "seen", BoardOptions(pose_model="one-person-pose", fallback=False, **FAST))
    hidden = run(path, tmp_path / "hidden", BoardOptions(pose_model="one-person-pose", redact_below=320,
                                                         fallback=False, **FAST))
    assert board_errors(seen, draw_room()) > 0.02
    assert board_errors(hidden, draw_room()) < 0.01


def test_setup_gives_crop_and_audience_line(fake_pose, video, tmp_path):
    setup = tmp_path / "setup.json"
    save_setup(setup, Setup(None, 360, (640, 400)))
    with pytest.raises(SetupError, match="not confirmed"):
        run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", setup=setup, **FAST))
    save_setup(setup, Setup(None, 360, (320, 200), confirmed=True))
    with pytest.raises(SetupError, match="320x200"):
        run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", setup=setup, **FAST))
    save_setup(setup, Setup(None, 360, (640, 400), confirmed=True))
    out = run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", setup=setup, **FAST))
    assert (cv2.imread(str(out / "median.png"))[360:] == 0).all(), "the audience is blacked out in the median"


def test_foreground_masks_the_lecturer_when_the_pose_misses(fake_pose, tmp_path):
    path = write_video(tmp_path / "blind.mp4", [draw_person(draw_room(), cx) for cx in STILL])
    options = dict(pose_model="blind-pose", redact_below=360, **FAST)
    without = run(path, tmp_path / "without", BoardOptions(fallback=False, **options))
    with_fallback = run(path, tmp_path / "with", BoardOptions(**options))
    assert board_errors(without, draw_room()) > 0.02
    assert board_errors(with_fallback, draw_room()) < 0.01
    frames = json.loads((with_fallback / "run.json").read_text(encoding="utf-8"))["frames"]
    assert frames["lecturer_by_foreground"] == 5 and frames["nobody"] == 0


def test_exposure_jump_changes_no_pixel(fake_pose, tmp_path):
    # The camera brightens by 12% halfway, with nothing written on the board. The room
    # is empty at first, so the first snapshot already shows the whole board
    positions = [None, None, None, 100, 260, 420, 500, 340, 180, 100, 260, 420, 500, 340, 180, 100, 260, 420]
    frames = [draw_room() if cx is None else draw_person(draw_room(), cx) for cx in positions]
    frames = [f if t < 7 else np.clip(f * 1.12, 0, 255).astype(np.uint8) for t, f in enumerate(frames)]
    path = write_video(tmp_path / "exposure.mp4", frames)
    options = dict(pose_model="fake-pose", change_threshold=6, **FAST)

    def changed(out):
        first = cv2.imread(str(out / "board_00h00m01s.png")).astype(np.float32)
        final = cv2.imread(str(out / "board_final.png")).astype(np.float32)
        return float((np.abs(final - first).max(axis=2) > 6).mean())

    matched = run(path, tmp_path / "matched", BoardOptions(**options))
    unmatched = run(path, tmp_path / "unmatched", BoardOptions(match_exposure=False, **options))
    assert changed(matched) < 0.004 < 0.04 < changed(unmatched)  # 0.004: the default min_change
    gain = json.loads((matched / "run.json").read_text(encoding="utf-8"))["exposure_gain"]
    assert abs(gain["p95"] / gain["p5"] - 1.12) < 0.02


def test_camera_moved_by_hand(fake_pose, tmp_path):
    # The camera moves by 6 px at t = 10 s: the run splits the video in two framings,
    # finds the board in each and rebuilds one board from both
    shift = np.float32([[1, 0, 4], [0, 1, 5]])
    frames = []
    for t in range(24):
        frame = draw_person(draw_room(writing=t >= 16, wall_marks=True), [100, 300, 500][t % 3])
        frames.append(frame if t < 10 else cv2.warpAffine(frame, shift, (640, 400), borderMode=cv2.BORDER_REPLICATE))
    path = write_video(tmp_path / "moved.mp4", frames)
    out = run(path, tmp_path / "out", BoardOptions(pose_model="fake-pose", framing_step=2, min_segment=5,
                                                   step=1.0, window=5))

    framings = json.loads((out / "run.json").read_text(encoding="utf-8"))["framings"]
    assert [f["used"] for f in framings] == [True, True]
    assert 8 <= framings[0]["end"] <= 11 and 9 <= framings[1]["start"] <= 12
    np.testing.assert_allclose(framings[0]["quad"], BOARD_QUAD, atol=3)
    np.testing.assert_allclose(framings[1]["quad"], BOARD_QUAD + [4, 5], atol=3)
    moved = cv2.warpAffine(draw_room(writing=True, wall_marks=True), shift, (640, 400), borderMode=cv2.BORDER_REPLICATE)
    assert board_errors(out, moved, np.float32(framings[1]["quad"])) < 0.01
