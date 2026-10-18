"""
The board pipeline end to end on a fully synthetic room, with a fake pose model
registered in the catalog: no private data, no model files, no MediaPipe.

It pins the output format the vault relies on (file names and pointing.jsonl keys)
and shows that the pose model is swapped by name.
"""
import json
import re

import cv2
import numpy as np
import pytest

from fakes import BOARD_QUAD, draw_person, draw_room
from vision_toolkit.core.geometry import load_quad, rectify_homography
from vision_toolkit.models import catalog
from vision_toolkit.models.catalog import ModelSpec
from vision_toolkit.usecases.board.pipeline import BoardOptions, run

POINTING_KEYS = {"t", "frame_xy", "board_xy", "board_xy_norm", "arm_angle", "occluded_fraction", "snapshot"}
POSITIONS = [100, 180, 260, 340, 420, 500, 420, 340, 260, 180, 100, 180]  # body centre per second
WRITING_FROM = 6  # seconds: a new chalk line appears


@pytest.fixture
def fake_pose(monkeypatch):
    monkeypatch.setattr(catalog, "_SPECS", dict(catalog._SPECS))
    catalog.register(ModelSpec(name="fake-pose", task="pose", backend="test", factory="fakes:ColorPose"))


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "room.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (640, 400))
    for t, cx in enumerate(POSITIONS):
        writer.write(draw_person(draw_room(writing=t >= WRITING_FROM), cx))
    writer.release()
    return path


def test_board_pipeline_outputs(fake_pose, video, tmp_path):
    out = run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose"))
    assert out == tmp_path / "out" / "room"

    names = {p.name for p in out.iterdir()}
    snapshots = sorted(n for n in names if n.startswith("board_") and n != "board_final.png")
    assert {"median.png", "quad.json", "board_final.png", "pointing.jsonl"} <= names
    assert snapshots and all(re.fullmatch(r"board_\d{2}h\d{2}m\d{2}s\.png", n) for n in snapshots)
    assert snapshots[-1] >= f"board_00h00m{WRITING_FROM:02d}s.png", "the new writing must trigger a snapshot"

    np.testing.assert_allclose(load_quad(out / "quad.json"), BOARD_QUAD, atol=6)

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
    H, size = rectify_homography(load_quad(out / "quad.json"))
    truth = cv2.warpPerspective(draw_room(writing=True), H, size).astype(np.float32)
    final = cv2.imread(str(out / "board_final.png")).astype(np.float32)
    assert (np.abs(final - truth).max(axis=2) > 40).mean() < 0.01


def test_wrong_model_task_is_refused(fake_pose, video, tmp_path):
    with pytest.raises(catalog.CatalogError, match="is a pose model"):
        run(video, tmp_path / "out", BoardOptions(pose_model="fake-pose", refine_model="fake-pose"))
