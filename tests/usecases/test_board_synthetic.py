"""
End-to-end test on a synthetic video built from a real lecture frame.

From the sample frame a background without the lecturer is derived (inpainting
of the lecturer's area), then a video is generated in which that silhouette moves in front of
the board. The reconstruction must return to the background, a single frame must not.
The frame shows a real person: it stays in data/ (outside git) and without it the
test is skipped.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_toolkit.core.geometry import rectify_homography
from vision_toolkit.core.io import iter_frames
from vision_toolkit.core.temporal import PlaneMedian
from vision_toolkit.usecases.board.detect import detect_board
from vision_toolkit.usecases.board.person_mask import PersonMasker

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "frame" / "lezione_prova_cit_censured.png"
VIDEO_CROP = (0, 68, 1674, 783)  # video area in the player screenshot
OFFSETS = [0, 250, 500, 750, 1000, 750, 500, 250, 0, -250]

pytestmark = pytest.mark.skipif(not FRAME.exists(), reason="sample frame not present in data/frame")


def make_scene(masker):
    x, y, w, h = VIDEO_CROP
    frame = cv2.imread(str(FRAME))[y:y + h, x:x + w]
    person = masker(frame).mask
    background = cv2.inpaint(frame, person, 7, cv2.INPAINT_TELEA)
    return frame, person, background


def paste(background, frame, person, dx):
    out = background.copy()
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    size = (frame.shape[1], frame.shape[0])
    moved = cv2.warpAffine(frame, M, size)
    mask = cv2.warpAffine(cv2.erode(person, np.ones((15, 15), np.uint8)), M, size) > 0
    out[mask] = moved[mask]
    return out


@pytest.fixture(scope="module")
def masker():
    with PersonMasker() as m:
        yield m


def test_reconstruction_removes_lecturer(masker, tmp_path):
    frame, person, background = make_scene(masker)
    quad = detect_board(frame, person)
    assert quad is not None
    H, size = rectify_homography(quad)

    # Write the synthetic video and read it back with iter_frames, as the CLI does
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1, (frame.shape[1], frame.shape[0]))
    for dx in OFFSETS:
        writer.write(paste(background, frame, person, dx))
    writer.release()

    rec = PlaneMedian(H, size, window=5)
    for _, f in iter_frames(video, step=1.0):
        rec.update(f, masker(f).mask)

    truth = cv2.warpPerspective(background, H, size).astype(np.float32)
    single = cv2.warpPerspective(paste(background, frame, person, OFFSETS[-1]), H, size).astype(np.float32)
    err_rec = np.abs(rec.background - truth).max(axis=2)
    err_single = np.abs(single - truth).max(axis=2)

    # "Wrong" pixels: off by more than 40 levels (allows for the mp4v compression)
    bad_rec = float((err_rec > 40).mean())
    bad_single = float((err_single > 40).mean())
    (ROOT / "output").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "output" / "test_synthetic_board.png"), rec.image())
    print(f"wrong pixels: reconstruction {bad_rec:.4f}, single frame {bad_single:.4f}, never seen {(~rec.seen).mean():.4f}")

    assert rec.seen.all()
    assert bad_rec < 0.002
    assert bad_single > 10 * bad_rec
