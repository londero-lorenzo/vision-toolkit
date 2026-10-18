"""
Test end-to-end su un video sintetico.

Dal frame di esempio si ricava uno sfondo senza docente (inpainting della sua
area), poi si genera un video in cui la sua sagoma scorre davanti alla lavagna.
La ricostruzione deve tornare allo sfondo, mentre un singolo frame no.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_segmentation.board import detect_board, rectify_homography
from image_segmentation.person import PersonSegmenter
from image_segmentation.reconstruction import BoardReconstructor
from image_segmentation.video import iter_frames

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data" / "frame" / "lezione_prova_cit_censured.png"
VIDEO_CROP = (0, 68, 1674, 783)  # area del video nello screenshot del player
OFFSETS = [0, 250, 500, 750, 1000, 750, 500, 250, 0, -250]


def make_scene(segment):
    x, y, w, h = VIDEO_CROP
    frame = cv2.imread(str(FRAME))[y:y + h, x:x + w]
    person = segment(frame).mask
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
def segment():
    with PersonSegmenter() as s:
        yield s


@pytest.mark.skipif(not FRAME.exists(), reason="frame di esempio non presente in data/frame")
def test_reconstruction_removes_teacher(segment, tmp_path):
    frame, person, background = make_scene(segment)
    quad = detect_board(frame, person)
    assert quad is not None
    H, size = rectify_homography(quad)

    # Scrive il video sintetico e lo rilegge con iter_frames, come la CLI
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1, (frame.shape[1], frame.shape[0]))
    for dx in OFFSETS:
        writer.write(paste(background, frame, person, dx))
    writer.release()

    rec = BoardReconstructor(H, size, window=5)
    for _, f in iter_frames(video, step=1.0):
        rec.update(f, segment(f).mask)

    truth = cv2.warpPerspective(background, H, size).astype(np.float32)
    single = cv2.warpPerspective(paste(background, frame, person, OFFSETS[-1]), H, size).astype(np.float32)
    err_rec = np.abs(rec.board - truth).max(axis=2)
    err_single = np.abs(single - truth).max(axis=2)

    # Pixel "sbagliati": scarto > 40 livelli (tiene conto della compressione mp4v)
    bad_rec = float((err_rec > 40).mean())
    bad_single = float((err_single > 40).mean())
    cv2.imwrite(str(ROOT / "output" / "test_synthetic_board.png"), rec.image())
    print(f"pixel errati: ricostruzione {bad_rec:.4f}, frame singolo {bad_single:.4f}, mai visti {(~rec.seen).mean():.4f}")

    assert rec.seen.all()
    assert bad_rec < 0.002
    assert bad_single > 10 * bad_rec
