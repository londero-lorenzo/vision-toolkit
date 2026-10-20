import cv2
import numpy as np
import pytest

from vision_toolkit.core.framing import Framing, camera_motion, framing_segments

SHIFT = np.float32([[1, 0, 5], [0, 1, 3]])


def textured(size=(320, 240)):
    """A wall with blurred blocks: texture an alignment can lock onto."""
    rng = np.random.default_rng(0)
    w, h = size
    image = np.full((h, w, 3), 170, np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, w - 30)), int(rng.integers(0, h - 30))
        cv2.rectangle(image, (x, y), (x + 24, y + 16), [int(v) for v in rng.integers(40, 250, 3)], -1)
    return cv2.GaussianBlur(image, (5, 5), 0)


def moved(image, matrix=SHIFT):
    return cv2.warpAffine(image, matrix, image.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, frames[0].shape[1::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def test_camera_motion_measures_shift_and_rotation():
    wall = textured()
    motion = camera_motion(wall, moved(wall))
    assert abs(motion.dx) == pytest.approx(5, abs=0.3) and abs(motion.dy) == pytest.approx(3, abs=0.3)
    assert motion.shift == pytest.approx(np.hypot(5, 3), abs=0.3) and abs(motion.angle) < 0.1

    turned = moved(wall, cv2.getRotationMatrix2D((160, 120), 1.0, 1.0))
    assert abs(camera_motion(wall, turned).angle) == pytest.approx(1.0, abs=0.1)
    assert camera_motion(wall, np.full_like(wall, 170)) is None  # nothing to align to


def test_a_move_splits_the_video_and_one_odd_frame_does_not(tmp_path):
    wall = textured()
    frames = [wall if t < 10 else moved(wall) for t in range(30)]
    frames[21] = moved(wall, np.float32([[1, 0, 12], [0, 1, 0]]))  # a single odd frame, on a sampled time
    path = write_video(tmp_path / "moved.mp4", frames)

    first, second = framing_segments(path, step=3)
    assert first.start == 0 and 9 <= first.end <= 10.5
    assert 9.5 <= second.start <= 12 and second.end == pytest.approx(30)
    assert second.reference in (12.0, 15.0) and second.duration > 18


def test_a_still_camera_or_a_flat_wall_give_one_segment(tmp_path):
    still = write_video(tmp_path / "still.mp4", [textured()] * 12)
    assert framing_segments(still, step=2) == [Framing(0.0, pytest.approx(12), 0.0)]

    flat = [np.full((240, 320, 3), 170, np.uint8)] * 6 + [np.full((240, 320, 3), 90, np.uint8)] * 6
    assert len(framing_segments(write_video(tmp_path / "flat.mp4", flat), step=2)) == 1

    mask = np.zeros((240, 320), np.uint8)
    mask[:60] = 255
    frames = [textured() if t < 6 else moved(textured()) for t in range(12)]
    moved_path = write_video(tmp_path / "moved.mp4", frames)
    assert len(framing_segments(moved_path, mask=mask, step=2)) == 2
    assert len(framing_segments(moved_path, mask=mask, step=2, end=5)) == 1
