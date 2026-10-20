import cv2
import numpy as np
import pytest

from vision_toolkit.core.io import FrameReader, apply_crop, content_box, format_time, iter_frames, median_frame


@pytest.fixture
def video(tmp_path):
    """4 s at 5 fps: gray background with a white square moving left to right."""
    path = tmp_path / "moving.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 96))
    for i in range(20):
        frame = np.full((96, 160, 3), 100, np.uint8)
        cv2.rectangle(frame, (i * 7, 30), (i * 7 + 20, 50), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_iter_frames_step_and_crop(video):
    frames = list(iter_frames(video, step=1.0, crop=(10, 20, 64, 32)))
    assert [t for t, _ in frames] == [0.0, 1.0, 2.0, 3.0]
    assert all(f.shape == (32, 64, 3) for _, f in frames)


def test_iter_frames_window(video):
    times = [t for t, _ in iter_frames(video, step=0.5, start=1.0, end=2.5)]
    assert times == [1.0, 1.5, 2.0]


def test_median_frame_removes_moving_object(video):
    median = median_frame(video, samples=10)
    assert np.abs(median.astype(int) - 100).max() < 15


def test_median_frame_of_a_stretch(tmp_path):
    path = tmp_path / "two_halves.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), 60 if i < 10 else 200, np.uint8))
    writer.release()
    assert abs(int(median_frame(path, samples=5, end=2.0).mean()) - 60) < 5
    assert abs(int(median_frame(path, samples=5, start=2.0).mean()) - 200) < 5


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        next(iter_frames(tmp_path / "missing.mp4"))


def test_apply_crop_and_format_time():
    frame = np.zeros((10, 20, 3), np.uint8)
    assert apply_crop(frame, None) is frame
    assert apply_crop(frame, (2, 3, 5, 4)).shape == (4, 5, 3)
    assert format_time(0) == "00h00m00s"
    assert format_time(693.7) == "00h11m33s"
    assert format_time(3725) == "01h02m05s"


def test_frame_reader_jumps_back_and_forth(video):
    with FrameReader(video, crop=(0, 20, 64, 32)) as reader:
        assert reader.duration == pytest.approx(4.0)
        late, early = reader.at(3.0), reader.at(0.0)
        assert late.shape == early.shape == (32, 64, 3)
        assert early.max() > 200 and late.max() < 200  # the white square starts at the left
        assert reader.at(4.0) is None and reader.at(-1.0) is None


def test_content_box_leaves_the_player_bars_out():
    frame = np.zeros((120, 200, 3), np.uint8)
    frame[10:90, 5:170] = 120       # the video
    frame[100:102, 2:198] = 200     # a progress bar across the whole width
    frame[108:116, 4:40] = 230      # a name label
    assert content_box(frame) == (5, 10, 165, 80)
    assert content_box(np.zeros((10, 10, 3), np.uint8)) is None
