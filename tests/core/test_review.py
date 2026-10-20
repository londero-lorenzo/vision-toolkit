import cv2
import numpy as np

from vision_toolkit.core.review import ReviewQueue

IMAGE = np.full((8, 8, 3), 100, np.uint8)
EMPTY = {"todo": 0, "approved": 0, "edited": 0, "rejected": 0, "missing": 0}


def move(root, name, group, to):
    (root / to).mkdir(exist_ok=True)
    (root / "todo" / group / f"{name}.png").rename(root / to / f"{name}.png")


def test_add_puts_the_image_in_todo(tmp_path):
    queue = ReviewQueue(tmp_path)
    assert queue.add(IMAGE, "frame_a", "auto", "Hidden?", reasons=["r"], boxes=[(1, 2, 3, 4)])
    item = queue.item("frame_a")
    assert item.state == "todo" and item.path == tmp_path / "todo" / "auto" / "frame_a.png"
    assert item.question == "Hidden?" and item.reasons == ["r"] and item.boxes == [[1, 2, 3, 4]]
    assert not item.edited
    np.testing.assert_array_equal(cv2.imread(str(item.path)), IMAGE)
    assert queue.status() == {"auto": {**EMPTY, "todo": 1}}
    assert queue.item("nope") is None


def test_decisions_are_moves_and_edits_are_detected(tmp_path):
    queue = ReviewQueue(tmp_path)
    for name in ("a", "b", "c", "d"):
        queue.add(IMAGE, name, "auto", "Hidden?")
    move(tmp_path, "a", "auto", "approved")
    cv2.imwrite(str(tmp_path / "todo" / "auto" / "b.png"), IMAGE // 2)  # edited in Paint.NET, then approved
    move(tmp_path, "b", "auto", "approved")
    move(tmp_path, "c", "auto", "rejected")
    (tmp_path / "todo" / "auto" / "d.png").unlink()
    assert queue.status() == {"auto": {"todo": 0, "approved": 2, "edited": 1, "rejected": 1, "missing": 1}}
    assert queue.decided("a") and queue.decided("c") and not queue.decided("d") and not queue.decided("nope")


def test_a_producer_never_overwrites_decisions_or_edits(tmp_path):
    queue = ReviewQueue(tmp_path)
    for name in ("a", "b", "c"):
        queue.add(IMAGE, name, "auto", "Hidden?")
    move(tmp_path, "a", "auto", "approved")
    approved = (tmp_path / "approved" / "a.png").read_bytes()
    cv2.imwrite(str(tmp_path / "todo" / "auto" / "b.png"), IMAGE // 2)  # being edited, not decided yet
    editing = (tmp_path / "todo" / "auto" / "b.png").read_bytes()

    other = np.zeros_like(IMAGE)
    assert not queue.add(other, "a", "doubtful", "Hidden?")
    assert not queue.add(other, "b", "doubtful", "Hidden?")
    assert queue.add(other, "c", "doubtful", "Hidden?")  # untouched: replaced, in its new group
    assert (tmp_path / "approved" / "a.png").read_bytes() == approved
    assert (tmp_path / "todo" / "auto" / "b.png").read_bytes() == editing
    assert not (tmp_path / "todo" / "auto" / "c.png").exists()
    assert queue.item("c").group == "doubtful" and queue.item("c").state == "todo"


def test_add_from_files(tmp_path):
    jpg, png = tmp_path / "snap.jpg", tmp_path / "board.png"
    cv2.imwrite(str(jpg), IMAGE)
    cv2.imwrite(str(png), IMAGE)
    queue = ReviewQueue(tmp_path / "queue")
    queue.add(jpg, "snap", "snapshots", "Is the lecturer absent?")
    queue.add(png, "board", "snapshots", "Is the lecturer absent?")
    assert cv2.imread(str(tmp_path / "queue" / "todo" / "snapshots" / "snap.png")).shape == (8, 8, 3)
    assert (tmp_path / "queue" / "todo" / "snapshots" / "board.png").read_bytes() == png.read_bytes()
