import cv2
import numpy as np

from vision_toolkit import cli


def test_review_add_and_status(tmp_path, capsys):
    image = tmp_path / "board_00h01m00s.png"
    cv2.imwrite(str(image), np.full((8, 8, 3), 100, np.uint8))
    queue = tmp_path / "queue"

    cli.main(["review", "status", str(queue)])
    assert "No items" in capsys.readouterr().out

    cli.main(["review", "add", str(queue), str(image), "--group", "snapshots", "--question", "Absent?"])
    assert "added: board_00h01m00s" in capsys.readouterr().out
    assert (queue / "todo" / "snapshots" / "board_00h01m00s.png").exists()

    cli.main(["review", "status", str(queue)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["group", "todo", "approved", "edited", "rejected", "missing"]
    assert lines[1].split() == ["snapshots", "1", "0", "0", "0", "0"]
