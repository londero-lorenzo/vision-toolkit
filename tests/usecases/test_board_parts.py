"""The pieces of the board use case on synthetic images: board corners, person blob, chalk."""
import cv2
import numpy as np

from fakes import BOARD_QUAD, draw_room
from vision_toolkit.usecases.board.chalk import change_threshold, stroke_contrast
from vision_toolkit.usecases.board.detect import detect_board, find_board
from vision_toolkit.usecases.board.person_mask import person_blob


def lit_room():
    """The room with a light strip along the board's top edge, whose glow lightens the top of the board."""
    room = draw_room()
    tl, tr = BOARD_QUAD[0], BOARD_QUAD[1]
    glow = np.float32([tl, tr, tr + [0, 14], tl + [0, 14]])
    cv2.fillPoly(room, [np.int32(glow)], (175, 175, 175))
    cv2.line(room, tuple(np.int32(tl - [0, 1])), tuple(np.int32(tr - [0, 1])), (252, 252, 252), 3)
    return room


def test_find_board_on_a_plain_room():
    np.testing.assert_allclose(find_board(draw_room()), BOARD_QUAD, atol=2)


def test_the_light_strip_moves_the_top_side_to_the_board_edge():
    room = lit_room()
    otsu = detect_board(room)
    assert otsu[0][1] - BOARD_QUAD[0][1] > 8, "Otsu takes the glow for wall"
    np.testing.assert_allclose(find_board(room), BOARD_QUAD, atol=3)


def test_find_board_ignores_what_lies_below_the_floor():
    room = draw_room()
    room[330:] = 0  # the audience blacked out: a dark band larger than the board
    np.testing.assert_allclose(find_board(room, floor=330), BOARD_QUAD, atol=2)


def test_person_blob_wants_a_standing_shape():
    foreground = np.zeros((400, 640), np.uint8)
    foreground[150:360, 300:340] = 255      # a person cut by the audience line at 360
    foreground[60:200, 400:600] = 255       # a large change on the board: does not reach the floor
    foreground[330:360, 50:250] = 255       # low and wide: not tall enough
    box = person_blob(foreground, floor=360)
    ys, xs = np.nonzero(box)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (150, 359, 300, 339)
    assert not person_blob(foreground).any()  # the frame bottom is the floor: nothing reaches it


def test_stroke_contrast_and_threshold():
    board = np.full((240, 600, 3), 100, np.uint8)
    for y in range(40, 200, 30):
        cv2.line(board, (40, y), (560, y + 5), (116, 116, 116), 3)  # faint chalk, 16 levels
    assert abs(stroke_contrast(board) - 16) <= 2
    hidden = np.ones(board.shape[:2], bool)
    assert stroke_contrast(board, hidden) is None
    assert stroke_contrast(np.full((240, 600, 3), 100, np.uint8)) is None
    slate = np.random.default_rng(0).normal(100, 2, (240, 600, 1)).repeat(3, axis=2)
    assert stroke_contrast(slate.clip(0, 255).astype(np.uint8)) is None  # texture, not chalk

    assert change_threshold([16, 18, None]) == 8.5
    assert change_threshold([4]) == 6.0 and change_threshold([180]) == 25.0
    assert change_threshold([None, None]) == 25.0
