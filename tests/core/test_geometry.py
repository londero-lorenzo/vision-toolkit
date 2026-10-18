import cv2
import numpy as np

from vision_toolkit.core.geometry import load_quad, order_corners, polygon_mask, rectify_homography, save_quad, to_plane

QUAD = np.float32([[60, 50], [580, 70], [570, 300], [70, 290]])  # tl, tr, br, bl


def test_order_corners_from_any_order():
    shuffled = QUAD[[2, 0, 3, 1]]
    np.testing.assert_array_equal(order_corners(shuffled), QUAD)


def test_rectify_maps_corners_to_rectangle():
    H, (w, h) = rectify_homography(QUAD)
    corners = cv2.perspectiveTransform(QUAD[None], H)[0]
    np.testing.assert_allclose(corners, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], atol=1e-3)
    assert w == round((np.linalg.norm(QUAD[1] - QUAD[0]) + np.linalg.norm(QUAD[2] - QUAD[3])) / 2)


def test_rectify_with_given_width_keeps_aspect():
    _, (w1, h1) = rectify_homography(QUAD)
    _, (w2, h2) = rectify_homography(QUAD, width=w1 * 2)
    assert w2 == w1 * 2 and abs(h2 - 2 * h1) <= 1


def test_to_plane_inside_and_outside():
    H, (w, h) = rectify_homography(QUAD)
    x, y = to_plane(tuple(QUAD.mean(axis=0)), H, (w, h))
    assert abs(x - w / 2) < 0.1 * w and abs(y - h / 2) < 0.1 * h
    assert to_plane((5, 5), H, (w, h)) is None
    assert to_plane(None, H, (w, h)) is None


def test_polygon_mask_area():
    mask = polygon_mask((100, 100, 3), [[10, 10], [59, 10], [59, 39], [10, 39]])
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
    assert (mask > 0).sum() == 50 * 30


def test_quad_json_round_trip(tmp_path):
    path = tmp_path / "quad.json"
    save_quad(path, QUAD)
    assert '"quad"' in path.read_text(encoding="utf-8")
    np.testing.assert_array_equal(load_quad(path), QUAD)
