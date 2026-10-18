import json

import cv2
import numpy as np


def detect_board(frame, person_mask=None):
    """
    Trova la lavagna come quadrilatero nel frame.

    Assume una lavagna scura su parete chiara (soglia di Otsu sul grigio).
    Se si passa la maschera della persona, i suoi pixel vengono esclusi prima
    di cercare il contorno: altrimenti le gambe che escono sotto la lavagna
    deformano il quadrilatero.

    Ritorna:
        quad: array (4, 2) float32 in ordine tl, tr, br, bl, oppure None
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if person_mask is not None:
        dark[person_mask > 0] = 0

    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))

    # Aumenta la tolleranza finché il contorno non si riduce a 4 vertici
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            return order_corners(quad.reshape(4, 2))
    return None


def order_corners(pts):
    """Ordina 4 punti come tl, tr, br, bl."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def board_mask(shape, quad):
    """Maschera uint8 (0/255) dell'area della lavagna nel frame."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(quad)], 255)
    return mask


def rectify_homography(quad, width=None):
    """
    Omografia frame -> lavagna rettificata.

    Se width non è dato, usa la lunghezza del lato superiore; l'altezza
    segue il rapporto medio dei lati del quadrilatero.

    Ritorna:
        H, (width, height)
    """
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if width is None:
        width = int(round(w))
    height = int(round(width * h / w))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(np.float32(quad), dst), (width, height)


def save_quad(path, quad):
    """Salva il quadrilatero (calibrazione manuale o automatica) in JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"quad": np.asarray(quad).tolist()}, f, indent=2)


def load_quad(path):
    with open(path, encoding="utf-8") as f:
        return np.float32(json.load(f)["quad"])
