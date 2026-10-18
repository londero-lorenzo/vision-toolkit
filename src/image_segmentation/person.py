from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Indici dei landmark di MediaPipe Pose
SHOULDER = {"left": 11, "right": 12}
INDEX = {"left": 19, "right": 20}
HIP = {"left": 23, "right": 24}
ARMS = [(11, 13, 15, 19), (12, 14, 16, 20)]  # spalla, gomito, polso, indice
DEEPLAB_PERSON = 15


@dataclass
class PersonResult:
    mask: np.ndarray            # uint8 0/255, già dilatata
    pointer: tuple | None       # (x, y) della punta dell'indice del braccio più sollevato
    arm_angle: float            # gradi tra braccio e busto: ~0 a riposo, ~90 teso
    landmarks: list | None


class PersonSegmenter:
    """
    Maschera del docente e punto indicato.

    Usa PoseLandmarker perché lavora in due passi (rileva la persona, poi la
    ritaglia): DeepLab V3 sul frame intero non vede il docente, che occupa
    pochi pixel dopo il ridimensionamento a 257x257, mentre sul ritaglio sì.
    Con refine=True, DeepLab gira sul ritaglio e si unisce alla maschera della
    posa, che a volte perde un tratto del braccio teso.
    """

    def __init__(self, refine=False, dilate=15, models_dir=MODELS_DIR):
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(models_dir / "pose_landmarker_heavy.task")),
            output_segmentation_masks=True,
        )
        self.pose = vision.PoseLandmarker.create_from_options(opts)
        self.deeplab = None
        if refine:
            seg_opts = vision.ImageSegmenterOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(models_dir / "deeplab_v3.tflite")),
                output_category_mask=True,
            )
            self.deeplab = vision.ImageSegmenter.create_from_options(seg_opts)
        self.kernel = np.ones((dilate, dilate), np.uint8) if dilate else None

    def close(self):
        self.pose.close()
        if self.deeplab:
            self.deeplab.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, frame):
        h, w = frame.shape[:2]
        result = self.pose.detect(_mp_image(frame))
        mask = np.zeros((h, w), np.uint8)
        if not result.pose_landmarks:
            return PersonResult(mask, None, 0.0, None)

        lm = result.pose_landmarks[0]
        mask[result.segmentation_masks[0].numpy_view().squeeze() > 0.5] = 255
        _draw_arms(mask, lm, w, h)
        if self.deeplab is not None:
            mask |= self._deeplab_on_crop(frame, mask)
        if self.kernel is not None:
            mask = cv2.dilate(mask, self.kernel)

        pointer, angle = _pointing_hand(lm, w, h)
        return PersonResult(mask, pointer, angle, lm)

    def _deeplab_on_crop(self, frame, mask, margin=0.25):
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        x0, y0 = max(0, x0 - mx), max(0, y0 - my)
        x1, y1 = min(frame.shape[1], x1 + mx), min(frame.shape[0], y1 + my)
        # numpy_view è una vista sulla memoria del risultato: copiarla prima che venga liberato
        cat = self.deeplab.segment(_mp_image(frame[y0:y1, x0:x1])).category_mask.numpy_view().squeeze().copy()
        out = np.zeros_like(mask)
        out[y0:y1, x0:x1][cat == DEEPLAB_PERSON] = 255
        return out


def _mp_image(frame_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _draw_arms(mask, lm, w, h, min_visibility=0.5):
    """
    Disegna le braccia come segmenti spessi tra i landmark.

    La maschera della posa a volte perde un tratto del braccio teso proprio
    sopra la lavagna; lo scheletro non dipende dall'aspetto e lo copre.
    """
    p = lambda i: (int(lm[i].x * w), int(lm[i].y * h))
    shoulders = np.linalg.norm(np.subtract(p(SHOULDER["left"]), p(SHOULDER["right"])))
    thickness = max(3, int(0.3 * shoulders))
    for chain in ARMS:
        for a, b in zip(chain, chain[1:]):
            if lm[a].visibility > min_visibility and lm[b].visibility > min_visibility:
                cv2.line(mask, p(a), p(b), 255, thickness)


def _pointing_hand(lm, w, h):
    """
    Sceglie il braccio più sollevato e ritorna la punta dell'indice in pixel
    con l'angolo tra braccio e busto (spalla -> anca): ~0° lungo il fianco,
    ~90° teso in orizzontale. La lunghezza del braccio non basta: uno lungo
    il fianco è lungo quanto uno teso.
    """
    p = lambda i: np.array([lm[i].x * w, lm[i].y * h])

    def angle(side):
        arm = p(INDEX[side]) - p(SHOULDER[side])
        torso = p(HIP[side]) - p(SHOULDER[side])
        cos = arm @ torso / (np.linalg.norm(arm) * np.linalg.norm(torso) or 1.0)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    best = max(("left", "right"), key=angle)
    x, y = p(INDEX[best])
    return (int(x), int(y)), angle(best)
