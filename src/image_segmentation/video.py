import cv2
import numpy as np


def iter_frames(path, step=1.0, start=0.0, end=None, crop=None):
    """
    Scorre il video un frame ogni `step` secondi.

    crop: (x, y, w, h) opzionale, per tagliare via l'interfaccia del player
    se il video è una registrazione dello schermo.

    Produce:
        (t_secondi, frame_bgr)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {path}")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
    end = duration if end is None else min(end, duration)

    t = start
    try:
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            yield t, apply_crop(frame, crop)
            t += step
    finally:
        cap.release()


def apply_crop(frame, crop):
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


def median_frame(path, samples=30, crop=None):
    """
    Mediana temporale di `samples` frame distribuiti su tutto il video.

    Il docente si sposta, quindi nella mediana sparisce: è il frame giusto
    su cui cercare la lavagna.
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(apply_crop(frame, crop))
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def format_time(t):
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"
