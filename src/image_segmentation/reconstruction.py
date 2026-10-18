import cv2
import numpy as np


class BoardReconstructor:
    """
    Ricostruisce la lavagna senza docente accumulando i frame nel tempo.

    Ogni frame viene rettificato con l'omografia della lavagna; i pixel coperti
    dal docente (maschera della persona) diventano NaN. Lo stato della lavagna è
    la mediana, per pixel, degli ultimi `window` campioni validi: un pixel coperto
    in tutta la finestra conserva l'ultimo valore noto. La mediana assorbe anche
    i pezzi di docente sfuggiti alla maschera, purché restino nello stesso punto
    per meno di metà finestra.
    """

    def __init__(self, H, size, window=7, change_threshold=25):
        self.H = H
        self.size = size  # (width, height)
        self.window = window
        self.change_threshold = change_threshold
        w, h = size
        self.buffer = np.full((window, h, w, 3), np.nan, np.float32)
        self.index = 0
        self.board = np.zeros((h, w, 3), np.float32)
        self.seen = np.zeros((h, w), bool)
        self.last_snapshot = None

    def update(self, frame, person_mask):
        """
        Aggiunge un frame. Ritorna la maschera dei pixel di lavagna occlusi
        (in coordinate rettificate).
        """
        warped = cv2.warpPerspective(frame, self.H, self.size).astype(np.float32)
        occluded = cv2.warpPerspective(person_mask, self.H, self.size, flags=cv2.INTER_NEAREST) > 0

        warped[occluded] = np.nan
        self.buffer[self.index % self.window] = warped
        self.index += 1

        valid = ~np.isnan(self.buffer[..., 0])
        any_valid = valid.any(axis=0)
        with np.errstate(all="ignore"):
            median = np.nanmedian(self.buffer[:, any_valid], axis=0)
        self.board[any_valid] = median
        self.seen |= any_valid
        return occluded

    def image(self, unseen_color=(255, 0, 255)):
        """Lavagna corrente in uint8; i pixel mai visti hanno unseen_color."""
        out = np.clip(self.board, 0, 255).astype(np.uint8)
        out[~self.seen] = unseen_color
        return out

    def change_since_snapshot(self):
        """
        Frazione di pixel cambiati (scrittura o cancellazione) rispetto
        all'ultimo snapshot preso.
        """
        if self.last_snapshot is None:
            return 1.0
        diff = np.abs(self.board - self.last_snapshot).max(axis=2)
        return float((diff[self.seen] > self.change_threshold).mean())

    def take_snapshot(self):
        self.last_snapshot = self.board.copy()
        return self.image()

    def to_board(self, point):
        """Porta un punto del frame in coordinate della lavagna rettificata."""
        if point is None:
            return None
        p = cv2.perspectiveTransform(np.float32([[point]]), self.H)[0, 0]
        w, h = self.size
        if 0 <= p[0] < w and 0 <= p[1] < h:
            return int(p[0]), int(p[1])
        return None
