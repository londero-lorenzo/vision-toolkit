import cv2
import numpy as np

from vision_toolkit.tasks.base import Model
from vision_toolkit.tasks.detection import Detection
from vision_toolkit.tasks.pose import Keypoint

# YuNet's five landmarks in output order, renamed to the task's names (the person's right eye first)
KEYPOINT_NAMES = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")


class YuNetFaceDetector(Model):
    """
    OpenCV's YuNet face detector (cv2.FaceDetectorYN): a box, a score and five
    facial keypoints per face. Light on CPU, it finds faces down to about 10 px,
    frontal or in partial profile. Satisfies tasks.detection.Detector.
    """

    labels = ("face",)

    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.3):
        self._detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                                   score_threshold, nms_threshold)

    def __call__(self, image):
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        detections = []
        for row in [] if faces is None else faces:
            x, y, bw, bh = row[:4]
            box = (max(0, int(x)), max(0, int(y)), min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh))))
            keypoints = {name: Keypoint(float(px), float(py))
                         for name, (px, py) in zip(KEYPOINT_NAMES, row[4:14].reshape(5, 2))}
            detections.append(Detection(box, float(row[14]), "face", keypoints))
        return detections
