import numpy as np
from mediapipe.tasks.python import vision

from vision_toolkit.models.mediapipe._common import base_options, read_mask, to_mp_image
from vision_toolkit.tasks.base import Model
from vision_toolkit.tasks.pose import Keypoint, Pose

# MediaPipe's 33 landmarks in index order, renamed to the task's names ("left_index", ...)
KEYPOINT_NAMES = tuple(lm.name.lower() for lm in vision.PoseLandmark)


class PoseLandmarker(Model):
    """
    MediaPipe PoseLandmarker: 33 keypoints and a body mask per person.

    It works in two stages (detect the person, then crop and analyse it), so it
    finds people that are small in the full frame. Satisfies tasks.pose.PoseEstimator.
    """

    keypoint_names = KEYPOINT_NAMES

    def __init__(self, model_path, num_poses=1, mask_threshold=0.5):
        opts = vision.PoseLandmarkerOptions(
            base_options=base_options(model_path),
            num_poses=num_poses,
            output_segmentation_masks=True,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.mask_threshold = mask_threshold

    def close(self):
        self._landmarker.close()

    def __call__(self, image):
        h, w = image.shape[:2]
        result = self._landmarker.detect(to_mp_image(image))
        poses = []
        for i, landmarks in enumerate(result.pose_landmarks):
            keypoints = {
                name: Keypoint(lm.x * w, lm.y * h, 1.0 if lm.visibility is None else lm.visibility)
                for name, lm in zip(KEYPOINT_NAMES, landmarks)
            }
            mask = None
            if result.segmentation_masks:
                mask = (read_mask(result.segmentation_masks[i]) > self.mask_threshold).astype(np.uint8) * 255
            poses.append(Pose(keypoints, mask))
        return poses
