import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt


def base_options(model_path):
    return mpt.BaseOptions(model_asset_path=str(model_path))


def to_mp_image(image_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def read_mask(mp_mask):
    """
    Copy a MediaPipe output mask into a NumPy array.

    numpy_view() is a view on the result's memory: once a temporary result is
    freed, the next access crashes natively (dead Jupyter kernel, access
    violation) instead of raising. Every MediaPipe mask is read through here.
    """
    return mp_mask.numpy_view().squeeze().copy()
