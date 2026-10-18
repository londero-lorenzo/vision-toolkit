"""Scarica in models/ i modelli MediaPipe usati dal progetto."""
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_landmarker_heavy.task": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "deeplab_v3.tflite": f"{BASE}/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite",
    "selfie_multiclass_256x256.tflite": f"{BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "models"
    target.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = target / name
        if path.exists():
            print(f"già presente: {name}")
            continue
        print(f"scarico {name}...")
        urllib.request.urlretrieve(url, path)
