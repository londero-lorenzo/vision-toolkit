import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS_ENV = "VISION_TOOLKIT_MODELS"


def models_dir():
    """
    Folder holding the downloaded model files.

    Defaults to `models/` at the repository root; the environment variable
    VISION_TOOLKIT_MODELS overrides it (read at call time, not at import).
    """
    return Path(os.environ.get(MODELS_ENV) or ROOT / "models")
