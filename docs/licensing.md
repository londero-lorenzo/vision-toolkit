# Licensing

vision-toolkit is released under the [Apache License 2.0](../LICENSE); see also [NOTICE](../NOTICE).
The reasons are in [decision 0007](decisions/0007-apache-2.0-licence.md). The licence covers every
published commit, including those made before `LICENSE` was added.

## Dependency policy

The core and the extras accept permissive dependencies (MIT, BSD, Apache-2.0, PSF, ISC, MPL-2.0).
`scripts/check_licenses.py` fails on any GPL, LGPL or AGPL distribution in the environment. Run it
with every extra installed:

```
pip install -e ".[dev,mediapipe]"
python scripts/check_licenses.py
```

| Dependency | Licence | Used by |
|---|---|---|
| NumPy | BSD-3-Clause | core |
| opencv-python | Apache-2.0; bundles LGPL libraries (below) | core, `models/opencv/` |
| MediaPipe | Apache-2.0 | `[mediapipe]` extra |
| opencv-contrib-python | Apache-2.0, as opencv-python | `[mediapipe]` extra, required by MediaPipe |
| JupyterLab, ipywidgets | BSD-3-Clause | `[dev]` extra |
| Matplotlib | Matplotlib License (PSF-based) | `[dev]` extra |
| pytest | MIT | `[dev]` extra |

## LGPL libraries

An LGPL library is allowed when it is used unmodified, as a separate file loaded dynamically, so that
the user can replace it. Each one is listed here, with the reason. The check reads package metadata,
which shows only OpenCV's Apache-2.0, so the libraries inside the OpenCV wheels are listed by hand.

| Library | Licence | Where | Why it is allowed |
|---|---|---|---|
| FFmpeg | LGPL-2.1 | every OpenCV wheel (`cv2/opencv_videoio_ffmpeg*.dll` on Windows) | OpenCV loads it at run time to read and write videos. It is not modified, and the wheel ships it as a separate file with its licence in `LICENSE-3RD-PARTY.txt` |
| Qt 5 | LGPL-3.0 | the non-headless OpenCV wheels for Linux and macOS | the same: a separate library of the wheel, used by OpenCV's windows, not by vision-toolkit |

## Models

Model files are not part of the repository. `vision-toolkit models download` fetches them from their
publishers into `models/` (or `$VISION_TOOLKIT_MODELS`), and their own licences apply to that copy.

| Catalog name | File | Licence | Source |
|---|---|---|---|
| `mediapipe-pose-heavy` | `pose_landmarker_heavy.task` | Apache-2.0 | [model card BlazePose GHUM 3D](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) |
| `deeplab-v3` | `deeplab_v3.tflite` | Apache-2.0 | [TensorFlow DeepLabV3 on Kaggle Models](https://www.kaggle.com/models/tensorflow/deeplabv3) |
| `selfie-multiclass` | `selfie_multiclass_256x256.tflite` | Apache-2.0 | [model card Multiclass Segmentation](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf) |
| `yunet-face` | `face_detection_yunet_2023mar.onnx` | MIT | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |

Checked on 2026-10-23. A model enters the catalog only if its licence allows any use, commercial use
included, and its row is added here in the same change. Models under copyleft licences (for example
Ultralytics YOLO, AGPL-3.0) or non-commercial ones go in separate packages, which register them
through the `vision_toolkit.models` entry points under their own terms.

## Copyleft components

GPL and AGPL libraries and models may be used only from **separate, optional packages** that register
models through the `vision_toolkit.models` entry points. Such packages carry their own licence and are
not part of this repository's code.

## Data

The licence covers the code and the documents. The lecture videos and frames used to calibrate the use
cases are private and outside git ([0005](decisions/0005-anonymized-frames.md),
[0006](decisions/0006-data-names.md)), and no licence is granted on them. Tests and examples use
synthetic data.

## Contributions

Contributions are accepted under the Apache License 2.0 (inbound = outbound, section 5 of the
licence): opening a pull request means agreeing that the contribution is licensed under the same terms
as the project. No CLA is needed, and contributors keep their copyright. Apache-2.0 allows anyone to
use the code in their own projects, provided the licence, the copyright notices and `NOTICE` are kept
and changed files are marked.
