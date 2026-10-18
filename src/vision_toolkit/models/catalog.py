"""
Model catalog: every model is created and downloaded by name.

A ModelSpec says which task a model satisfies, which class builds it (imported
only when the model is created) and which files it needs in the models folder.
External packages add specs through the entry-point group "vision_toolkit.models":
the entry point resolves to a ModelSpec, an iterable of specs, or a callable
returning either.
"""
import importlib
import urllib.request
import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path

from vision_toolkit.core.paths import models_dir

ENTRY_POINT_GROUP = "vision_toolkit.models"
MEDIAPIPE_MODELS = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str                                    # catalog key, used by use cases and the CLI
    task: str                                    # task interface it satisfies: "segmentation", "pose", ...
    backend: str                                 # "mediapipe", later "onnx", "torch", ...
    factory: str                                 # "package.module:Class", imported lazily
    files: dict = field(default_factory=dict)    # file name in the models folder -> download URL
    extra: str | None = None                     # pip extra that installs the backend
    options: dict = field(default_factory=dict)  # default keyword arguments for the factory
    description: str = ""


class CatalogError(Exception):
    """A model cannot be used: unknown name, wrong task, missing files or missing backend."""


BUILTIN = (
    ModelSpec(
        name="mediapipe-pose-heavy",
        task="pose",
        backend="mediapipe",
        factory="vision_toolkit.models.mediapipe.pose_landmarker:PoseLandmarker",
        files={"pose_landmarker_heavy.task":
               f"{MEDIAPIPE_MODELS}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"},
        extra="mediapipe",
        description="MediaPipe PoseLandmarker (heavy): 33 keypoints and a body mask. "
                    "Two-stage, so it finds people that are small in the full frame.",
    ),
    ModelSpec(
        name="deeplab-v3",
        task="segmentation",
        backend="mediapipe",
        factory="vision_toolkit.models.mediapipe.image_segmenter:ImageSegmenter",
        files={"deeplab_v3.tflite":
               f"{MEDIAPIPE_MODELS}/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite"},
        extra="mediapipe",
        description="DeepLab V3, 257x257 input, 21 PASCAL VOC labels. "
                    "Misses small subjects in a full frame: run it on a crop (ops.crop_refine).",
    ),
    ModelSpec(
        name="selfie-multiclass",
        task="segmentation",
        backend="mediapipe",
        factory="vision_toolkit.models.mediapipe.image_segmenter:ImageSegmenter",
        files={"selfie_multiclass_256x256.tflite":
               f"{MEDIAPIPE_MODELS}/image_segmenter/selfie_multiclass_256x256/float32/latest/"
               "selfie_multiclass_256x256.tflite"},
        extra="mediapipe",
        description="MediaPipe selfie multiclass, 256x256: background, hair, body-skin, "
                    "face-skin, clothes, others. Made for close-up people.",
    ),
)

_SPECS = {spec.name: spec for spec in BUILTIN}
_discovered = False


def register(spec):
    """Add a spec to the catalog. Names are unique: a duplicate raises CatalogError."""
    if spec.name in _SPECS:
        raise CatalogError(f"Model {spec.name!r} is already in the catalog")
    _SPECS[spec.name] = spec


def get(name):
    specs = _specs()
    if name not in specs:
        raise CatalogError(f"Unknown model {name!r}. Available: {', '.join(sorted(specs))}")
    return specs[name]


def available(task=None):
    """Specs in the catalog, optionally only those of one task, sorted by name."""
    return [s for _, s in sorted(_specs().items()) if task is None or s.task == task]


def model_paths(spec, directory=None):
    directory = Path(directory or models_dir())
    return [directory / filename for filename in spec.files]


def missing_files(spec, directory=None):
    return [p for p in model_paths(spec, directory) if not p.exists()]


def create(name, task=None, directory=None, **options):
    """
    Build the model `name`. Its files are passed positionally (in the order of
    spec.files), followed by spec.options updated with `options`.

    task: if given, the model must satisfy that task (a pose model where a
    segmenter is expected fails here, with a clear message).
    """
    spec = get(name)
    if task is not None and spec.task != task:
        raise CatalogError(f"Model {name!r} is a {spec.task} model, a {task} model is needed here. "
                           f"{task.capitalize()} models: {', '.join(s.name for s in available(task)) or 'none'}")
    missing = missing_files(spec, directory)
    if missing:
        raise CatalogError(f"Missing file(s) for model {name!r}: {', '.join(str(p) for p in missing)}. "
                           f"Run: vision-toolkit models download {name}")
    factory = _load_factory(spec)
    return factory(*model_paths(spec, directory), **{**spec.options, **options})


def download(names=None, directory=None, log=print):
    """Download the files of the given models (all of them if names is empty). Returns their paths."""
    directory = Path(directory or models_dir())
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in [get(n) for n in names] if names else available():
        for filename, url in spec.files.items():
            path = directory / filename
            if path.exists():
                log(f"already present: {filename}")
            else:
                log(f"downloading {filename}...")
                partial = path.with_name(path.name + ".part")
                urllib.request.urlretrieve(url, partial)
                partial.replace(path)
            paths.append(path)
    return paths


def _load_factory(spec):
    module_name, _, attr = spec.factory.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        factory_itself = e.name == module_name or module_name.startswith(f"{e.name}.")
        if spec.extra and not factory_itself:
            raise CatalogError(f"Model {spec.name!r} needs the {spec.backend} backend ({e.name} is not installed): "
                               f'pip install -e ".[{spec.extra}]"') from e
        raise
    return getattr(module, attr)


def _specs():
    global _discovered
    if not _discovered:
        _discovered = True
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                found = ep.load()
                found = found() if callable(found) else found
                for spec in [found] if isinstance(found, ModelSpec) else found:
                    register(spec)
            except Exception as e:
                warnings.warn(f"Skipping model plugin {ep.name!r}: {e}")
    return _SPECS
