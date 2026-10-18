import tomllib
from pathlib import Path

import pytest

from vision_toolkit.models import catalog
from vision_toolkit.models.catalog import CatalogError, ModelSpec
from vision_toolkit.tasks.pose import PoseEstimator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry(monkeypatch):
    """Isolated catalog: specs registered in a test do not leak into the others."""
    monkeypatch.setattr(catalog, "_SPECS", dict(catalog._SPECS))
    return catalog


def fake_pose_spec(**changes):
    fields = dict(name="fake-pose", task="pose", backend="test", factory="fakes:ColorPose")
    return ModelSpec(**{**fields, **changes})


def test_builtin_specs_are_consistent():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    names = [s.name for s in catalog.BUILTIN]
    assert len(names) == len(set(names))
    for spec in catalog.BUILTIN:
        module, _, attr = spec.factory.partition(":")
        assert module.startswith(f"vision_toolkit.models.{spec.backend}.") and attr, spec.name
        assert (ROOT / "src" / Path(*module.split("."))).with_suffix(".py").exists(), spec.name
        assert spec.task in {"segmentation", "pose"}, spec.name
        assert spec.files and all(url.startswith("https://") for url in spec.files.values()), spec.name
        assert spec.extra in extras, spec.name


def test_available_filters_by_task():
    assert {s.name for s in catalog.available("pose")} >= {"mediapipe-pose-heavy"}
    assert {s.name for s in catalog.available("segmentation")} >= {"deeplab-v3", "selfie-multiclass"}
    assert all(s.task == "pose" for s in catalog.available("pose"))


def test_create_resolves_factory_lazily(registry):
    registry.register(fake_pose_spec(options={"tolerance": 10}))
    with registry.create("fake-pose", task="pose") as model:
        assert isinstance(model, PoseEstimator)
        assert model.tolerance == 10
    assert registry.create("fake-pose", tolerance=5).tolerance == 5


def test_files_are_passed_to_the_factory(registry, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"0")
    registry.register(fake_pose_spec(name="with-file", factory="pathlib:Path", files={"weights.bin": "https://x"}))
    assert registry.create("with-file", directory=tmp_path) == tmp_path / "weights.bin"


def test_clear_errors(registry, tmp_path, monkeypatch):
    # A model module that exists but whose backend is not installed
    (tmp_path / "plugin_model.py").write_text("import a_backend_that_is_not_installed\n", encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)
    registry.register(fake_pose_spec())
    registry.register(fake_pose_spec(name="needs-file", files={"weights.bin": "https://x"}))
    registry.register(fake_pose_spec(name="needs-backend", factory="plugin_model:Model", extra="onnx"))

    with pytest.raises(CatalogError, match="Unknown model 'nope'.*fake-pose"):
        registry.create("nope")
    with pytest.raises(CatalogError, match="is a pose model, a segmentation model is needed"):
        registry.create("fake-pose", task="segmentation")
    with pytest.raises(CatalogError, match="vision-toolkit models download needs-file"):
        registry.create("needs-file", directory=tmp_path)
    with pytest.raises(CatalogError, match=r'pip install -e "\.\[onnx\]"'):
        registry.create("needs-backend")
    with pytest.raises(CatalogError, match="already in the catalog"):
        registry.register(fake_pose_spec())


def test_entry_point_plugins(monkeypatch):
    class EntryPoint:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def load(self):
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    plugins = [EntryPoint("one", fake_pose_spec(name="plugin-one")),
               EntryPoint("many", lambda: [fake_pose_spec(name="plugin-a"), fake_pose_spec(name="plugin-b")]),
               EntryPoint("broken", ImportError("boom"))]
    monkeypatch.setattr(catalog, "_SPECS", dict(catalog._SPECS))
    monkeypatch.setattr(catalog, "_discovered", False)
    monkeypatch.setattr(catalog, "entry_points", lambda group: plugins if group == catalog.ENTRY_POINT_GROUP else [])

    with pytest.warns(UserWarning, match="Skipping model plugin 'broken'"):
        names = {s.name for s in catalog.available()}
    assert {"plugin-one", "plugin-a", "plugin-b", "deeplab-v3"} <= names


def test_download_skips_present_files(registry, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"0")
    registry.register(fake_pose_spec(name="with-file", files={"weights.bin": "https://unreachable.invalid/w"}))
    log = []
    assert registry.download(["with-file"], directory=tmp_path, log=log.append) == [tmp_path / "weights.bin"]
    assert log == ["already present: weights.bin"]
