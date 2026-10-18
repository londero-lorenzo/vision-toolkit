"""
Dependency rules between the layers of docs/architecture.md, checked on the imports
of every module in src/vision_toolkit. If a rule has to change, change the document
first, then ALLOWED here.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vision_toolkit"
ARCHITECTURE = ROOT / "docs" / "architecture.md"

# Layer -> layers its modules may import ("root" is vision_toolkit/__init__.py)
ALLOWED = {
    "root": {"root"},
    "core": {"root", "core"},
    "tasks": {"root", "core", "tasks"},
    "models": {"root", "core", "tasks", "models"},
    "catalog": {"root", "core"},
    "ops": {"root", "core", "tasks", "ops"},
    "usecases": {"root", "core", "tasks", "ops", "usecases", "catalog"},
    "cli": {"root", "core", "tasks", "catalog", "ops", "usecases", "cli"},
}
# Libraries that only models/<backend>/ may import
BACKEND_LIBRARIES = {"mediapipe", "onnxruntime", "torch", "torchvision", "tensorflow", "ultralytics"}


def layer_of(parts):
    """Layer of a module given its path parts below vision_toolkit."""
    if not parts or parts == ["__init__"]:
        return "root"
    if parts[0] == "models" and parts[1:2] == ["catalog"]:
        return "catalog"
    return parts[0] if parts[0] in ALLOWED else "root"


def module_parts(path):
    return list(path.relative_to(SRC).with_suffix("").parts)


def imports_of(path):
    """(top-level library, parts below vision_toolkit or None) for every import in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module_parts(path)[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], _internal(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ["vision_toolkit", *package[:len(package) - node.level + 1]]
                module = base + (node.module.split(".") if node.module else [])
            else:
                module = node.module.split(".")
            for alias in node.names:
                full = module + [alias.name]
                is_module = (SRC.parent / Path(*full)).with_suffix(".py").exists() or (SRC.parent / Path(*full)).is_dir()
                yield module[0], _internal(full if is_module else module)


def _internal(parts):
    return parts[1:] if parts[0] == "vision_toolkit" else None


def violations(path):
    src_parts = module_parts(path)
    src_layer = layer_of(src_parts)
    found = []
    for library, target in imports_of(path):
        if target is None:
            if library in BACKEND_LIBRARIES and not (src_parts[0] == "models" and len(src_parts) > 2):
                found.append(f"imports backend library {library!r} outside models/<backend>/")
            continue
        target_layer = layer_of(target)
        if target_layer not in ALLOWED[src_layer]:
            found.append(f"{src_layer} imports {'.'.join(target)} ({target_layer})")
        elif src_layer == "usecases" == target_layer and target[1:2] != src_parts[1:2]:
            found.append(f"use case {src_parts[1]} imports another use case: {'.'.join(target)}")
        elif src_layer == "models" == target_layer and len(src_parts) > 2 and len(target) > 2 \
                and target[1] != src_parts[1]:
            found.append(f"backend {src_parts[1]} imports another backend: {'.'.join(target)}")
    return found


MODULES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_layer_rules(path):
    assert violations(path) == []


def test_checker_catches_violations(tmp_path, monkeypatch):
    fake_src = tmp_path / "vision_toolkit"
    (fake_src / "ops").mkdir(parents=True)
    (fake_src / "models" / "mediapipe").mkdir(parents=True)
    (fake_src / "models" / "mediapipe" / "x.py").write_text("")
    bad = fake_src / "ops" / "bad.py"
    bad.write_text("import mediapipe\nfrom vision_toolkit.models.mediapipe import x\nfrom ..usecases import y\n")
    monkeypatch.setattr(sys.modules[__name__], "SRC", fake_src)
    found = violations(bad)
    assert len(found) == 3, found


def test_architecture_doc_lists_every_module():
    doc = ARCHITECTURE.read_text(encoding="utf-8")
    missing = [p.relative_to(SRC).as_posix() for p in MODULES if p.name != "__init__.py" and p.name not in doc]
    assert missing == [], "add these modules to docs/architecture.md"
