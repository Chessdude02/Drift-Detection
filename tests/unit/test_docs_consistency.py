"""docs/execution.md must match the code: every repo path it mentions exists, and every
function it names as `name()` is defined somewhere in src/, scripts/ or tests/ (or is a
known library call). Fails the build when code changes without the doc."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "execution.md"

# Library / framework calls the doc names on purpose.
EXTERNAL = {
    "mlflow.pyfunc.log_model",
    "log_model",
    "mlflow.pyfunc.load_model",
    "load_model",
    "get_latest_versions",
    "MlflowClient.download_artifacts",
    "download_artifacts",
    "model.predict",
    "predict",
    "update",
    "GroupShuffleSplit",
}


def _defined_names() -> set[str]:
    names: set[str] = set()
    for folder in ("src", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
    return names


def test_paths_in_execution_doc_exist():
    text = DOC.read_text(encoding="utf-8")
    paths = set(
        re.findall(
            r"`((?:src|scripts|config|tests|deploy|docs|monitoring|data/holdout)/[^`*<> ]+)`", text
        )
    )
    paths |= {f"docs/{p}" for p in re.findall(r"\]\((call_graph/[^)]+)\)", text)}
    missing = sorted(p for p in paths if not (ROOT / p.rstrip("/")).exists())
    assert paths, "no paths found - regex broken?"
    assert missing == [], f"execution.md mentions paths that do not exist: {missing}"


def test_functions_in_execution_doc_exist():
    text = DOC.read_text(encoding="utf-8")
    called = set(re.findall(r"([A-Za-z_][\w.]*)\(\)", text))
    defined = _defined_names()
    missing = sorted(
        name for name in called if name not in EXTERNAL and name.split(".")[-1] not in defined
    )
    assert called, "no function names found - regex broken?"
    assert missing == [], f"execution.md names functions that do not exist: {missing}"
