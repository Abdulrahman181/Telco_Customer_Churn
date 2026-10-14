from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main  # noqa: E402


def test_python_sources_parse() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_missing_artifacts_are_reported_without_fallback(tmp_path: Path) -> None:
    missing = main.missing_artifacts(tmp_path)
    assert missing
    with pytest.raises(FileNotFoundError, match="Required model artifacts are missing"):
        main.load_artifacts(tmp_path)


def test_notebook_is_valid_json_and_has_no_stale_outputs() -> None:
    notebook = json.loads((ROOT / "Customer_Churn_Prediction.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "/content/" not in source
