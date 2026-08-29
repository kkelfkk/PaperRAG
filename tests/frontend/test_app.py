"""Streamlit component-tree smoke test for the PaperRAG interface."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_streamlit_app_renders_without_exceptions() -> None:
    app_path = Path(__file__).parents[2] / "frontend" / "app.py"

    rendered = streamlit_testing.AppTest.from_file(str(app_path)).run(timeout=20)

    assert not rendered.exception
    assert [title.value for title in rendered.title] == ["📚 PaperRAG"]
    assert [tab.label for tab in rendered.tabs] == ["上传论文", "检索与问答"]
