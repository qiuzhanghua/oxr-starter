"""pytest 共享夹具。"""

from __future__ import annotations

import pytest

from tests.helpers import make_pdf


@pytest.fixture
def pdf_factory(tmp_path):
    """在 tmp_path 下生成 PDF：`pdf_factory(pages=2, images=True)`。"""

    def _make(name: str = "doc.pdf", **kwargs):
        return make_pdf(tmp_path / name, **kwargs)

    return _make
