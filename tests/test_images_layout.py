"""images.py（位图提取）与 layout.py（区域检测 / 表格识别）。"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pymupdf

from oxr_starter import backend as backend_module
from oxr_starter import images, layout
from tests.helpers import png_bytes


# --------------------------------------------------------------------------- #
# images.py
# --------------------------------------------------------------------------- #
def test_extract_document_writes_files_and_manifest(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=2, images=True)
    out = tmp_path / "imgs"

    items = images.extract_document(pdf, out)

    assert [it["file"] for it in items] == ["p0001-img01.png", "p0002-img01.png"]
    assert [it["page"] for it in items] == [1, 2]
    for item in items:
        path = out / item["file"]
        assert path.is_file()
        assert item["bytes"] == path.stat().st_size
        assert (item["width"], item["height"]) == (120, 80)
        assert item["bbox"] == [60.0, 200.0, 180.0, 280.0]
        assert item["colorspace"] is not None

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"pdf": "doc.pdf", "images": items}


def test_extract_document_without_bitmaps_writes_no_manifest(pdf_factory, tmp_path):
    out = tmp_path / "imgs"
    assert images.extract_document(pdf_factory(), out) == []
    assert not (out / "manifest.json").exists()


def test_extract_page_images_filters_tiny_images(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=1, images=True, tiny_image=True)
    out = tmp_path / "imgs"
    out.mkdir()

    items = images.extract_page_images(pymupdf.open(pdf), 1, out)

    assert [it["file"] for it in items] == ["p0001-img01.png"]


def test_extract_page_images_honours_min_area(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=1, images=True)
    out = tmp_path / "imgs"
    out.mkdir()

    items = images.extract_page_images(
        pymupdf.open(pdf), 1, out, min_area_ratio=0.5
    )

    assert items == []


def test_extract_page_images_deduplicates_same_xref(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=400)
    payload = png_bytes()
    page.insert_image(pymupdf.Rect(20, 20, 140, 100), stream=payload)
    page.insert_image(pymupdf.Rect(160, 200, 280, 280), stream=payload)
    pdf = tmp_path / "dup.pdf"
    doc.save(pdf)
    doc.close()
    out = tmp_path / "imgs"
    out.mkdir()

    items = images.extract_page_images(pymupdf.open(pdf), 1, out)

    assert len(items) == 1


def test_images_cli(monkeypatch, tmp_path, pdf_factory, capsys):
    pdf = pdf_factory(pages=1, images=True)
    out = tmp_path / "imgs"
    monkeypatch.setattr(sys, "argv", ["oxr-images", str(pdf), "-o", str(out)])

    images.main()

    assert len(list(out.glob("*.png"))) == 1
    assert "提取 1 张图片" in capsys.readouterr().out


def test_images_cli_reports_vector_only(monkeypatch, tmp_path, pdf_factory, capsys):
    pdf = pdf_factory(pages=1)
    monkeypatch.setattr(sys, "argv", ["oxr-images", str(pdf), "-o", str(tmp_path / "imgs")])

    images.main()

    assert "没有可提取的位图" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# layout.py
# --------------------------------------------------------------------------- #
class FakeTable:
    def __init__(self, bbox=(10, 20, 90, 80), rows=(1, 2, 3), names=("a", "b")):
        self.bbox = bbox
        self.rows = list(rows)
        self.header = SimpleNamespace(names=list(names)) if names is not None else None


class FakePage:
    def __init__(self, tables=()):
        self._tables = list(tables)

    def get_text(self, kind):
        assert kind == "blocks"
        return [
            (0, 0, 100, 50, "hello\nworld", 0, 0),
            (0, 60, 100, 80, "", 0, 1),
        ]

    def find_tables(self):
        return self._tables


def test_detect_regions_types_order_and_previews():
    regions = layout.detect_regions(FakePage([FakeTable()]))

    assert [r["type"] for r in regions] == ["text", "table", "image"]
    assert regions[0]["box"] == [0, 0, 100, 50]
    assert regions[0]["preview"] == "hello world"
    assert regions[1]["preview"] == "3 行 × 2 列"
    assert regions[2]["preview"] == ""


def test_detect_regions_handles_table_without_header():
    regions = layout.detect_regions(FakePage([FakeTable(names=None)]))
    table = next(r for r in regions if r["type"] == "table")
    assert table["preview"] == "3 行 × 0 列"


def test_detect_regions_ignores_table_detection_errors():
    class Broken(FakePage):
        def find_tables(self):
            raise ValueError("no tables here")

    regions = layout.detect_regions(Broken())
    assert [r["type"] for r in regions] == ["text", "image"]


def test_layout_crop_renders_requested_box(pdf_factory):
    pdf = pdf_factory()
    doc = pymupdf.open(pdf)
    try:
        img = layout.crop(doc[0], [10, 10, 110, 60], 72)
    finally:
        doc.close()
    assert img.size == (100, 50)
    assert img.mode == "RGB"


def test_layout_cli_json(monkeypatch, pdf_factory, capsys):
    pdf = pdf_factory()
    monkeypatch.setattr(sys, "argv", ["oxr-layout", str(pdf), "--page", "1", "--json"])

    layout.main()

    data = json.loads(capsys.readouterr().out)
    assert data["page"] == 1
    assert data["regions"][0]["type"] == "text"


def test_layout_cli_text(monkeypatch, pdf_factory, capsys):
    pdf = pdf_factory()
    monkeypatch.setattr(sys, "argv", ["oxr-layout", str(pdf), "--page", "1"])

    layout.main()

    out = capsys.readouterr().out
    assert "doc.pdf 第 1 页" in out
    assert "个区域" in out


def test_layout_cli_parse_tables_uses_local_backend(monkeypatch, pdf_factory, capsys):
    pdf = pdf_factory()
    created = {}

    class FakeLocal:
        name = "local"

        def __init__(self, **kwargs):
            created.update(kwargs)
            self.device = "cpu"
            self.dtype = "torch.float32"
            self.calls = []
            self.closed = False

        def recognize(self, image, prompt, max_tokens):
            self.calls.append((image.size, prompt, max_tokens))
            return "<table>refined</table>"

        def close(self):
            self.closed = True

    backend = FakeLocal(device="cpu", dtype="fp32")
    monkeypatch.setattr(backend_module, "LocalBackend", lambda **kwargs: backend)
    monkeypatch.setattr(
        layout,
        "detect_regions",
        lambda page: [{"type": "table", "box": [10, 10, 60, 60], "preview": "1 行 × 1 列"}],
    )
    monkeypatch.setattr(
        sys, "argv", ["oxr-layout", str(pdf), "--page", "1", "--parse-tables", "--max-tokens", "8", "--dpi", "72"]
    )

    layout.main()

    out = capsys.readouterr().out
    assert "<table>refined</table>" in out
    assert backend.closed is True
    assert backend.calls[0][0] == (50, 50)
    assert backend.calls[0][2] == 8
