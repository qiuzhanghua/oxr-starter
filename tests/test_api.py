"""api.py：Document 数据类型、OXRClient 与一次性 parse()。"""

from __future__ import annotations

import json

import pymupdf
import pytest

from oxr_starter import api, parse_pdf
from oxr_starter.api import Document, ImageRef, OXRClient, Page, Region
from tests.helpers import FakeBackend


def patch_backend(monkeypatch, backend):
    """替换 api.make_backend，返回记录到的构造参数列表。"""
    calls = []

    def fake(kind="local", base_url="http://127.0.0.1:8001/v1", device=None,
             dtype=None, max_pixels=None):
        calls.append({"kind": kind, "base_url": base_url, "device": device,
                      "dtype": dtype, "max_pixels": max_pixels})
        return backend

    monkeypatch.setattr(api, "make_backend", fake)
    return calls


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def test_document_markdown_contains_pages_images_and_failures():
    doc = Document(
        source="paper.pdf",
        pages=[
            Page(
                number=1,
                markdown="# hi",
                tables=[Region(bbox=[0, 0, 1, 1], html="<table/>")],
                images=[ImageRef(file="a.png", bbox=[0, 0, 1, 1], width=1, height=1, bytes=10)],
            ),
            Page(number=2, markdown="", failed=True),
        ],
        meta={"dpi": 150},
    )

    md = doc.markdown
    assert md.startswith("# paper\n")
    assert "<!-- 第 1 页 -->" in md
    assert "# hi" in md
    assert "![a.png](a.png)" in md
    assert "<!-- 第 2 页 识别失败 -->" in md
    assert doc.failed_pages == [2]


def test_document_markdown_prefixes_images_dir(tmp_path):
    doc = Document(
        source="paper.pdf",
        pages=[Page(1, "x", images=[ImageRef(file="a.png", bbox=[0, 0, 1, 1])])],
        images_dir=tmp_path / "paper.images",
    )
    assert "![a.png](paper.images/a.png)" in doc.markdown


def test_document_to_dict_and_json():
    doc = Document(
        source="中文.pdf",
        pages=[
            Page(
                number=1,
                markdown="正文",
                tables=[Region(bbox=[1, 2, 3, 4], html="<table>x</table>")],
                images=[ImageRef(file="a.png", bbox=[0, 0, 1, 1], width=2, height=3, bytes=9)],
            )
        ],
        meta={"dpi": 150},
    )

    data = doc.to_dict()
    assert data["source"] == "中文.pdf"
    assert data["meta"] == {"dpi": 150}
    assert data["pages"][0] == {
        "page": 1,
        "markdown": "正文",
        "tables": [{"bbox": [1, 2, 3, 4], "html": "<table>x</table>"}],
        "images": [{"file": "a.png", "bbox": [0, 0, 1, 1], "width": 2, "height": 3, "bytes": 9}],
        "failed": False,
    }
    assert json.loads(doc.to_json()) == data
    assert '"正文"' in doc.to_json()  # ensure_ascii=False


def test_document_save_writes_markdown_and_optional_json(tmp_path):
    doc = Document(source="paper.pdf", pages=[Page(1, "hello")])

    out = doc.save(tmp_path / "out.md")
    assert out == tmp_path / "out.md"
    assert out.read_text(encoding="utf-8") == doc.markdown
    assert not (tmp_path / "out.json").exists()

    doc.save(tmp_path / "out.md", json_path=True)
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == doc.to_dict()


# --------------------------------------------------------------------------- #
# OXRClient 构造
# --------------------------------------------------------------------------- #
def test_client_passes_device_options(monkeypatch):
    backend = FakeBackend(name="local")
    calls = patch_backend(monkeypatch, backend)

    client = OXRClient(
        backend="local", base_url="http://h/v1", workers=4, device="cpu", dtype="fp16"
    )

    assert client.workers == 1  # local 后端强制串行
    assert calls == [
        {
            "kind": "local",
            "base_url": "http://h/v1",
            "device": "cpu",
            "dtype": "fp16",
            "max_pixels": None,
        }
    ]


def test_client_passes_max_pixels(monkeypatch):
    calls = patch_backend(monkeypatch, FakeBackend(name="local"))

    OXRClient(backend="local", max_pixels=1_200_000)

    assert calls[0]["max_pixels"] == 1_200_000


def test_client_keeps_workers_for_vllm(monkeypatch):
    patch_backend(monkeypatch, FakeBackend(name="vllm"))
    assert OXRClient(backend="vllm", workers=4).workers == 4


def test_client_close_delegates_to_backend(monkeypatch):
    backend = FakeBackend(name="vllm")
    patch_backend(monkeypatch, backend)

    with OXRClient(backend="vllm") as client:
        assert client.workers == 1

    assert backend.closed is True


# --------------------------------------------------------------------------- #
# OXRClient.parse
# --------------------------------------------------------------------------- #
def test_client_parse_selected_pages(monkeypatch, pdf_factory):
    backend = FakeBackend(name="vllm", reply="text")
    patch_backend(monkeypatch, backend)
    client = OXRClient(backend="vllm", workers=2)

    doc = client.parse(pdf_factory(pages=3), pages="1,3", dpi=72, max_tokens=32)

    assert [p.number for p in doc.pages] == [1, 3]
    assert [p.markdown for p in doc.pages] == ["text", "text"]
    assert doc.source == "doc.pdf"
    assert doc.meta == {
        "dpi": 72,
        "backend": "vllm",
        "workers": 2,
        "parse_tables": False,
        "failed": [],
    }
    assert backend.calls[0].max_tokens == 32


def test_client_parse_marks_failed_pages(monkeypatch, pdf_factory):
    backend = FakeBackend(name="local", reply="ok", fail_calls=(2,))
    patch_backend(monkeypatch, backend)
    client = OXRClient(backend="local")

    doc = client.parse(pdf_factory(pages=3), dpi=72)

    assert doc.failed_pages == [2]
    assert doc.pages[1].failed is True
    assert doc.pages[1].markdown == ""
    assert doc.meta["failed"] == [2]
    assert "<!-- 第 2 页 识别失败 -->" in doc.markdown


def test_client_parse_replaces_tables(monkeypatch, pdf_factory):
    backend = FakeBackend(
        name="local",
        reply="<table>raw</table>",
        table_reply="<table>refined</table>",
    )
    patch_backend(monkeypatch, backend)
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda page: [[10, 10, 110, 60]])

    doc = OXRClient(backend="local").parse(pdf_factory(), dpi=72, parse_tables=True)

    page = doc.pages[0]
    assert page.markdown == "<table>refined</table>"
    assert page.tables == [Region(bbox=[10, 10, 110, 60], html="<table>refined</table>")]
    assert doc.meta["parse_tables"] is True


def test_client_parse_reuses_parts_dir(monkeypatch, pdf_factory, tmp_path):
    backend = FakeBackend(name="local", reply="cached")
    calls = patch_backend(monkeypatch, backend)
    client = OXRClient(backend="local")
    pdf = pdf_factory(pages=2)
    parts = tmp_path / "cache"

    first = client.parse(pdf, dpi=72, parts_dir=parts)
    assert backend.closed is False
    assert {p.stem for p in parts.glob("p*.md")} == {"p0001", "p0002"}

    second_backend = FakeBackend(name="local", reply="new")
    calls.clear()
    patch_backend(monkeypatch, second_backend)
    second = OXRClient(backend="local").parse(pdf, dpi=72, parts_dir=parts)

    assert second_backend.calls == []
    assert [p.markdown for p in second.pages] == [p.markdown for p in first.pages] == ["cached", "cached"]


def test_client_parse_resume_uses_default_parts_dir(monkeypatch, pdf_factory):
    backend = FakeBackend(name="local", reply="cached")
    patch_backend(monkeypatch, backend)
    client = OXRClient(backend="local")
    pdf = pdf_factory()

    doc = client.parse(pdf, dpi=72, resume=True)

    assert doc.meta["dpi"] == 72
    assert (pdf.parent / "doc.parts" / "p0001.md").exists()


def test_client_parse_extracts_images(monkeypatch, pdf_factory):
    backend = FakeBackend(name="local", reply="text")
    patch_backend(monkeypatch, backend)
    pdf = pdf_factory(pages=1, images=True)

    doc = OXRClient(backend="local").parse(pdf, dpi=72, extract_images=True)

    assert doc.images_dir is not None and doc.images_dir.name == "doc.images"
    assert doc.images_dir.is_dir()
    images = doc.pages[0].images
    assert len(images) == 1
    assert images[0].file.startswith("p0001-img01")
    assert images[0].width == 120 and images[0].height == 80
    assert images[0].bytes and images[0].bytes > 0
    assert f"doc.images/{images[0].file}" in doc.markdown


def test_client_parse_missing_file(monkeypatch, tmp_path):
    patch_backend(monkeypatch, FakeBackend(name="vllm"))
    with pytest.raises(FileNotFoundError):
        OXRClient(backend="vllm").parse(tmp_path / "missing.pdf")


# --------------------------------------------------------------------------- #
# 一次性 parse()
# --------------------------------------------------------------------------- #
def test_parse_creates_and_closes_client(monkeypatch, pdf_factory):
    backend = FakeBackend(name="vllm", reply="text")
    calls = patch_backend(monkeypatch, backend)

    doc = api.parse(
        pdf_factory(), backend="vllm", workers=2, dpi=72, max_tokens=16
    )

    assert doc.pages[0].markdown == "text"
    assert doc.meta["workers"] == 2
    assert calls[0]["kind"] == "vllm"
    assert backend.closed is True


def test_parse_accepts_max_pixels(monkeypatch, pdf_factory):
    calls = patch_backend(monkeypatch, FakeBackend(name="local", reply="x"))

    api.parse(pdf_factory(), backend="local", max_pixels=900000, dpi=72)

    assert calls[0]["max_pixels"] == 900000


def test_parse_accepts_device_options(monkeypatch, pdf_factory):
    calls = patch_backend(monkeypatch, FakeBackend(name="local", reply="x"))

    api.parse(pdf_factory(), backend="local", device="cpu", dtype="fp32", dpi=72)

    assert calls[0]["device"] == "cpu"
    assert calls[0]["dtype"] == "fp32"


def test_client_parse_closes_pdf(monkeypatch, pdf_factory):
    real_open = pymupdf.open
    closed = []

    class TrackedDoc:
        def __init__(self, doc):
            self._doc = doc

        def __getattr__(self, name):
            return getattr(self._doc, name)

        def __getitem__(self, index):
            return self._doc[index]

        def close(self):
            closed.append(True)
            self._doc.close()

    pdf = pdf_factory()
    monkeypatch.setattr(pymupdf, "open", lambda path: TrackedDoc(real_open(path)))
    patch_backend(monkeypatch, FakeBackend(name="local", reply="x"))

    OXRClient(backend="local").parse(pdf, dpi=72)

    assert closed == [True]
