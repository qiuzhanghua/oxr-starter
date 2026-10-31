"""parse_pdf.py：分页、表格替换、并发、断点缓存与 CLI。

全部用 FakeBackend，不加载模型；PDF 由 PyMuPDF 现场合成。
"""

from __future__ import annotations

import json
import sys

import pymupdf
import pytest

from oxr_starter import parse_pdf
from tests.helpers import TABLE_PROMPT, FakeBackend


def _page(pdf_path):
    doc = pymupdf.open(pdf_path)
    return doc, doc[0]


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["oxr-parse", *map(str, args)])


# --------------------------------------------------------------------------- #
# 页码解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("pages", "total", "expected"),
    [
        ("", 5, [1, 2, 3, 4, 5]),
        ("1", 3, [1]),
        ("3,1", 3, [1, 3]),
        ("2-4", 6, [2, 3, 4]),
        ("4-2", 6, []),
        ("1-2,2,5", 5, [1, 2, 5]),
        ("0,7", 3, []),
        ("1", 0, []),
    ],
)
def test_parse_pageset(pages, total, expected):
    assert parse_pdf.parse_pageset(pages, total) == expected


def test_parse_pageset_rejects_non_numeric():
    with pytest.raises(ValueError):
        parse_pdf.parse_pageset("1-x", 3)


# --------------------------------------------------------------------------- #
# 渲染 / 裁剪
# --------------------------------------------------------------------------- #
def test_render_page_scales_with_dpi(pdf_factory):
    doc, page = _page(pdf_factory())
    try:
        assert parse_pdf.render_page(page, 72).size == (300, 400)
        big = parse_pdf.render_page(page, 144)
        assert big.size == (600, 800)
        assert big.mode == "RGB"
    finally:
        doc.close()


def test_crop_region_uses_given_box(pdf_factory):
    doc, page = _page(pdf_factory())
    try:
        img = parse_pdf.crop_region(page, [20, 40, 120, 140], 72)
        assert img.size == (100, 100)
        assert img.mode == "RGB"
    finally:
        doc.close()


def test_table_boxes_empty_for_plain_text_page(pdf_factory):
    doc, page = _page(pdf_factory())
    try:
        assert parse_pdf.table_boxes(page) == []
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# 表格替换
# --------------------------------------------------------------------------- #
def test_replace_tables_replaces_matching_blocks(pdf_factory, monkeypatch):
    doc, page = _page(pdf_factory())
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    backend = FakeBackend(table_reply="<table><tr><td>refined</td></tr></table>")
    markdown = "before\n<table><tr><td>raw</td></tr></table>\nafter"
    try:
        out, tables = parse_pdf.replace_tables(backend, page, markdown, 72, 32)
    finally:
        doc.close()

    assert "refined" in out
    assert "raw" not in out
    assert tables == [
        {"bbox": [10, 10, 110, 60], "html": "<table><tr><td>refined</td></tr></table>"}
    ]
    assert backend.calls[0].prompt == TABLE_PROMPT
    assert backend.calls[0].max_tokens == 32


def test_replace_tables_keeps_original_when_model_does_not_return_table(
    pdf_factory, monkeypatch
):
    doc, page = _page(pdf_factory())
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    backend = FakeBackend(table_reply="I cannot read this table")
    markdown = "<table><tr><td>raw</td></tr></table>"
    try:
        out, tables = parse_pdf.replace_tables(backend, page, markdown, 72, 32)
    finally:
        doc.close()

    assert out == markdown
    assert tables == []
    assert len(backend.calls) == 1  # 仍然尝试识别了


def test_replace_tables_noop_without_boxes(pdf_factory, monkeypatch):
    doc, page = _page(pdf_factory())
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [])
    backend = FakeBackend()
    try:
        out, tables = parse_pdf.replace_tables(backend, page, "<table>x</table>", 72, 32)
    finally:
        doc.close()

    assert out == "<table>x</table>"
    assert tables == []
    assert backend.calls == []


def test_replace_tables_only_uses_min_of_boxes_and_tables(pdf_factory, monkeypatch):
    doc, page = _page(pdf_factory())
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[0, 0, 10, 10], [10, 10, 20, 20]])
    backend = FakeBackend(table_reply="<table>new</table>")
    markdown = "<table>one</table>"
    try:
        out, tables = parse_pdf.replace_tables(backend, page, markdown, 72, 32)
    finally:
        doc.close()

    assert out == "<table>new</table>"
    assert len(tables) == 1
    assert len(backend.calls) == 1


def test_replace_tables_replaces_each_table(pdf_factory, monkeypatch):
    doc, page = _page(pdf_factory())
    boxes = [[0, 0, 10, 10], [10, 10, 20, 20]]
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: boxes)
    backend = FakeBackend(table_reply="<table>new</table>")
    markdown = "<table>one</table> and <table>two</table>"
    try:
        out, tables = parse_pdf.replace_tables(backend, page, markdown, 72, 32)
    finally:
        doc.close()

    assert out.count("<table>new</table>") == 2
    assert [t["bbox"] for t in tables] == boxes


def test_parse_one_strips_text_and_skips_tables_when_disabled(pdf_factory):
    doc, page = _page(pdf_factory())
    backend = FakeBackend(reply="  text  ")
    try:
        text, tables = parse_pdf._parse_one(backend, page, 72, 16, parse_tables=False)
    finally:
        doc.close()
    assert (text, tables) == ("text", [])
    assert len(backend.calls) == 1


# --------------------------------------------------------------------------- #
# process_pages
# --------------------------------------------------------------------------- #
def test_process_pages_serial(pdf_factory):
    pdf = pdf_factory(pages=3)
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="local", reply="same")
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2, 3], 72, 16, False, workers=1
        )
    finally:
        doc.close()

    assert failed == []
    assert results == {1: ("same", []), 2: ("same", []), 3: ("same", [])}
    assert all(c.thread == "MainThread" for c in backend.calls)


def test_process_pages_isolates_page_failure(pdf_factory):
    pdf = pdf_factory(pages=3)
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="local", reply="ok", fail_calls=(2,))
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2, 3], 72, 16, False, workers=1
        )
    finally:
        doc.close()

    assert failed == [2]
    assert set(results) == {1, 3}


def test_process_pages_local_ignores_workers(pdf_factory):
    pdf = pdf_factory(pages=2)
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="local", reply="same")
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2], 72, 16, False, workers=4
        )
    finally:
        doc.close()

    assert failed == []
    assert set(results) == {1, 2}
    assert all(c.thread == "MainThread" for c in backend.calls)


def test_process_pages_parallel_uses_thread_pool(pdf_factory):
    pdf = pdf_factory(pages=3)
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="vllm", reply="same")
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2, 3], 72, 16, False, workers=2
        )
    finally:
        doc.close()

    assert failed == []
    assert set(results) == {1, 2, 3}
    assert any(c.thread.startswith("ThreadPoolExecutor") for c in backend.calls)


def test_process_pages_parallel_reports_every_failure(pdf_factory):
    pdf = pdf_factory(pages=3)
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="vllm", fail_calls=(1, 2, 3))
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2, 3], 72, 16, False, workers=2
        )
    finally:
        doc.close()

    assert results == {}
    assert sorted(failed) == [1, 2, 3]


def test_process_pages_parallel_refines_tables(pdf_factory, monkeypatch):
    pdf = pdf_factory(pages=2)
    doc = pymupdf.open(pdf)
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    backend = FakeBackend(
        name="vllm",
        reply="<table><tr><td>raw</td></tr></table>",
        table_reply="<table><tr><td>refined</td></tr></table>",
    )
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1, 2], 72, 16, True, workers=2
        )
    finally:
        doc.close()

    assert failed == []
    for text, tables in results.values():
        assert "refined" in text
        assert tables == [
            {
                "bbox": [10, 10, 110, 60],
                "html": "<table><tr><td>refined</td></tr></table>",
            }
        ]


def test_process_pages_serial_refines_tables(pdf_factory, monkeypatch):
    pdf = pdf_factory(pages=1)
    doc = pymupdf.open(pdf)
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    backend = FakeBackend(
        name="local",
        reply="<table>raw</table>",
        table_reply="<table>refined</table>",
    )
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1], 72, 16, True, workers=1
        )
    finally:
        doc.close()

    assert failed == []
    assert results[1][0] == "<table>refined</table>"
    assert len(results[1][1]) == 1


# --------------------------------------------------------------------------- #
# 断点缓存
# --------------------------------------------------------------------------- #
def test_process_pages_writes_and_resumes_parts(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=2)
    parts = tmp_path / "doc.parts"
    doc = pymupdf.open(pdf)
    try:
        first = FakeBackend(name="local", reply="first")
        results, _ = parse_pdf.process_pages(
            first, doc, [1, 2], 72, 16, False, parts_dir=parts
        )
        assert set(results) == {1, 2}
        assert (parts / "p0001.md").read_text(encoding="utf-8") == "first"

        second = FakeBackend(name="local", reply="second")
        results2, failed = parse_pdf.process_pages(
            second, doc, [1, 2], 72, 16, False, parts_dir=parts
        )
    finally:
        doc.close()

    assert second.calls == []  # 全部命中缓存
    assert failed == []
    assert results2 == results


def test_process_pages_resume_loads_table_json(pdf_factory, tmp_path, monkeypatch):
    pdf = pdf_factory(pages=1)
    parts = tmp_path / "doc.parts"
    parts.mkdir()
    (parts / "p0001.md").write_text("<table>refined</table>", encoding="utf-8")
    (parts / "p0001.json").write_text(
        json.dumps({"tables": [{"bbox": [1, 2, 3, 4], "html": "<table>refined</table>"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[9, 9, 9, 9]])
    doc = pymupdf.open(pdf)
    backend = FakeBackend(name="local")
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1], 72, 16, True, parts_dir=parts
        )
    finally:
        doc.close()

    assert backend.calls == []
    assert failed == []
    assert results[1] == (
        "<table>refined</table>",
        [{"bbox": [1, 2, 3, 4], "html": "<table>refined</table>"}],
    )


def test_process_pages_resume_tolerates_corrupt_table_json(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=1)
    parts = tmp_path / "doc.parts"
    parts.mkdir()
    (parts / "p0001.md").write_text("cached", encoding="utf-8")
    (parts / "p0001.json").write_text("{broken", encoding="utf-8")
    doc = pymupdf.open(pdf)
    try:
        results, failed = parse_pdf.process_pages(
            FakeBackend(name="local"), doc, [1], 72, 16, True, parts_dir=parts
        )
    finally:
        doc.close()

    assert failed == []
    assert results[1] == ("cached", [])


def test_process_pages_without_parts_dir_writes_nothing(pdf_factory, tmp_path):
    pdf = pdf_factory(pages=1)
    doc = pymupdf.open(pdf)
    try:
        parse_pdf.process_pages(FakeBackend(name="local"), doc, [1], 72, 16, False)
    finally:
        doc.close()

    assert list(tmp_path.glob("*.parts")) == []


# --------------------------------------------------------------------------- #
# build_json
# --------------------------------------------------------------------------- #
def test_build_json_marks_missing_pages_as_failed(tmp_path):
    data = parse_pdf.build_json(
        tmp_path / "doc.pdf",
        [1, 2],
        {1: ("text", [{"bbox": [1], "html": "<table>x</table>"}])},
        {1: [{"file": "a.png"}]},
        {"dpi": 150},
    )

    assert data["source"] == "doc.pdf"
    assert data["meta"] == {"dpi": 150}
    assert data["pages"][0] == {
        "page": 1,
        "markdown": "text",
        "tables": [{"bbox": [1], "html": "<table>x</table>"}],
        "images": [{"file": "a.png"}],
        "failed": False,
    }
    assert data["pages"][1] == {
        "page": 2,
        "markdown": "",
        "tables": [],
        "images": [],
        "failed": True,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture
def patched_backend(monkeypatch):
    """把 CLI 的 make_backend 换掉，并返回一个可改属性的 holder。"""

    class Holder:
        backend = FakeBackend(name="local", reply="page")

        def make(self, *args, **kwargs):
            self.args = args
            return self.backend

    holder = Holder()
    monkeypatch.setattr(parse_pdf, "make_backend", holder.make)
    return holder


def test_cli_writes_markdown_json_and_parts(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=2)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--json")

    parse_pdf.main()

    text = out.read_text(encoding="utf-8")
    assert "# doc" in text
    assert "<!-- 第 1 页 -->" in text and "<!-- 第 2 页 -->" in text
    assert text.count("page") == 2

    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [p["page"] for p in data["pages"]] == [1, 2]
    assert data["pages"][0]["markdown"] == "page"
    assert data["meta"]["backend"] == "local"
    assert data["meta"]["dpi"] == 72
    assert data["meta"]["workers"] == 1

    parts = tmp_path / "out.parts"
    assert (parts / "p0001.md").exists() and (parts / "p0002.md").exists()


def test_cli_passes_max_pixels(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory()
    _argv(monkeypatch, pdf, "-o", tmp_path / "out.md", "--dpi", 72,
          "--max-pixels", "1200000")

    parse_pdf.main()

    assert patched_backend.args[4] == 1_200_000


def test_cli_default_output_next_to_pdf(monkeypatch, pdf_factory, patched_backend):
    pdf = pdf_factory(name="paper.pdf")
    _argv(monkeypatch, pdf, "--dpi", 72)

    parse_pdf.main()

    assert (pdf.parent / "paper.md").exists()


def test_cli_resume_skips_finished_pages(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=2)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72)
    parse_pdf.main()

    second = FakeBackend(name="local", reply="page")
    patched_backend.backend = second
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72)
    parse_pdf.main()

    assert second.calls == []


def test_cli_no_resume_ignores_cache(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=1)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--no-resume")

    parse_pdf.main()

    assert not (tmp_path / "out.parts").exists()


def test_cli_clean_parts_removes_cache(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=1)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--clean-parts")

    parse_pdf.main()

    assert not (tmp_path / "out.parts").exists()


def test_cli_partial_failure_exits_1_and_keeps_output(
    monkeypatch, pdf_factory, tmp_path, patched_backend
):
    pdf = pdf_factory(pages=2)
    out = tmp_path / "out.md"
    patched_backend.backend = FakeBackend(name="local", reply="page", fail_calls=(2,))
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--json")

    with pytest.raises(SystemExit) as exc:
        parse_pdf.main()

    assert exc.value.code == 1
    text = out.read_text(encoding="utf-8")
    assert "第 2 页 识别失败" in text
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["pages"][1]["failed"] is True
    assert not (tmp_path / "out.parts" / "p0002.md").exists()


def test_cli_pages_filter(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=3)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--pages", "2", "--json")

    parse_pdf.main()

    text = out.read_text(encoding="utf-8")
    assert "<!-- 第 2 页 -->" in text
    assert "<!-- 第 1 页 -->" not in text
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert [p["page"] for p in data["pages"]] == [2]


def test_cli_parse_tables(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=1)
    out = tmp_path / "out.md"
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    patched_backend.backend = FakeBackend(
        name="local", reply="<table>raw</table>", table_reply="<table>refined</table>"
    )
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--parse-tables", "--json")

    parse_pdf.main()

    assert "<table>refined</table>" in out.read_text(encoding="utf-8")
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["pages"][0]["tables"][0]["bbox"] == [10, 10, 110, 60]


def test_cli_extract_images(monkeypatch, pdf_factory, tmp_path, patched_backend):
    pdf = pdf_factory(pages=1, images=True)
    out = tmp_path / "out.md"
    _argv(monkeypatch, pdf, "-o", out, "--dpi", 72, "--extract-images", "--json")

    parse_pdf.main()

    images_dir = tmp_path / "out.images"
    files = sorted(images_dir.glob("*.png"))
    assert len(files) == 1
    text = out.read_text(encoding="utf-8")
    assert f"out.images/{files[0].name}" in text
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["pages"][0]["images"][0]["file"] == files[0].name


def test_cli_missing_input(monkeypatch, tmp_path, patched_backend):
    _argv(monkeypatch, tmp_path / "nope.pdf")
    with pytest.raises(SystemExit, match="文件不存在"):
        parse_pdf.main()


def test_cli_backend_error_becomes_system_exit(monkeypatch, pdf_factory, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("没有可用设备")

    pdf = pdf_factory()
    monkeypatch.setattr(parse_pdf, "make_backend", boom)
    _argv(monkeypatch, pdf, "-o", tmp_path / "out.md", "--dpi", 72)

    with pytest.raises(SystemExit, match="没有可用设备"):
        parse_pdf.main()


def test_table_boxes_swallows_pymupdf_errors():
    class BrokenPage:
        def find_tables(self):
            raise ValueError("no table support")

    assert parse_pdf.table_boxes(BrokenPage()) == []


def test_process_pages_parallel_keeps_text_when_table_recognition_fails(
    pdf_factory, monkeypatch
):
    pdf = pdf_factory(pages=1)
    doc = pymupdf.open(pdf)
    monkeypatch.setattr(parse_pdf, "table_boxes", lambda p: [[10, 10, 110, 60]])
    backend = FakeBackend(
        name="vllm", reply="<table>raw</table>", fail_tables=True
    )
    try:
        results, failed = parse_pdf.process_pages(
            backend, doc, [1], 72, 16, True, workers=2
        )
    finally:
        doc.close()

    assert failed == []
    assert results[1] == ("<table>raw</table>", [])
