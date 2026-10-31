"""parse_batch.py：目录收集、批量主流程、跳过/限制/失败计数。"""

from __future__ import annotations

import json
import sys

import pytest

from oxr_starter import parse_batch
from tests.helpers import FakeBackend, make_pdf


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["oxr-batch", *map(str, args)])


@pytest.fixture
def backend_holder(monkeypatch):
    class Holder:
        backend = FakeBackend(name="local", reply="text")

        def make(self, *args, **kwargs):
            self.args = args
            return self.backend

    holder = Holder()
    monkeypatch.setattr(parse_batch, "make_backend", holder.make)
    return holder


@pytest.fixture
def papers(tmp_path, pdf_factory):
    root = tmp_path / "papers"
    make_pdf(root / "a.pdf", pages=1)
    make_pdf(root / "b.pdf", pages=2)
    make_pdf(root / "sub" / "c.pdf", pages=1)
    (root / "notes.txt").write_text("not a pdf", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# collect_pdfs
# --------------------------------------------------------------------------- #
def test_collect_pdfs_recursive_and_sorted(papers):
    found = parse_batch.collect_pdfs(papers, "*.pdf")
    assert [p.relative_to(papers).as_posix() for p in found] == [
        "a.pdf",
        "b.pdf",
        "sub/c.pdf",
    ]


def test_collect_pdfs_honours_pattern(papers):
    assert parse_batch.collect_pdfs(papers, "c*.pdf") == [papers / "sub" / "c.pdf"]
    assert parse_batch.collect_pdfs(papers, "*.txt") == [papers / "notes.txt"]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def test_batch_converts_every_pdf(monkeypatch, tmp_path, papers, backend_holder, capsys):
    out = tmp_path / "md"
    _argv(monkeypatch, papers, "-o", out, "--dpi", 72, "--json")

    parse_batch.main()

    assert (out / "a.md").read_text(encoding="utf-8").count("text") == 1
    assert (out / "b.md").read_text(encoding="utf-8").count("text") == 2
    assert (out / "sub" / "c.md").exists()
    data = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert data["source"] == "a.pdf"
    assert data["meta"] == {"dpi": 72, "backend": "local", "workers": 1}
    assert (out / "a.parts" / "p0001.md").exists()

    printed = capsys.readouterr().out
    assert "找到 3 个文件" in printed
    assert "成功 3，跳过 0，失败 0" in printed


def test_batch_skip_existing(monkeypatch, tmp_path, papers, backend_holder, capsys):
    out = tmp_path / "md"
    out.mkdir()
    (out / "a.md").write_text("old", encoding="utf-8")
    _argv(monkeypatch, papers, "-o", out, "--dpi", 72, "--skip-existing")

    parse_batch.main()

    assert (out / "a.md").read_text(encoding="utf-8") == "old"
    assert (out / "b.md").exists()
    assert len(backend_holder.backend.calls) == 3  # b 两页 + c 一页
    assert "跳过（已存在）" in capsys.readouterr().out


def test_batch_passes_max_pixels(monkeypatch, tmp_path, papers, backend_holder):
    _argv(monkeypatch, papers, "-o", tmp_path / "md", "--dpi", 72,
          "--max-pixels", "1200000")

    parse_batch.main()

    assert backend_holder.args[4] == 1_200_000


def test_batch_limit(monkeypatch, tmp_path, papers, backend_holder):
    out = tmp_path / "md"
    _argv(monkeypatch, papers, "-o", out, "--dpi", 72, "--limit", "1")

    parse_batch.main()

    assert (out / "a.md").exists()
    assert not (out / "b.md").exists()
    assert not (out / "sub").exists()


def test_batch_pages_filter(monkeypatch, tmp_path, papers):
    out = tmp_path / "md"
    backend = FakeBackend(name="local", reply="text")
    monkeypatch.setattr(parse_batch, "make_backend", lambda *a, **k: backend)
    _argv(monkeypatch, papers, "-o", out, "--dpi", 72, "--pages", "2")

    parse_batch.main()

    a_text = (out / "a.md").read_text(encoding="utf-8")
    assert "第 1 页" not in a_text
    assert (out / "b.md").read_text(encoding="utf-8").count("<!-- 第 2 页 -->") == 1


def test_batch_counts_failed_files(monkeypatch, tmp_path, papers, backend_holder, capsys):
    out = tmp_path / "md"
    # a.pdf 第 1 次调用失败；b.pdf 两次调用成功；sub/c.pdf 第 4 次调用失败
    backend_holder.backend = FakeBackend(name="local", reply="text", fail_calls=(1, 4))
    _argv(monkeypatch, papers, "-o", out, "--dpi", 72)

    with pytest.raises(SystemExit) as exc:
        parse_batch.main()

    assert exc.value.code == 1
    assert not (out / "a.md").exists()
    assert (out / "b.md").exists()
    assert not (out / "sub" / "c.md").exists()
    printed = capsys.readouterr().out
    assert "失败: a.pdf" in printed
    assert "失败: sub/c.pdf" in printed
    assert "成功 1，跳过 0，失败 2" in printed


def test_batch_no_pdfs(monkeypatch, tmp_path, backend_holder):
    empty = tmp_path / "empty"
    empty.mkdir()
    _argv(monkeypatch, empty, "-o", tmp_path / "md")

    with pytest.raises(SystemExit, match="没找到"):
        parse_batch.main()


def test_batch_input_not_directory(monkeypatch, tmp_path, backend_holder):
    _argv(monkeypatch, tmp_path / "missing", "-o", tmp_path / "md")

    with pytest.raises(SystemExit, match="不是目录"):
        parse_batch.main()


def test_batch_backend_error_becomes_system_exit(monkeypatch, papers, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("没有可用设备")

    monkeypatch.setattr(parse_batch, "make_backend", boom)
    _argv(monkeypatch, papers, "-o", tmp_path / "md")

    with pytest.raises(SystemExit, match="没有可用设备"):
        parse_batch.main()
