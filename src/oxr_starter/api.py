"""Python API：把 CLI 的能力包成可复用的函数与数据类型。

基本用法：

    from oxr_starter.api import parse

    doc = parse("paper.pdf", backend="vllm", workers=4, parse_tables=True)
    print(doc.markdown)              # 整份 markdown
    print(doc.pages[0].markdown)     # 单页文本
    print(doc.pages[0].tables)       # [Region(bbox=[...], html="<table>…")]
    doc.save("out.md")               # 落盘（可同时写 json）

批量时可复用后端，避免重复建连接：

    from oxr_starter.api import OXRClient

    with OXRClient(backend="vllm", workers=4) as client:
        for p in pdfs:
            doc = client.parse(p, parse_tables=True)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .backend import make_backend
from .parse_pdf import (
    PROMPT_WHOLE_PAGE,
    build_json,
    parse_pageset,
    process_pages,
)


@dataclass
class Region:
    """页面上的一个结构化区域（目前是表格）。"""
    bbox: list[float]
    html: str


@dataclass
class ImageRef:
    """提取出的位图。"""
    file: str
    bbox: list[float]
    width: int | None = None
    height: int | None = None
    bytes: int | None = None


@dataclass
class Page:
    number: int
    markdown: str
    tables: list[Region] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    failed: bool = False


@dataclass
class Document:
    source: str
    pages: list[Page]
    meta: dict = field(default_factory=dict)
    images_dir: Path | None = None

    @property
    def markdown(self) -> str:
        """整份文档的 markdown（带页分隔注释）。"""
        parts = [f"# {Path(self.source).stem}\n"]
        for p in self.pages:
            if p.failed:
                parts.append(f"\n<!-- 第 {p.number} 页 识别失败 -->\n")
                continue
            parts.append(f"\n<!-- 第 {p.number} 页 -->\n\n{p.markdown}\n")
            for img in p.images:
                rel = f"{self.images_dir.name}/{img.file}" if self.images_dir else img.file
                parts.append(f"\n![{img.file}]({rel})\n")
        return "".join(parts)

    @property
    def failed_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.failed]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "pages": [
                {
                    "page": p.number,
                    "markdown": p.markdown,
                    "tables": [asdict(t) for t in p.tables],
                    "images": [asdict(i) for i in p.images],
                    "failed": p.failed,
                }
                for p in self.pages
            ],
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path, *, json_path: bool = False) -> Path:
        out = Path(path)
        out.write_text(self.markdown, encoding="utf-8")
        if json_path:
            out.with_suffix(".json").write_text(self.to_json(), encoding="utf-8")
        return out


class OXRClient:
    """持有后端的客户端，可复用于多份文档。"""

    def __init__(self, backend: str = "vllm",
                 base_url: str = "http://127.0.0.1:8001/v1",
                 workers: int = 1,
                 device: str = "auto",
                 dtype: str = "auto"):
        self.backend = make_backend(backend, base_url, device, dtype)
        self.workers = workers if self.backend.name == "vllm" else 1

    def parse(self, pdf: str | Path, *, dpi: int = 150, max_tokens: int = 2048,
              parse_tables: bool = False, extract_images: bool = False,
              pages: str = "", resume: bool = False,
              parts_dir: str | Path | None = None) -> Document:
        pdf_path = Path(pdf).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        doc = pymupdf.open(pdf_path)
        indices = parse_pageset(pages, doc.page_count)
        results, failed = process_pages(
            self.backend, doc, indices, dpi, max_tokens, parse_tables,
            self.workers, Path(parts_dir) if parts_dir else None,
        )

        image_map: dict[int, list[dict]] = {}
        images_dir = None
        if extract_images:
            from .images import extract_document

            images_dir = pdf_path.parent / f"{pdf_path.stem}.images"
            for it in extract_document(pdf_path, images_dir):
                image_map.setdefault(it["page"], []).append(it)

        pages_out: list[Page] = []
        for i in indices:
            if i not in results:
                pages_out.append(Page(number=i, markdown="", failed=True))
                continue
            text, tables = results[i]
            pages_out.append(Page(
                number=i,
                markdown=text,
                tables=[Region(bbox=t["bbox"], html=t["html"]) for t in tables],
                images=[ImageRef(**{k: it[k] for k in ("file", "bbox", "width", "height", "bytes")})
                        for it in image_map.get(i, [])],
            ))

        meta = {"dpi": dpi, "backend": self.backend.name, "workers": self.workers,
                "parse_tables": parse_tables, "failed": failed}
        return Document(source=pdf_path.name, pages=pages_out, meta=meta,
                        images_dir=images_dir)

    def close(self) -> None:
        """释放本地模型（vllm 后端无需操作）。"""
        self.backend.close()

    def __enter__(self) -> "OXRClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def parse(pdf: str | Path, **kwargs) -> Document:
    """一次性解析：内部创建并丢弃客户端。"""
    client_kwargs = {k: kwargs.pop(k) for k in ("backend", "base_url", "workers", "device", "dtype")
                     if k in kwargs}
    with OXRClient(**client_kwargs) as client:
        return client.parse(pdf, **kwargs)
