"""PDF → Markdown：逐页渲染后用 OXR 识别。

支持两种后端（`--backend`）：
  local  进程内加载 transformers（用完即走，占 3.9 GB 显存）
  vllm   调用已启动的 vLLM 服务（快约 20 倍，需先启动服务）

并发：`--workers N` 只对 vllm 后端生效——pymupdf 的渲染仍在主线程，
识别请求并行发出。
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from PIL import Image

from .backend import make_backend

PROMPT_WHOLE_PAGE = "Recognize the whole page content of the image."
PROMPT_TABLE = "Recognize the table of the image into html format."
TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """把 PDF 页渲染成 PIL 图像。"""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def crop_region(page: pymupdf.Page, box, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*box))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def table_boxes(page: pymupdf.Page) -> list[list[int]]:
    try:
        return [[round(v) for v in t.bbox] for t in page.find_tables()]
    except Exception:
        return []


def replace_tables(backend, page: pymupdf.Page, markdown: str,
                   dpi: int, max_tokens: int) -> tuple[str, int]:
    """用表格区域的精细识别结果替换整页 markdown 里的 <table> 块。"""
    boxes = table_boxes(page)
    if not boxes:
        return markdown, 0

    md_tables = list(TABLE_RE.finditer(markdown))
    out, replaced = markdown, 0
    for idx in range(min(len(boxes), len(md_tables))):
        html = backend.recognize(crop_region(page, boxes[idx], dpi), PROMPT_TABLE, max_tokens)
        if not html.lstrip().lower().startswith("<table"):
            continue
        out = out.replace(md_tables[idx].group(0), html, 1)
        replaced += 1
    return out, replaced


def _parse_one(backend, page: pymupdf.Page, dpi: int,
               max_tokens: int, parse_tables: bool) -> tuple[str, int]:
    image = render_page(page, dpi)
    text = backend.recognize(image, PROMPT_WHOLE_PAGE, max_tokens).strip()
    n = 0
    if parse_tables:
        text, n = replace_tables(backend, page, text, dpi, max_tokens)
    return text, n


def process_pages(backend, doc: pymupdf.Document, indices: list[int],
                  dpi: int, max_tokens: int, parse_tables: bool,
                  workers: int = 1) -> dict[int, tuple[str, int]]:
    """处理若干页，返回 {页码: (文本, 替换的表格数)}。

    workers > 1 且后端是 vllm 时按批并发：渲染在主线程完成（pymupdf 非线程安全），
    识别请求并行发出。
    """
    results: dict[int, tuple[str, int]] = {}

    if workers <= 1 or backend.name != "vllm":
        for i in indices:
            results[i] = _parse_one(backend, doc[i - 1], dpi, max_tokens, parse_tables)
        return results

    for start in range(0, len(indices), workers):
        chunk = indices[start:start + workers]
        # 1) 主线程渲染，避免 pymupdf 并发访问
        images = {i: render_page(doc[i - 1], dpi) for i in chunk}
        crops = {i: [crop_region(doc[i - 1], b, dpi) for b in table_boxes(doc[i - 1])]
                 for i in chunk} if parse_tables else {i: [] for i in chunk}

        # 2) 整页识别并发
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {i: ex.submit(backend.recognize, images[i], PROMPT_WHOLE_PAGE, max_tokens)
                       for i in chunk}
            texts = {i: f.result().strip() for i, f in futures.items()}

            # 3) 表格识别并发，并替换回 markdown
            for i in chunk:
                text, n = texts[i], 0
                if crops[i]:
                    tf = [ex.submit(backend.recognize, c, PROMPT_TABLE, max_tokens)
                          for c in crops[i]]
                    htmls = [f.result().strip() for f in tf]
                    md_tables = list(TABLE_RE.finditer(text))
                    for idx, html in enumerate(htmls[:len(md_tables)]):
                        if html.lstrip().lower().startswith("<table"):
                            text = text.replace(md_tables[idx].group(0), html, 1)
                            n += 1
                results[i] = (text, n)
    return results


def parse_pageset(pages: str, total: int) -> list[int]:
    """解析 `1-3,7` 这类页码表达式。"""
    if not pages:
        return list(range(1, total + 1))
    wanted: set[int] = set()
    for part in pages.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))
    return [i for i in range(1, total + 1) if i in wanted]


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF → Markdown（OXR）")
    ap.add_argument("pdf", type=Path, help="输入 PDF")
    ap.add_argument("-o", "--output", type=Path, help="输出 .md（默认与 PDF 同名）")
    ap.add_argument("--dpi", type=int, default=150, help="渲染 DPI，默认 150")
    ap.add_argument("--max-tokens", type=int, default=2048, help="每页最大生成 token")
    ap.add_argument("--pages", type=str, default="", help="只处理指定页，如 1-3,7")
    ap.add_argument("--backend", choices=["local", "vllm"], default="local",
                    help="local=进程内加载；vllm=调用已启动的 vLLM 服务")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1", help="vLLM 服务地址")
    ap.add_argument("--workers", type=int, default=1, help="并发页数（仅 vllm 后端生效）")
    ap.add_argument("--parse-tables", action="store_true",
                    help="表格区域单独精细识别，替换整页结果里的 <table>")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"文件不存在: {args.pdf}")

    doc = pymupdf.open(args.pdf)
    indices = parse_pageset(args.pages, doc.page_count)
    out_path = args.output or args.pdf.with_suffix(".md")

    t0 = time.time()
    backend = make_backend(args.backend, args.base_url)
    workers = args.workers if backend.name == "vllm" else 1
    print(f"输入: {args.pdf}（{doc.page_count} 页，处理 {len(indices)} 页，"
          f"{args.dpi} DPI，后端 {backend.name}，并发 {workers}）")

    results = process_pages(backend, doc, indices, args.dpi, args.max_tokens,
                            args.parse_tables, workers)

    parts = [f"# {args.pdf.stem}\n"]
    for i in indices:
        text, n = results[i]
        extra = f"，表格 {n}" if args.parse_tables else ""
        print(f"  第 {i}/{doc.page_count} 页：{len(text)} 字符{extra}")
        parts.append(f"\n<!-- 第 {i} 页 -->\n\n{text}\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"完成：{out_path}（{time.time() - t0:.1f}s 总计）")


if __name__ == "__main__":
    main()
