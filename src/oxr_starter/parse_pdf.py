"""PDF → Markdown + 可选 JSON 结构化输出。

后端：`--backend local`（进程内 transformers）或 `vllm`（HTTP 服务）。
并发：`--workers N` 仅对 vllm 生效。
断点：每页成功后写入 `<输出>.parts/pNNNN.md`，重跑自动续传。
图片：`--extract-images` 导出原始位图并写入 manifest。
JSON：`--json` 额外产出结构化结果（页、表格 bbox、图片 bbox）。
"""

from __future__ import annotations

import argparse
import json
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

# 每页结果：(markdown 文本, 表格列表 [{"bbox": [...], "html": "..."}])
PageResult = tuple[str, list[dict]]


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
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
                   dpi: int, max_tokens: int) -> PageResult:
    """用表格区域的精细识别结果替换整页 markdown 里的 <table> 块。"""
    boxes = table_boxes(page)
    if not boxes:
        return markdown, []

    md_tables = list(TABLE_RE.finditer(markdown))
    out, tables = markdown, []
    for idx in range(min(len(boxes), len(md_tables))):
        html = backend.recognize(crop_region(page, boxes[idx], dpi), PROMPT_TABLE, max_tokens)
        if not html.lstrip().lower().startswith("<table"):
            continue
        out = out.replace(md_tables[idx].group(0), html, 1)
        tables.append({"bbox": boxes[idx], "html": html})
    return out, tables


def _part_md(parts_dir: Path, page_no: int) -> Path:
    return parts_dir / f"p{page_no:04d}.md"


def _part_json(parts_dir: Path, page_no: int) -> Path:
    return parts_dir / f"p{page_no:04d}.json"


def _parse_one(backend, page: pymupdf.Page, dpi: int,
               max_tokens: int, parse_tables: bool) -> PageResult:
    image = render_page(page, dpi)
    text = backend.recognize(image, PROMPT_WHOLE_PAGE, max_tokens).strip()
    if not parse_tables:
        return text, []
    return replace_tables(backend, page, text, dpi, max_tokens)


def process_pages(backend, doc: pymupdf.Document, indices: list[int],
                  dpi: int, max_tokens: int, parse_tables: bool,
                  workers: int = 1, parts_dir: Path | None = None
                  ) -> tuple[dict[int, PageResult], list[int]]:
    """处理若干页，返回 ({页码: 结果}, 失败页码)。

    parts_dir 非空时启用页码级断点：成功页写入 `pNNNN.md`（表格另存
    `pNNNN.json`），重跑跳过已完成页。单页失败不影响其余页。
    """
    results: dict[int, PageResult] = {}
    failed: list[int] = []
    todo: list[int] = []

    if parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)
        for i in indices:
            md = _part_md(parts_dir, i)
            if not md.exists():
                todo.append(i)
                continue
            text = md.read_text(encoding="utf-8")
            tables = []
            js = _part_json(parts_dir, i)
            if js.exists():
                try:
                    tables = json.loads(js.read_text(encoding="utf-8")).get("tables", [])
                except Exception:
                    tables = []
            results[i] = (text, tables)
    else:
        todo = list(indices)

    def _record(i: int, text: str, tables: list[dict]) -> None:
        results[i] = (text, tables)
        if parts_dir is not None:
            _part_md(parts_dir, i).write_text(text, encoding="utf-8")
            if tables:
                _part_json(parts_dir, i).write_text(
                    json.dumps({"tables": tables}, ensure_ascii=False), encoding="utf-8")

    if not todo:
        return results, failed

    if workers <= 1 or backend.name != "vllm":
        for i in todo:
            try:
                text, tables = _parse_one(backend, doc[i - 1], dpi, max_tokens, parse_tables)
                _record(i, text, tables)
            except Exception as e:
                print(f"  ✗ 第 {i} 页失败: {type(e).__name__}: {e}")
                failed.append(i)
        return results, failed

    for start in range(0, len(todo), workers):
        chunk = todo[start:start + workers]
        images = {i: render_page(doc[i - 1], dpi) for i in chunk}
        crops = {i: [crop_region(doc[i - 1], b, dpi) for b in table_boxes(doc[i - 1])]
                 for i in chunk} if parse_tables else {i: [] for i in chunk}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {i: ex.submit(backend.recognize, images[i], PROMPT_WHOLE_PAGE, max_tokens)
                       for i in chunk}
            texts: dict[int, str] = {}
            for i, f in futures.items():
                try:
                    texts[i] = f.result().strip()
                except Exception as e:
                    print(f"  ✗ 第 {i} 页失败: {type(e).__name__}: {e}")
                    failed.append(i)

            for i in chunk:
                if i not in texts:
                    continue
                text, tables = texts[i], []
                if crops[i]:
                    tf = [ex.submit(backend.recognize, c, PROMPT_TABLE, max_tokens)
                          for c in crops[i]]
                    htmls = []
                    for f in tf:
                        try:
                            htmls.append(f.result().strip())
                        except Exception:
                            htmls.append("")
                    md_tables = list(TABLE_RE.finditer(text))
                    for idx, html in enumerate(htmls[:len(md_tables)]):
                        if html.lstrip().lower().startswith("<table"):
                            text = text.replace(md_tables[idx].group(0), html, 1)
                            tables.append({"bbox": table_boxes(doc[i - 1])[idx], "html": html})
                _record(i, text, tables)
    return results, failed


def parse_pageset(pages: str, total: int) -> list[int]:
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


def build_json(pdf_path: Path, indices: list[int], results: dict[int, PageResult],
               images: dict[int, list[dict]], meta: dict) -> dict:
    pages = []
    for i in indices:
        text, tables = results.get(i, ("", []))
        pages.append({
            "page": i,
            "markdown": text,
            "tables": tables,
            "images": images.get(i, []),
            "failed": i not in results,
        })
    return {"source": pdf_path.name, "pages": pages, "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF → Markdown（OXR）")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="输出 .md（默认与 PDF 同名）")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--pages", type=str, default="", help="只处理指定页，如 1-3,7")
    ap.add_argument("--backend", choices=["local", "vllm"], default="local")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--device", default="auto", help="local 后端的设备：auto / cuda / mps / cpu")
    ap.add_argument("--dtype", default="auto", help="local 后端的精度：auto / bf16 / fp16 / fp32")
    ap.add_argument("--workers", type=int, default=1, help="并发页数（仅 vllm）")
    ap.add_argument("--parse-tables", action="store_true")
    ap.add_argument("--extract-images", action="store_true")
    ap.add_argument("--json", action="store_true", help="额外输出 <输出>.json 结构化结果")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--clean-parts", action="store_true")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"文件不存在: {args.pdf}")

    doc = pymupdf.open(args.pdf)
    indices = parse_pageset(args.pages, doc.page_count)
    out_path = args.output or args.pdf.with_suffix(".md")
    parts_dir = None if args.no_resume else out_path.parent / f"{out_path.stem}.parts"

    t0 = time.time()
    try:
        backend = make_backend(args.backend, args.base_url, args.device, args.dtype)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    workers = args.workers if backend.name == "vllm" else 1
    dev_info = ""
    if hasattr(backend, "device"):  # LocalBackend 才有设备概念
        dev_info = f"（{backend.device}/{str(backend.dtype).replace('torch.', '')}）"
    print(f"输入: {args.pdf}（{doc.page_count} 页，处理 {len(indices)} 页，"
          f"{args.dpi} DPI，后端 {backend.name}{dev_info}，并发 {workers}）")
    if parts_dir is not None and parts_dir.exists():
        print(f"断点续跑: {parts_dir}（已有 {len(list(parts_dir.glob('p*.md')))} 页）")

    results, failed = process_pages(backend, doc, indices, args.dpi, args.max_tokens,
                                    args.parse_tables, workers, parts_dir)

    img_items: dict[int, list[dict]] = {}
    img_dir = None
    if args.extract_images:
        from .images import extract_document

        img_dir = out_path.parent / f"{out_path.stem}.images"
        items = extract_document(args.pdf, img_dir)
        for it in items:
            img_items.setdefault(it["page"], []).append(it)
        print(f"图片: 提取 {len(items)} 张 → {img_dir}")

    parts = [f"# {args.pdf.stem}\n"]
    missing = []
    for i in indices:
        if i not in results:
            missing.append(i)
            parts.append(f"\n<!-- 第 {i} 页 识别失败，重跑可续传 -->\n")
            continue
        text, tables = results[i]
        extra = f"，表格 {len(tables)}" if args.parse_tables else ""
        print(f"  第 {i}/{doc.page_count} 页：{len(text)} 字符{extra}")
        parts.append(f"\n<!-- 第 {i} 页 -->\n\n{text}\n")
        for it in img_items.get(i, []):
            rel = f"{img_dir.name}/{it['file']}"
            parts.append(f"\n<!-- 图片 {it['file']} bbox={it['bbox']} -->\n"
                         f"![{it['file']}]({rel})\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    elapsed = time.time() - t0
    print(f"完成：{out_path}（{elapsed:.1f}s 总计）")

    if args.json:
        meta = {"dpi": args.dpi, "backend": backend.name, "workers": workers,
                "parse_tables": args.parse_tables, "elapsed_sec": round(elapsed, 1)}
        doc_json = build_json(args.pdf, indices, results, img_items, meta)
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"JSON：{json_path}")

    if failed or missing:
        bad = sorted(set(failed) | set(missing))
        print(f"⚠ {len(bad)} 页未完成: {bad}")
        print(f"  重跑同一条命令即可续传（缓存: {parts_dir}）")
        raise SystemExit(1)
    if args.clean_parts and parts_dir is not None and parts_dir.exists():
        for f in list(parts_dir.glob("p*")):
            f.unlink()
        parts_dir.rmdir()
        print(f"已清理断点缓存: {parts_dir}")


if __name__ == "__main__":
    main()
