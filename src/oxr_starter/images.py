"""从 PDF 页面提取位图图片。

按页导出原始嵌入图片（不是渲染截图，保留原始分辨率和格式），
并生成 JSON 清单（页码、bbox、文件名、尺寸）。

用法：
    uv run python -m oxr_starter.images doc.pdf -o out/images
    uv run python -m oxr_starter.images doc.pdf -o out/images --min-area 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf


def extract_page_images(doc: pymupdf.Document, page_no: int, out_dir: Path,
                        min_area_ratio: float = 0.005,
                        min_side: int = 24) -> list[dict]:
    """提取某页的位图图片，返回清单项。"""
    page = doc[page_no - 1]
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    items: list[dict] = []
    seen: set[int] = set()
    seq = 0

    for info in page.get_images(full=True):
        xref = info[0]
        if xref in seen:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]

        # 过滤小图标、装饰性色块
        if abs(rect.width * rect.height) / page_area < min_area_ratio:
            continue
        if rect.width < min_side or rect.height < min_side:
            continue

        try:
            raw = doc.extract_image(xref)
        except Exception:  # noqa: BLE001, S112 - 损坏的图片直接跳过
            continue
        if not raw or not raw.get("image"):
            continue

        seen.add(xref)
        seq += 1
        ext = raw.get("ext", "png")
        name = f"p{page_no:04d}-img{seq:02d}.{ext}"
        (out_dir / name).write_bytes(raw["image"])
        items.append({
            "page": page_no,
            "file": name,
            "bbox": [round(v, 1) for v in (rect.x0, rect.y0, rect.x1, rect.y1)],
            "width": raw.get("width"),
            "height": raw.get("height"),
            "colorspace": raw.get("colorspace"),
            "bytes": len(raw["image"]),
        })

    return items


def extract_document(pdf_path: Path, out_dir: Path,
                     min_area_ratio: float = 0.005) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    all_items: list[dict] = []
    for page_no in range(1, doc.page_count + 1):
        all_items.extend(extract_page_images(doc, page_no, out_dir, min_area_ratio))
    if all_items:
        (out_dir / "manifest.json").write_text(
            json.dumps({"pdf": pdf_path.name, "images": all_items},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return all_items


def main() -> None:
    ap = argparse.ArgumentParser(description="提取 PDF 中的位图图片")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    ap.add_argument("--min-area", type=float, default=0.005,
                    help="最小面积占比（相对页面），默认 0.5%%")
    args = ap.parse_args()

    items = extract_document(args.pdf, args.output, args.min_area)
    if not items:
        print(f"{args.pdf.name}: 没有可提取的位图（可能是纯矢量文档）")
        return
    print(f"{args.pdf.name}: 提取 {len(items)} 张图片 → {args.output}")
    for it in items:
        print(f"  第 {it['page']} 页 {it['file']}  "
              f"{it['width']}x{it['height']}  {it['bytes']/1024:.0f} KB  bbox={it['bbox']}")


if __name__ == "__main__":
    main()
