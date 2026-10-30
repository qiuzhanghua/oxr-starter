"""PDF → Markdown：逐页渲染后用 OXR 识别。

用法：
    uv run python -m oxr_starter.parse_pdf input.pdf -o out.md
    uv run python -m oxr_starter.parse_pdf input.pdf --dpi 150 --max-tokens 2048

说明：
- 模型只加载一次，逐页推理，避免每页重复加载
- 页面渲染成位图交给模型（OXR 是视觉模型，不做 PDF 文本提取）
- 输出按页加分隔标题，方便对照原文档
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import re

import pymupdf
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = "SoMarkAI/OXR-1.0"
PROMPT_WHOLE_PAGE = "Recognize the whole page content of the image."


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """把 PDF 页渲染成 PIL 图像。"""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def to_cuda(x):
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return torch.cat(list(x), dim=0).to("cuda")
    return x.to("cuda")


PROMPT_TABLE = "Recognize the table of the image into html format."
TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


def _crop(page: pymupdf.Page, box, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*box))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _recognize(model, image: Image.Image, prompt: str, max_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages)
    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids.to("cuda"),
            pixel_values=to_cuda(pixel_values),
            grid_thws=to_cuda(grid_thws),
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    return model.text_tokenizer.decode(out[0], skip_special_tokens=True).strip()


def replace_tables(model, page: pymupdf.Page, markdown: str,
                   dpi: int, max_tokens: int) -> tuple[str, int]:
    """用表格区域的精细识别结果替换整页 markdown 里的 <table> 块。

    返回 (替换后的 markdown, 实际替换个数)。
    """
    try:
        boxes = [[round(v) for v in t.bbox] for t in page.find_tables()]
    except Exception:
        return markdown, 0
    if not boxes:
        return markdown, 0

    tables_in_md = list(TABLE_RE.finditer(markdown))
    replaced = 0
    out = markdown
    for idx in range(min(len(boxes), len(tables_in_md))):
        try:
            html = _recognize(model, _crop(page, boxes[idx], dpi), PROMPT_TABLE, max_tokens)
        except Exception:
            continue
        if not html.lstrip().lower().startswith("<table"):
            continue
        out = out.replace(tables_in_md[idx].group(0), html, 1)
        replaced += 1
    return out, replaced


def parse_page(model, image: Image.Image, max_tokens: int) -> str:
    return _recognize(model, image, PROMPT_WHOLE_PAGE, max_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF → Markdown（OXR）")
    ap.add_argument("pdf", type=Path, help="输入 PDF")
    ap.add_argument("-o", "--output", type=Path, help="输出 .md（默认与 PDF 同名）")
    ap.add_argument("--dpi", type=int, default=150, help="渲染 DPI，默认 150")
    ap.add_argument("--max-tokens", type=int, default=2048, help="每页最大生成 token")
    ap.add_argument("--pages", type=str, default="", help="只处理指定页，如 1-3,7")
    ap.add_argument("--parse-tables", action="store_true",
                    help="表格区域单独精细识别，替换整页结果里的 <table>")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用：请在 systemd 服务里运行")
    if not args.pdf.exists():
        raise SystemExit(f"文件不存在: {args.pdf}")

    doc = pymupdf.open(args.pdf)
    total = doc.page_count
    wanted = set()
    if args.pages:
        for part in args.pages.split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))
    indices = [i for i in range(1, total + 1) if not wanted or i in wanted]

    out_path = args.output or args.pdf.with_suffix(".md")
    print(f"输入: {args.pdf}（{total} 页，处理 {len(indices)} 页，{args.dpi} DPI）")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"模型就绪（{time.time() - t0:.1f}s）")

    parts: list[str] = [f"# {args.pdf.stem}\n"]
    for i in indices:
        t1 = time.time()
        page = doc[i - 1]
        img = render_page(page, args.dpi)
        text = parse_page(model, img, args.max_tokens).strip()
        n_tables = 0
        if args.parse_tables:
            text, n_tables = replace_tables(model, page, text, args.dpi, args.max_tokens)
        dt = time.time() - t1
        extra = f"，替换表格 {n_tables} 个" if args.parse_tables else ""
        print(f"  第 {i}/{total} 页：{img.size[0]}x{img.size[1]}，"
              f"{len(text)} 字符，{dt:.1f}s{extra}")
        parts.append(f"\n<!-- 第 {i} 页 -->\n\n{text}\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"完成：{out_path}（{time.time() - t0:.1f}s 总计）")
    print(f"显存峰值: {round(torch.cuda.max_memory_allocated() / 1024**3, 2)} GB")


if __name__ == "__main__":
    main()
