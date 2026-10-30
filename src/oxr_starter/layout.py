"""版面区域检测 + 表格精细识别。

背景：OXR 本体**不输出布局框**——用官方 layout 提示词实测，模型仍返回整页内容。
真正的版面检测由独立的 D-FINE 模型负责（`layout_analysis/last.pth`，需要官方
38 个文件的 FastAPI 服务）。本项目改用 PyMuPDF 的结构信息做区域检测，
对表格区域单独用官方表格提示词识别，从而提升表格准确率。

用法：
    uv run python -m oxr_starter.layout doc.pdf                 # 打印检测到的区域
    uv run python -m oxr_starter.layout doc.pdf --parse-tables  # 只识别表格区域
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = "SoMarkAI/OXR-1.0"
PROMPT_TABLE = "Recognize the table of the image into html format."


def to_cuda(x):
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return torch.cat(list(x), dim=0).to("cuda")
    return x.to("cuda")


def detect_regions(page: pymupdf.Page) -> list[dict]:
    """用 PyMuPDF 的结构信息列出区域：文本块、图片、表格。"""
    regions: list[dict] = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _no, btype = block[:7]
        regions.append({
            "type": "text" if btype == 0 else "image",
            "box": [round(x0), round(y0), round(x1), round(y1)],
            "preview": (text or "").strip().replace("\n", " ")[:60],
        })

    try:
        for tab in page.find_tables():
            regions.append({
                "type": "table",
                "box": [round(v) for v in tab.bbox],
                "preview": f"{len(tab.rows)} 行 × {len(tab.header.names) if tab.header else 0} 列",
            })
    except Exception:
        pass

    regions.sort(key=lambda r: (r["box"][1], r["box"][0]))  # 按阅读顺序从上到下
    return regions


def crop(page: pymupdf.Page, box: list[int], dpi: int) -> Image.Image:
    rect = pymupdf.Rect(*box)
    pix = page.get_pixmap(dpi=dpi, clip=rect)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def recognize_table(model, image: Image.Image, max_tokens: int = 1024) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_TABLE},
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


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF 区域检测 / 表格识别")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--page", type=int, default=1, help="页码，默认第 1 页")
    ap.add_argument("--dpi", type=int, default=200, help="裁剪区域渲染 DPI")
    ap.add_argument("--parse-tables", action="store_true", help="对表格区域做 OXR 识别")
    ap.add_argument("--json", action="store_true", help="输出区域 JSON")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    page = doc[args.page - 1]
    regions = detect_regions(page)

    if args.json:
        print(json.dumps({"page": args.page, "regions": regions},
                         ensure_ascii=False, indent=2))
    else:
        print(f"{args.pdf.name} 第 {args.page} 页：{len(regions)} 个区域")
        for r in regions:
            print(f"  {r['type']:<6} {r['box']}  {r.get('preview', '')}")

    tables = [r for r in regions if r["type"] == "table"]
    if args.parse_tables and tables:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用：请在 systemd 服务里运行")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="cuda:0", low_cpu_mem_usage=True,
        )
        model.eval()
        for i, t in enumerate(tables, 1):
            img = crop(page, t["box"], args.dpi)
            out = recognize_table(model, img)
            print(f"\n--- 表格 {i} {t['box']} （{img.size[0]}x{img.size[1]}）---")
            print(out)


if __name__ == "__main__":
    main()
